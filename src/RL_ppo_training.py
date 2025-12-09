import os, argparse, random
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from torch.optim import SGD, AdamW
from tqdm import trange
import wandb

from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from .scorers import COMETQE, LMVerifier, Sentinel
from .constraints import constraint_score
from .losses import method2_loss

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

DEFAULT_BASE = "Qwen/Qwen3-0.6B-Base"
DEFAULT_INSTRUCTION = (
    """Rewrite this sentence to be extremely difficult for machine translation using idioms, ambiguity, and wordplay, while keeping it grammatically correct English: "{seed}"

    Only return the single edited sentence."""
)

def read_seeds(path: str, k: int | None = None) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        sents = [l.strip() for l in f if l.strip()]
    return sents[:k] if k else sents

def filter_by_word_count(input):
    word_count = len(input['text'].split())
    return word_count <= 200


@dataclass
class Method2Rewarder:
    """
    Computes reward = -L using:
      - delta difficulty from COMET QE (reference-free)
      - constraint score
      - verifier (LM perplexity proxy -> higher better)
    Now computes the baseline (de_old) per batch instead of globally.
    """
    qe: Sentinel
    vf: LMVerifier
    x: float = 1.0
    y: float = 0.3
    z: float = 0.3
    f: str = "relu"

    @torch.inference_mode()
    def reward(
        self,
        original_seeds: List[str],   # s0
        edits: List[str],            # s1
    ):
        """
        original_seeds: the unedited sentences (batch)
        edits:          the model's rewritten sentences (batch)
        """
        # baseline difficulty
        de_old = self.qe.difficulty(original_seeds)

        # new difficulty
        de_new = self.qe.difficulty(edits)

        cons = constraint_score(edits)
        ver = self.vf.score(edits)

        L = method2_loss(de_new, de_old, cons, ver,
                         x=self.x, y=self.y, z=self.z, f=self.f)

        delta = de_new - np.array(de_old)  # difficulty increase

        # PPO wants high reward
        return [float(l) for l in L], cons, ver, delta.tolist()


def build_prompts(seeds: List[str], tokenizer) -> List[str]:
    """
    Build prompts.
    Returns list of formatted prompt strings ready for tokenization.
    """
    prompts = []
    for seed in seeds:
        prompt = DEFAULT_INSTRUCTION.format(seed=seed)
        prompts.append(prompt)
    return prompts

class PromptDataset(Dataset):
    def __init__(self, seeds, tokenizer):
        """
        seeds: list of strings (plain sentences)
        tokenizer: HF tokenizer (we only use it for build_prompts)
        """
        self.original_seeds = list(seeds)  # keep original seeds for rewarder
        self.prompts = build_prompts(self.original_seeds, tokenizer)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # We return both the prompt and the index so the rewarder
        # can look up de_old[idx] later.
        return {
            "prompt": self.prompts[idx],
            "idx": idx,
        }
    
def freeze_layers(model, trainable_layers: int, freeze_embeddings: bool = True):
    """
    Freeze early layers of the model, keeping only the last N transformer layers trainable.

    This is beneficial for fine-tuning because:
    - Early layers learn general language features that transfer well
    - Later layers learn task-specific features that need adaptation
    - Training fewer parameters reduces overfitting and memory usage

    Args:
        model: AutoModelForCausalLMWithValueHead instance
        trainable_layers: Number of last transformer layers to keep trainable (0 = train all)
        freeze_embeddings: Whether to freeze embedding layers

    Returns:
        tuple: (total_params, trainable_params, frozen_params)
    """
    if trainable_layers == 0:
        # Train all parameters
        for param in model.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        return total, total, 0

    # Get the base model (pretrained_model in AutoModelForCausalLMWithValueHead)
    base_model = model.pretrained_model

    # First, freeze all parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # Find transformer layers - handle different model architectures
    # Common naming conventions: model.layers, model.h, transformer.h, etc.
    transformer_layers = None
    layer_attr_names = ['layers', 'h', 'block', 'blocks']

    # Navigate to the actual model (might be wrapped)
    inner_model = base_model
    if hasattr(inner_model, 'model'):
        inner_model = inner_model.model
    if hasattr(inner_model, 'transformer'):
        inner_model = inner_model.transformer

    for attr_name in layer_attr_names:
        if hasattr(inner_model, attr_name):
            transformer_layers = getattr(inner_model, attr_name)
            break

    if transformer_layers is None:
        print("Warning: Could not find transformer layers. Training all parameters.")
        for param in base_model.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        return total, total, 0

    num_layers = len(transformer_layers)
    layers_to_train = min(trainable_layers, num_layers)
    start_layer = num_layers - layers_to_train

    print(f"Model has {num_layers} transformer layers")
    print(f"Freezing layers 0-{start_layer-1}, training layers {start_layer}-{num_layers-1}")

    # Unfreeze the last N layers
    for i, layer in enumerate(transformer_layers):
        if i >= start_layer:
            for param in layer.parameters():
                param.requires_grad = True

    # Unfreeze LM head (output projection) - critical for generation quality
    if hasattr(base_model, 'lm_head'):
        for param in base_model.lm_head.parameters():
            param.requires_grad = True
        print("LM head unfrozen")

    # Optionally unfreeze embeddings
    if not freeze_embeddings:
        # Handle different embedding attribute names
        embed_attrs = ['embed_tokens', 'wte', 'word_embeddings', 'embeddings']
        for attr in embed_attrs:
            if hasattr(inner_model, attr):
                for param in getattr(inner_model, attr).parameters():
                    param.requires_grad = True
                print(f"Embeddings ({attr}) unfrozen")
                break

    # The value head (added by TRL) should always be trainable
    if hasattr(model, 'v_head'):
        for param in model.v_head.parameters():
            param.requires_grad = True
        print("Value head unfrozen")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nParameter summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    return total_params, trainable_params, frozen_params


class huggingfaceDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        """
        hf_dataset: Huggingface dataset object (streaming or not)
        tokenizer: HF tokenizer (we only use it for build_prompts)
        """
        self.hf_dataset = hf_dataset
        self.original_seeds = hf_dataset['text']
        self.prompts = build_prompts(self.original_seeds, tokenizer)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # We return both the prompt and the index so the rewarder
        # can look up de_old[idx] later.
        return {
            "prompt": self.prompts[idx],
            "idx": idx,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="data/seeds.txt")
    ap.add_argument("--dataset", type=str, default="agentlans/high-quality-english-sentences", help="optional dataset(pass empty string for seeds.txt)")
    ap.add_argument("--k", type=int, default=500, help="num seeds to use")
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate of the optimizer")
    ap.add_argument("--base_model", type=str, default=DEFAULT_BASE)
    ap.add_argument("--save_dir", type=str, default="checkpoints/gpt2-ppo-method2")
    ap.add_argument("--steps", type=int, default=200)           # PPO update steps
    ap.add_argument("--batch_size", type=int, default=2)        # prompts per PPO step (reduced for memory efficiency with Qwen3 + PPO's 2x model requirement)
    ap.add_argument("--gen_max_new_tokens", type=int, default=64, help="Max tokens to generate. Reduced for memory efficiency with PPO's 2x model requirement. 64-128 is reasonable for sentence rewriting.")
    ap.add_argument("--top_p", type=float, default=0.8, help="Qwen3 recommended: 0.95 for thinking mode, 0.8 for non-thinking")
    ap.add_argument("--temperature", type=float, default=0.7, help="Qwen3 recommended: 0.6 for thinking mode, 0.7 for non-thinking")
    ap.add_argument("--mt_device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for MT translator (defaults to CPU to save VRAM)")
    ap.add_argument("--lm_verifier_device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for LM verifier scorer")
    ap.add_argument("--comet_accelerator", type=str, default="gpu", choices=["cpu", "gpu"], help="Accelerator used by COMET QE scorer")
    # loss weights
    ap.add_argument("--x", type=float, default=1.0) # Weight on difficulty delta
    ap.add_argument("--y", type=float, default=0.3) # Weight on constraints
    ap.add_argument("--z", type=float, default=0.3) # Weight on verifier
    ap.add_argument("--f", type=str, default="none", choices=["relu","sigmoid","none"])
    ap.add_argument("--seed", type=int, default=42)
    # layer freezing
    ap.add_argument("--trainable_layers", type=int, default=4, help="Number of last transformer layers to train (0=train all, default=4)")
    ap.add_argument("--freeze_embeddings", action="store_true", help="Freeze embedding layers (recommended for partial fine-tuning)")
    # wandb
    ap.add_argument("--wandb_project", type=str, default="mt-breaker-ppo")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Args parsed.")

    # ---- initialize wandb ----
    use_wandb = not args.no_wandb
    if use_wandb:
        if os.path.exists("wandb.key"):
            with open("wandb.key") as f:
                wandb.login(key=f.read().strip())
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "base_model": args.base_model,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "gen_max_new_tokens": args.gen_max_new_tokens,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "loss_x": args.x,
                "loss_y": args.y,
                "loss_z": args.z,
                "loss_f": args.f,
                "seed": args.seed,
                "num_seeds": args.k,
                "ppo_epochs": 4,
                "init_kl_coef": 0.2,
                "target_kl": 0.1,
                "cliprange": 0.2,
                "trainable_layers": args.trainable_layers,
                "freeze_embeddings": args.freeze_embeddings,
            }
        )

    # ---- policy + ref model (needed for tokenizer to build prompts) ----
    tok = AutoTokenizer.from_pretrained(args.base_model)
    # Qwen3 has its own pad token - only set it if missing (e.g., for GPT-2)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # Required for batch generation with causal LMs

    # ---- data ----
    if args.dataset:
        hs_dataset = load_dataset(args.dataset, num_proc=4, split = "train")
        hs_dataset = hs_dataset.filter(filter_by_word_count, num_proc=4)
        dataset = huggingfaceDataset(hs_dataset, tok)

    else :
        dataset = PromptDataset(read_seeds(args.seeds, k=args.k), tok)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # ---- external scorers ----
    qe = Sentinel()
    vf = LMVerifier(device=args.lm_verifier_device)

    rewarder = Method2Rewarder(
        qe=qe, vf=vf,
        x=args.x, y=args.y, z=args.z, f=args.f
    )

    # ---- policy + ref model ----
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear cache before loading models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before model load: {torch.cuda.memory_allocated(device)/1e9:.2f} GB allocated, {torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved")

    # Load model in FP32 - mixed precision training will handle memory efficiency
    # Loading in FP16 directly conflicts with GradScaler which expects FP32 gradients
    # Mixed precision will automatically convert to FP16 during forward pass while keeping gradients in FP32
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model,
        # Don't specify torch_dtype - let it load in default precision (FP32)
        # Mixed precision training will handle the conversion for memory efficiency
    )

    # DEBUG: Check if weights are actually loaded
    print(f"Model loaded. Checking first layer weights...")
    first_param = next(policy.pretrained_model.parameters())
    print(f"First param shape: {first_param.shape}, mean: {first_param.mean().item():.6f}, std: {first_param.std().item():.6f}")
    print(f"First param is on device: {first_param.device}")

    # Explicitly move to device
    if torch.cuda.is_available():
        policy = policy.to(device)
        print(f"GPU memory after policy load: {torch.cuda.memory_allocated(device)/1e9:.2f} GB allocated, {torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved")

    # Apply layer freezing for partial fine-tuning
    # This keeps early layers frozen (preserving general language knowledge)
    # while only training the last few layers for task adaptation
    print("\n" + "="*80)
    print("APPLYING LAYER FREEZING FOR PARTIAL FINE-TUNING")
    print("="*80)
    total_params, trainable_params, frozen_params = freeze_layers(
        policy,
        trainable_layers=args.trainable_layers,
        freeze_embeddings=args.freeze_embeddings
    )
    print("="*80 + "\n")

    # Log parameter counts to wandb
    if use_wandb:
        wandb.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 1.0,
        })

    # Enable gradient checkpointing to save memory during training
    if hasattr(policy.pretrained_model, 'gradient_checkpointing_enable'):
        policy.pretrained_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # TRL will create a frozen reference model internally (this doubles memory usage!)
    # For Qwen3-0.6B: ~1.2B parameters total (2x 0.6B), with FP32 weights this is ~4.8GB
    # Plus activations, gradients, optimizer states, and generation buffers = easily 10-11GB on 1080Ti

    PPO_EPOCHS = 2
    # Only optimize trainable parameters (after layer freezing)
    trainable_params_list = [p for p in policy.parameters() if p.requires_grad]
    # optimizer = SGD(trainable_params_list, lr=args.lr)
    optimizer = AdamW(trainable_params_list, lr=args.lr)
    print(f"Optimizer created with {len(trainable_params_list)} parameter groups")
    # num_training_steps = args.steps * PPO_EPOCHS
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=5,
    #     num_training_steps=num_training_steps,
    # )

    ppo_config = PPOConfig(
        model_name=args.base_model,
        learning_rate=args.lr,            # Lower LR for more stable training
        batch_size=args.batch_size,    # samples used per PPO step
        mini_batch_size=1,             # Process one sample at a time to minimize memory usage
        gradient_accumulation_steps=1,
        ppo_epochs=PPO_EPOCHS,                  # Reduced from 4 to save memory (fewer forward/backward passes)
        cliprange=0.2,
        cliprange_value=0.2,           # Clip value function updates
        vf_coef=0.3,                   # Value function coefficient
        kl_penalty="kl",
        init_kl_coef=0.2,              # Higher initial KL penalty (was 0.05)
        target_kl=0.1,                 # Lower target KL to prevent divergence (was 0.15)
        adap_kl_ctrl=True,             # Adaptive KL control
        seed=args.seed,
        accelerator_kwargs={
            "device_placement": True,
            "mixed_precision": "fp16" if torch.cuda.is_available() else "no"
        },
        log_with=None,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        tokenizer=tok,
        optimizer=optimizer,
        # lr_scheduler=lr_scheduler,
    )
    
    # After PPOTrainer creation, check memory again (should be ~2x due to reference model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear any fragmentation
        print(f"GPU memory after PPOTrainer init: {torch.cuda.memory_allocated(device)/1e9:.2f} GB allocated, {torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved")
        print(f"Max GPU memory: {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB")

    gen_kwargs = dict(
        do_sample=True,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.gen_max_new_tokens,
        min_new_tokens=2,  # Ensure at least 2 tokens to avoid masking issues
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        top_k=30,  # Increased from 20 to allow more vocabulary diversity
        repetition_penalty=1.1,  # Penalize repetitions to avoid repeating prompt
        no_repeat_ngram_size=3,  # Prevent repeating 3-grams (helps avoid instruction repetition)
    )

    # ---- Test generation BEFORE training ----
    print("\n" + "="*80)
    print("TESTING MODEL GENERATION BEFORE TRAINING")
    print("="*80)
    test_prompt = DEFAULT_INSTRUCTION.format(seed="The cat sat on the mat.")
    test_ids = tok(test_prompt, return_tensors="pt").input_ids.squeeze(0).to(trainer.accelerator.device)
    policy.eval()  # Set to eval mode
    with torch.no_grad():
        test_output = policy.generate(test_ids.unsqueeze(0), max_new_tokens=50, do_sample=False)
        test_response = tok.decode(test_output[0][len(test_ids):], skip_special_tokens=True)
    print(f"Test prompt: {test_prompt}")
    print(f"Test output: {test_response}")
    print("="*80 + "\n")
    policy.train()  # Set back to train mode

    # ---- PPO loop ----
    n = len(dataset)
    print(f"Starting PPO: seeds={len(dataset)}, steps={args.steps}, batch={args.batch_size}")

    data_iter = iter(train_loader)

    for step in trange(args.steps, desc="PPO"):
        # Get next batch from DataLoader, loop back to start if exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        prompts = batch["prompt"]              # list of strings
        idx = batch["idx"].tolist()            # list of ints, for de_old indexing

        original_batch = [dataset.original_seeds[i] for i in idx]

        # 1) tokenize prompts
        query_tensors = [tok(p, return_tensors="pt").input_ids.squeeze(0) for p in prompts]
        query_tensors = [q.to(trainer.accelerator.device) for q in query_tensors]

        # 2) generate responses
        # For causal models like Qwen3, we need to extract only the generated part (excluding the prompt)
        response_tensors = []
        # Clear cache before generation to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # CRITICAL: Set model to eval mode during generation to disable dropout/layernorm randomness
        trainer.model.eval()
        with torch.no_grad():
            for query in query_tensors:
                gen_output = trainer.generate(query, **gen_kwargs)
                # For causal models, extract only the generated tokens (excluding the input prompt)
                input_len = query.shape[0]
                response = gen_output.squeeze()[input_len:]
                """---------------------------------------------------------------------------------
                CHECK THIS FALLBACK!!!
                ---------------------------------------------------------------------------------"""
                # Handle empty or very short responses - regenerate with higher temperature or use fallback
                # Need at least 2 tokens to avoid masking issues in PPO trainer
                max_retries = 3
                retry_count = 0
                while len(response) < 2 and retry_count < max_retries:
                    # Generate at least something with higher temperature
                    fallback_kwargs = {**gen_kwargs, "temperature": min(1.5 + retry_count * 0.2, 2.0), "min_new_tokens": 5}
                    gen_output = trainer.generate(query, **fallback_kwargs)
                    response = gen_output.squeeze()[input_len:]
                    retry_count += 1

                # If still too short after retries, create a minimal valid response
                if len(response) < 2:
                    # Create a minimal response with at least 2 tokens
                    fallback_text = "The sentence is difficult."
                    response = tok(fallback_text, return_tensors="pt").input_ids.squeeze(0).to(trainer.accelerator.device)

                response_tensors.append(response)
                
                # Clear cache after each generation to prevent memory accumulation
                if torch.cuda.is_available() and len(response_tensors) % 2 == 0:  # Clear every 2 generations
                    torch.cuda.empty_cache()

        # Set model back to train mode for PPO update
        trainer.model.train()

        # 3) decode responses for reward computation
        cleaned = [tok.decode(r, skip_special_tokens=True).strip() for r in response_tensors]

        # Final safeguard: replace any empty strings with a minimal fallback
        cleaned = [s if s else "." for s in cleaned]

        # 4) compute rewards (external scorers)
        rewards_list, con_rewards, ver_rewards, diff_rewards = rewarder.reward(
            original_batch,   # s0: original sentences
            cleaned,          # s1: edited sentences
        )
        # Move rewards to the same device as query/response tensors (GPU)
        rewards = [torch.tensor(r, device=trainer.accelerator.device) for r in rewards_list]

        # 5) feed PPO step (query tensors, response tensors, reward tensors)
        # Clear cache before PPO step to ensure maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stats = trainer.step(query_tensors, response_tensors, rewards)
        # Clear cache after PPO step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 6) Log metrics to wandb and console
        mean_reward = sum(rewards_list) / len(rewards_list)
        min_reward = min(rewards_list)
        max_reward = max(rewards_list)
        con_reward_mean = sum(con_rewards) / len(con_rewards)
        ver_reward_mean = sum(ver_rewards) / len(ver_rewards)
        diff_reward_mean = sum(diff_rewards) / len(diff_rewards)

        # Compute response length statistics
        response_lengths = [len(r) for r in response_tensors]
        mean_response_len = sum(response_lengths) / len(response_lengths)

        # Prepare wandb logging dict with rewards and generation stats
        log_dict = {
            "reward/mean": mean_reward,
            "reward/min": min_reward,
            "reward/max": max_reward,
            "reward/constraint_mean": con_reward_mean,
            "reward/verifier_mean": ver_reward_mean,
            "reward/difficulty_delta_mean": diff_reward_mean,
            "generation/mean_response_length": mean_response_len,
            "generation/sample_text": wandb.Html(f"<pre>{cleaned[0]}</pre>"),
        }

        # Add all PPO stats from trainer - they come with their own prefixes
        # Filter out NaN values and convert tensors/arrays to scalars
        if isinstance(stats, dict):
            for key, value in stats.items():
                # Convert tensors/arrays to Python scalars and filter NaN
                if torch.is_tensor(value):
                    val = value.item() if value.numel() == 1 else value.mean().item()
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    # Handle numpy arrays or lists
                    val = float(np.mean(value)) if len(value) > 0 else 0.0
                else:
                    val = float(value)

                # Only log if not NaN or inf
                if not (torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val))):
                    log_dict[key] = val

        if use_wandb:
            wandb.log(log_dict, step=step)

        # Extract specific stats for console logging
        ppo_loss = stats.get('ppo/loss/total', stats.get('loss', 0.0)) if isinstance(stats, dict) else 0.0
        kl_div = stats.get('objective/kl', 0.0) if isinstance(stats, dict) else 0.0

        # Log statistics periodically to console
        if (step + 1) % 10 == 0:
            print(f"\nStep {step+1}: reward={mean_reward:.3f}, kl={kl_div:.3f}, loss={ppo_loss:.3f}")
            print(f"Sample output: {cleaned[0][:100]}...")

        if (step + 1) % 20 == 0:
            trainer.save_pretrained(args.save_dir)
            torch.cuda.empty_cache()

    # final save
    trainer.save_pretrained(args.save_dir)
    print(f"Saved PPO policy to: {args.save_dir}")

    # tiny sample after training
    demo_seed = "He saw her duck."
    demo_instruction = DEFAULT_INSTRUCTION.format(seed=demo_seed)
    demo_messages = [{"role": "user", "content": demo_instruction}]
    demo_text = tok.apply_chat_template(
        demo_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    ids = tok([demo_text], return_tensors="pt").to(trainer.accelerator.device)
    with torch.no_grad():
        gen = trainer.model.generate(**ids, **gen_kwargs)
    # Extract only the generated part (excluding the prompt)
    input_len = ids.input_ids.shape[1]
    generated_ids = gen[0][input_len:]
    final_sample = tok.decode(generated_ids, skip_special_tokens=True)
    print("Sample:", final_sample)

    # Log final model to wandb
    if use_wandb:
        # Log final sample
        wandb.log({
            "final/sample_output": wandb.Html(f"<pre>{final_sample}</pre>"),
        })

        # Save model as artifact
        '''artifact = wandb.Artifact(
            name=f"ppo-model-{wandb.run.id}",
            type="model",
            description=f"PPO fine-tuned {args.base_model} for MT-breaking"
        )
        artifact.add_dir(args.save_dir)
        wandb.log_artifact(artifact)'''

        wandb.finish()


if __name__ == "__main__":
    main()
