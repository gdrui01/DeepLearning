import os, argparse, random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch.optim import SGD
from tqdm import trange
import wandb

from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

#from .translate import MTEnDe
from .scorers import COMETQE, LMVerifier, Sentinel
from .constraints import constraint_score
from .losses import method2_loss, RewardNormalizer

PPO_EPOCHS = 4
LR = 5e-5 # 1e-4

DEFAULT_BASE = "Qwen/Qwen3-0.6B-Base"

# DEFAULT_INSTRUCTION = (
#     """Rewrite this sentence to be extremely difficult for machine translation using idioms, ambiguity, and wordplay, while keeping it grammatically correct English: "{seed}"

#     Only return the single edited sentence."""
# )
# DEFAULT_INSTRUCTION = (
#     """
#     Rewrite this sentence to be extremely difficult for machine translation using idioms, ambiguity, and wordplay while keeping it grammatically correct English, returning only the single edited sentence: "{seed}"
#     Do not output anything else, except for the edited sentence.
#     Do not give an explanation.
#     """
# )
DEFAULT_INSTRUCTION = (
    """
    Rewrite the following sentence to be extremely difficult for machine translation using idioms, ambiguity, and wordplay while keeping it grammatically correct English. Return only plain text without any formatting, markdown, or special characters. Output only a single edited sentence:

    "{seed}"
    """
)
# # Could be a better prompt
# DEFAULT_INSTRUCTION = (
#     """Easy sentence: "The weather is nice today."
#     Difficult sentence: "The whether of whether or not to weather the storm is up in the air."

#     Easy sentence: "He went to the bank."
#     Difficult sentence: "He went to the bank, though whether for money or a riverbank remained unclear."

#     Easy sentence: "{seed}"
#     Difficult sentence:"""
# )


def read_seeds(path: str, k: int | None = None) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        sents = [l.strip() for l in f if l.strip()]
    return sents[:k] if k else sents


@dataclass
class Method2Rewarder:
    """
    Computes reward = -L using:
      - delta difficulty from COMET QE (reference-free)
      - constraint score
      - verifier (LM perplexity proxy -> higher better)
    We precompute de_old = 1 - QE(s0, t0) for all seeds once to save time.
    """
    seeds: List[str]
    de_old: List[float]
    #mt: MTEnDe
    qe: Sentinel #COMETQE
    vf: LMVerifier
    x: float = 1.0
    y: float = 0.3
    z: float = 0.3
    f: str = "relu"
    constraint_threshold: float = 0.5
    constraint_sharpness: float = 10.0
    normalize_rewards: bool = False
    clip_normalized: float = 3.0
    delta_normalizer: Optional[RewardNormalizer] = None
    verifier_normalizer: Optional[RewardNormalizer] = None
    constraint_normalizer: Optional[RewardNormalizer] = None

    def __post_init__(self):
        """Initialize normalizers if normalization is enabled."""
        if self.normalize_rewards:
            if self.delta_normalizer is None:
                self.delta_normalizer = RewardNormalizer()
            if self.verifier_normalizer is None:
                self.verifier_normalizer = RewardNormalizer()
            if self.constraint_normalizer is None:
                self.constraint_normalizer = RewardNormalizer()

    @torch.inference_mode()
    def reward(self, edits: List[str], indices: List[int]) -> List[float]:
        """
        Compute rewards for the given edits.
        indices: which seeds these edits correspond to (for selecting de_old)
        """
        #t1 = self.mt.translate(edits)
        de_new = self.qe.difficulty(edits) #self.qe.difficulty(edits, t1)
        cons = constraint_score(edits)
        ver = self.vf.score(edits)
        # Select the corresponding de_old values for this batch
        de_old_batch = [self.de_old[i] for i in indices]

        # Prepare normalizers tuple if normalization is enabled
        normalizers = None
        if self.normalize_rewards:
            normalizers = (self.delta_normalizer, self.verifier_normalizer, self.constraint_normalizer)

        L, delta = method2_loss(
            de_new, de_old_batch, cons, ver,
            x=self.x, y=self.y, z=self.z, f=self.f,
            normalizers=normalizers,
            normalize_rewards=self.normalize_rewards,
            clip_normalized=self.clip_normalized if self.normalize_rewards else None,
            constraint_threshold=self.constraint_threshold,
            constraint_sharpness=self.constraint_sharpness
        )
        # PPO wants rewards (higher is better)
        return [float(l) for l in L], cons, ver, delta.tolist()


def clean_markdown(text: str) -> str:
    """Remove markdown formatting like **bold** and *italic*."""
    import re
    # Remove bold (**text**)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    # Remove italic (*text*)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    return text


def build_prompts(seeds: List[str], tokenizer) -> List[str]:
    """
    Build prompts using Qwen3's chat template format.
    Returns list of formatted prompt strings ready for tokenization.
    """
    prompts = []
    for seed in seeds:
        instruction = DEFAULT_INSTRUCTION.format(seed=seed)
        # messages = [{"role": "user", "content": instruction}]
        # # Apply chat template with thinking disabled for efficiency during PPO training
        # formatted_prompt = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        #     enable_thinking=False  # Disable thinking mode for faster generation during PPO
        # )
        prompts.append(instruction)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="data/seeds.txt")
    ap.add_argument("--k", type=int, default=500, help="num seeds to use")
    ap.add_argument("--base_model", type=str, default=DEFAULT_BASE)
    ap.add_argument("--save_dir", type=str, default="checkpoints/gpt2-ppo-method2")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from")
    ap.add_argument("--steps", type=int, default=200)           # PPO update steps
    ap.add_argument("--batch_size", type=int, default=2)        # prompts per PPO step (reduced for memory efficiency with Qwen3 + PPO's 2x model requirement)
    ap.add_argument("--gen_max_new_tokens", type=int, default=45, help="Max tokens to generate (~30 words at 1.5 tokens/word). Aligned with constraint_score max_len=30 words.")
    ap.add_argument("--top_p", type=float, default=0.8, help="Qwen3 recommended: 0.95 for thinking mode, 0.8 for non-thinking")
    ap.add_argument("--temperature", type=float, default=0.7, help="Qwen3 recommended: 0.6 for thinking mode, 0.7 for non-thinking")
    ap.add_argument("--mt_device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for MT translator (defaults to CPU to save VRAM)")
    ap.add_argument("--lm_verifier_device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for LM verifier scorer")
    ap.add_argument("--comet_accelerator", type=str, default="gpu", choices=["cpu", "gpu"], help="Accelerator used by COMET QE scorer")
    # loss weights
    ap.add_argument("--x", type=float, default=5.0)
    ap.add_argument("--y", type=float, default=1.0)
    ap.add_argument("--z", type=float, default=0.3)
    ap.add_argument("--f", type=str, default="none", choices=["relu","sigmoid","none"])
    # constraint gating
    ap.add_argument("--constraint_threshold", type=float, default=0.5, help="Threshold for soft gating of difficulty reward (default 0.5)")
    ap.add_argument("--constraint_sharpness", type=float, default=10.0, help="Sharpness of sigmoid gate - higher = sharper transition (default 10.0)")
    # reward normalization
    ap.add_argument("--normalize_rewards", action="store_true", help="Enable reward normalization (z-score)")
    ap.add_argument("--clip_normalized", type=float, default=3.0, help="Clip normalized rewards to [-N, +N] std devs")
    ap.add_argument("--seed", type=int, default=42)
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
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "base_model": args.base_model,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": LR,
                "gen_max_new_tokens": args.gen_max_new_tokens,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "loss_x": args.x,
                "loss_y": args.y,
                "loss_z": args.z,
                "loss_f": args.f,
                "constraint_threshold": args.constraint_threshold,
                "constraint_sharpness": args.constraint_sharpness,
                "normalize_rewards": args.normalize_rewards,
                "clip_normalized": args.clip_normalized,
                "seed": args.seed,
                "num_seeds": args.k,
                "ppo_epochs": 4,
                "init_kl_coef": 0.2,
                "target_kl": 0.1,
                "cliprange": 0.2,
            }
        )

    # ---- policy + ref model (needed for tokenizer to build prompts) ----
    tok = AutoTokenizer.from_pretrained(args.base_model)
    # Qwen3 has its own pad token - only set it if missing (e.g., for GPT-2)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # Required for batch generation with causal LMs

    # Get asterisk token IDs to suppress markdown formatting
    asterisk_token_ids = tok.encode('*', add_special_tokens=False)

    # ---- data ----
    seeds = read_seeds(args.seeds, args.k)
    if not seeds:
        raise SystemExit(f"No seeds found in {args.seeds}")
    prompts_all = build_prompts(seeds, tok)

    # ---- external scorers ----
    #mt = MTEnDe(device=args.mt_device)
    qe = Sentinel() #COMETQE(accelerator_preference=args.comet_accelerator)            # wmt22-cometkiwi-da (reference-free)
    vf = LMVerifier(device=args.lm_verifier_device)

    # Precompute de_old once (1 - QE(s0, t0))
    #t0 = mt.translate(seeds)
    de_old = qe.difficulty(seeds) #qe.difficulty(seeds, t0)

    # rewarder = Method2Rewarder(
    #     seeds=seeds, de_old=de_old, mt=mt, qe=qe, vf=vf,
    #     x=args.x, y=args.y, z=args.z, f=args.f,
    #     normalize_rewards=args.normalize_rewards,
    #     clip_normalized=args.clip_normalized
    # )
    rewarder = Method2Rewarder(
        seeds=seeds, de_old=de_old, qe=qe, vf=vf,
        x=args.x, y=args.y, z=args.z, f=args.f,
        constraint_threshold=args.constraint_threshold,
        constraint_sharpness=args.constraint_sharpness,
        normalize_rewards=args.normalize_rewards,
        clip_normalized=args.clip_normalized
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
    # If resuming from checkpoint, load the fine-tuned model; otherwise load base model
    model_path = args.resume_from_checkpoint if args.resume_from_checkpoint else args.base_model
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        # Verify checkpoint directory contains model files
        checkpoint_files = os.listdir(args.resume_from_checkpoint)
        print(f"Checkpoint directory contains: {checkpoint_files}")

        # Check for required model files
        required_files = ['pytorch_model.bin', 'config.json']  # Standard HF checkpoint files
        missing_files = [f for f in required_files if f not in checkpoint_files]
        if missing_files:
            print(f"WARNING: Missing expected checkpoint files: {missing_files}")
            print("This may cause issues loading the checkpoint. Falling back to base model.")
            model_path = args.base_model

    # Configure model for gradient checkpointing compatibility
    model_config = {
        # Don't specify torch_dtype - let it load in default precision (FP32)
        # Mixed precision training will handle the conversion for memory efficiency
    }

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        **model_config
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

    # Enable gradient checkpointing to save memory during training
    # IMPORTANT: Must disable use_cache for gradient checkpointing to work
    if hasattr(policy.pretrained_model, 'gradient_checkpointing_enable'):
        policy.pretrained_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Explicitly disable caching for gradient checkpointing compatibility
    if hasattr(policy.pretrained_model.config, 'use_cache'):
        policy.pretrained_model.config.use_cache = False
        print("Model cache disabled for gradient checkpointing compatibility")

    # Ensure all model parameters require gradients (critical for training)
    # When loading from checkpoint, params might not have requires_grad set properly
    policy.train()  # Set to training mode
    for param in policy.parameters():
        param.requires_grad = True
    print(f"Model parameters set to trainable. Total params: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")

    # TRL will create a frozen reference model internally (this doubles memory usage!)
    # For Qwen3-0.6B: ~1.2B parameters total (2x 0.6B), with FP32 weights this is ~4.8GB
    # Plus activations, gradients, optimizer states, and generation buffers = easily 10-11GB on 1080Ti

    optimizer = SGD(policy.parameters(), lr=LR, momentum=0.8)
    num_training_steps = args.steps * PPO_EPOCHS
    lr_scheduler = get_constant_schedule_with_warmup( # get_linear_schedule_with_warmup
        optimizer,
        num_warmup_steps=0,
        #num_training_steps=num_training_steps,
    )

    ppo_config = PPOConfig(
        model_name=args.base_model,
        # learning_rate=LR,            # Lower LR for more stable training
        batch_size=args.batch_size,    # samples used per PPO step
        mini_batch_size=1,             # Process one sample at a time to minimize memory usage
        gradient_accumulation_steps=1,
        ppo_epochs=PPO_EPOCHS,                  # Reduced from 4 to save memory (fewer forward/backward passes)
        cliprange=0.2,
        cliprange_value=0.2,           # Clip value function updates
        vf_coef=0.2,                   # Value function coefficient
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
        lr_scheduler=lr_scheduler,
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
        top_k=50,  # Increased from 20 to allow more vocabulary diversity
        repetition_penalty=1.1,  # Penalize repetitions to avoid repeating prompt
        no_repeat_ngram_size=3,  # Prevent repeating 3-grams (helps avoid instruction repetition)
        bad_words_ids=[asterisk_token_ids],  # Prevent generating asterisks (markdown formatting)
    )

    # ---- Load checkpoint state if resuming ----
    start_step = 0
    if args.resume_from_checkpoint:
        checkpoint_state_path = os.path.join(args.resume_from_checkpoint, "training_state.pt")
        if os.path.exists(checkpoint_state_path):
            print(f"Loading training state from: {checkpoint_state_path}")
            checkpoint_state = torch.load(checkpoint_state_path, map_location=device)

            # Restore optimizer and scheduler
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])

            # Restore random states for reproducibility
            if "random_state" in checkpoint_state:
                random.setstate(checkpoint_state["random_state"])
            if "torch_random_state" in checkpoint_state:
                torch.set_rng_state(checkpoint_state["torch_random_state"])
            if "torch_cuda_random_state" in checkpoint_state and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint_state["torch_cuda_random_state"])

            # Restore reward normalizer states if they exist
            if args.normalize_rewards and "delta_normalizer" in checkpoint_state:
                rewarder.delta_normalizer.mean = checkpoint_state["delta_normalizer"]["mean"]
                rewarder.delta_normalizer.var = checkpoint_state["delta_normalizer"]["var"]
                rewarder.delta_normalizer.count = checkpoint_state["delta_normalizer"]["count"]

                rewarder.verifier_normalizer.mean = checkpoint_state["verifier_normalizer"]["mean"]
                rewarder.verifier_normalizer.var = checkpoint_state["verifier_normalizer"]["var"]
                rewarder.verifier_normalizer.count = checkpoint_state["verifier_normalizer"]["count"]

                rewarder.constraint_normalizer.mean = checkpoint_state["constraint_normalizer"]["mean"]
                rewarder.constraint_normalizer.var = checkpoint_state["constraint_normalizer"]["var"]
                rewarder.constraint_normalizer.count = checkpoint_state["constraint_normalizer"]["count"]

                print("Restored reward normalizer states from checkpoint")

            start_step = checkpoint_state["step"] + 1  # Resume from next step
            print(f"Resuming from step {start_step}")
        else:
            print(f"Warning: No training_state.pt found in {args.resume_from_checkpoint}, starting from step 0")

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
    print(f"Test prompt: {test_prompt[:100]}...")
    print(f"Test output: {test_response}")
    print("="*80 + "\n")
    policy.train()  # Set back to train mode

    # ---- PPO loop ----
    n = len(prompts_all)
    print(f"Starting PPO: seeds={n}, steps={args.steps}, batch={args.batch_size}")
    if start_step > 0:
        print(f"Resuming training from step {start_step}")
    for step in trange(start_step, args.steps, desc="PPO", initial=start_step, total=args.steps):
        # sample a mini-batch of prompts (cyclic with wrapping)
        start = (step * args.batch_size) % n
        idx = [(start + i) % n for i in range(args.batch_size)]
        prompts = [prompts_all[i] for i in idx]
        # 1) tokenize prompts
        query_tensors = [tok(p, return_tensors="pt").input_ids.squeeze(0) for p in prompts]
        query_tensors = [q.to(trainer.accelerator.device) for q in query_tensors] # move to device

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
                # print(f"Decoded prompt: {tok.decode(query, skip_special_tokens=True).strip()}")
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
        cleaned = [clean_markdown(tok.decode(r, skip_special_tokens=True).strip()) for r in response_tensors]

        # Final safeguard: replace any empty strings with a minimal fallback
        cleaned = [s if s else "." for s in cleaned]

        # 4) compute rewards (external scorers)
        rewards_list, con_rewards, ver_rewards, diff_rewards = rewarder.reward(cleaned, idx)
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

        # Add normalization statistics if enabled
        if args.normalize_rewards:
            log_dict.update({
                "norm_stats/delta_mean": rewarder.delta_normalizer.mean,
                "norm_stats/delta_std": rewarder.delta_normalizer.std,
                "norm_stats/verifier_mean": rewarder.verifier_normalizer.mean,
                "norm_stats/verifier_std": rewarder.verifier_normalizer.std,
                "norm_stats/constraint_mean": rewarder.constraint_normalizer.mean,
                "norm_stats/constraint_std": rewarder.constraint_normalizer.std,
            })

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
            # Save the full model (both pretrained_model and v_head)
            # Use trainer.save_pretrained() which saves the model with value head
            trainer.save_pretrained(args.save_dir)

            # Also explicitly save the model's config to ensure proper loading
            if hasattr(policy.pretrained_model, 'config'):
                policy.pretrained_model.config.save_pretrained(args.save_dir)

            # Save training state (optimizer, scheduler, step, normalizers)
            training_state = {
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                training_state["torch_cuda_random_state"] = torch.cuda.get_rng_state()

            # Save reward normalizer states if enabled
            if args.normalize_rewards:
                training_state["delta_normalizer"] = {
                    "mean": rewarder.delta_normalizer.mean,
                    "var": rewarder.delta_normalizer.var,
                    "count": rewarder.delta_normalizer.count,
                }
                training_state["verifier_normalizer"] = {
                    "mean": rewarder.verifier_normalizer.mean,
                    "var": rewarder.verifier_normalizer.var,
                    "count": rewarder.verifier_normalizer.count,
                }
                training_state["constraint_normalizer"] = {
                    "mean": rewarder.constraint_normalizer.mean,
                    "var": rewarder.constraint_normalizer.var,
                    "count": rewarder.constraint_normalizer.count,
                }

            torch.save(training_state, os.path.join(args.save_dir, "training_state.pt"))

            # Verify checkpoint files exist
            checkpoint_files = os.listdir(args.save_dir)
            print(f"Checkpoint saved at step {step + 1}. Files: {checkpoint_files}")

            torch.cuda.empty_cache()

    # final save
    trainer.save_pretrained(args.save_dir)

    # Also explicitly save the model's config to ensure proper loading
    if hasattr(policy.pretrained_model, 'config'):
        policy.pretrained_model.config.save_pretrained(args.save_dir)

    # Save final training state
    final_training_state = {
        "step": args.steps - 1,  # Last step completed
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        final_training_state["torch_cuda_random_state"] = torch.cuda.get_rng_state()

    # Save reward normalizer states if enabled
    if args.normalize_rewards:
        final_training_state["delta_normalizer"] = {
            "mean": rewarder.delta_normalizer.mean,
            "var": rewarder.delta_normalizer.var,
            "count": rewarder.delta_normalizer.count,
        }
        final_training_state["verifier_normalizer"] = {
            "mean": rewarder.verifier_normalizer.mean,
            "var": rewarder.verifier_normalizer.var,
            "count": rewarder.verifier_normalizer.count,
        }
        final_training_state["constraint_normalizer"] = {
            "mean": rewarder.constraint_normalizer.mean,
            "var": rewarder.constraint_normalizer.var,
            "count": rewarder.constraint_normalizer.count,
        }

    torch.save(final_training_state, os.path.join(args.save_dir, "training_state.pt"))
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

        # NOTE: Artifact logging disabled to prevent storage clutter
        # To enable, uncomment the following lines:
        # artifact = wandb.Artifact(
        #     name=f"ppo-model-{wandb.run.id}",
        #     type="model",
        #     description=f"PPO fine-tuned {args.base_model} for MT-breaking"
        # )
        # artifact.add_dir(args.save_dir)
        # wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    main()
