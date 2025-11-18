import os, argparse, random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from tqdm import trange
import wandb

from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead

from .translate import MTEnDe
from .scorers import COMETQE, LMVerifier
from .constraints import constraint_score
from .losses import method2_loss


DEFAULT_BASE = "google/flan-t5-base"
DEFAULT_INSTRUCTION = (
        """
    We want to find a sentence in English that’s exceptionally difficult for a machine translation model to translate into some other language. The goal is to expose a wide range of translation errors and severely challenge the MT model’s capabilities.

    Use this English sentence as a foundation and try to make it even more difficult to translate by adding words, changing the structure of the sentence or making other modifications: „{seed}“

    Only return the difficult-to-translate English sentence, nothing else!
    """
)


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
    mt: MTEnDe
    qe: COMETQE
    vf: LMVerifier
    x: float = 1.0
    y: float = 0.3
    z: float = 0.3
    f: str = "relu"

    @torch.inference_mode()
    def reward(self, edits: List[str], indices: List[int]) -> List[float]:
        """
        Compute rewards for the given edits.
        indices: which seeds these edits correspond to (for selecting de_old)
        """
        t1 = self.mt.translate(edits)
        de_new = self.qe.difficulty(edits, t1)
        cons = constraint_score(edits)
        ver = self.vf.score(edits)
        # Select the corresponding de_old values for this batch
        de_old_batch = [self.de_old[i] for i in indices]
        L = method2_loss(de_new, de_old_batch, cons, ver, x=self.x, y=self.y, z=self.z, f=self.f)
        # PPO wants rewards (higher is better)
        return [-float(l) for l in L]


def build_prompts(seeds: List[str]) -> List[str]:
    return [DEFAULT_INSTRUCTION.format(seed=s) for s in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="data/seeds.txt")
    ap.add_argument("--k", type=int, default=200, help="num seeds to use")
    ap.add_argument("--base_model", type=str, default=DEFAULT_BASE)
    ap.add_argument("--save_dir", type=str, default="checkpoints/gpt2-ppo-method2")
    ap.add_argument("--steps", type=int, default=200)           # PPO update steps
    ap.add_argument("--batch_size", type=int, default=8)        # prompts per PPO step
    ap.add_argument("--gen_max_new_tokens", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.9)
    # loss weights
    ap.add_argument("--x", type=float, default=1.0)
    ap.add_argument("--y", type=float, default=0.3)
    ap.add_argument("--z", type=float, default=0.3)
    ap.add_argument("--f", type=str, default="none", choices=["relu","sigmoid","none"])
    ap.add_argument("--seed", type=int, default=42)
    # wandb
    ap.add_argument("--wandb_project", type=str, default="mt-breaker-ppo")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
                "learning_rate": 1e-4,
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
            }
        )

    # ---- data ----
    seeds = read_seeds(args.seeds, args.k)
    if not seeds:
        raise SystemExit(f"No seeds found in {args.seeds}")
    prompts_all = build_prompts(seeds)

    # ---- external scorers ----
    mt = MTEnDe()
    qe = COMETQE()            # wmt22-cometkiwi-da (reference-free)
    vf = LMVerifier()

    # Precompute de_old once (1 - QE(s0, t0))
    t0 = mt.translate(seeds)
    de_old = qe.difficulty(seeds, t0)

    rewarder = Method2Rewarder(
        seeds=seeds, de_old=de_old, mt=mt, qe=qe, vf=vf,
        x=args.x, y=args.y, z=args.z, f=args.f
    )

    # ---- policy + ref model ----
    tok = AutoTokenizer.from_pretrained(args.base_model)
    # T5 models already have pad_token configured, no need to set it manually

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.base_model)
    # TRL will create a frozen reference model internally

    ppo_config = PPOConfig(
        model_name=args.base_model,
        learning_rate=1e-4,            # Lower LR for more stable training
        batch_size=args.batch_size,    # samples used per PPO step
        mini_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        cliprange=0.2,
        cliprange_value=0.2,           # Clip value function updates
        vf_coef=0.1,                   # Value function coefficient
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
    )

    gen_kwargs = dict(
        do_sample=True,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.gen_max_new_tokens,
        min_new_tokens=2,  # Ensure at least 2 tokens to avoid masking issues
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # ---- PPO loop ----
    n = len(prompts_all)
    print(f"Starting PPO: seeds={n}, steps={args.steps}, batch={args.batch_size}")
    for step in trange(args.steps, desc="PPO"):
        # sample a mini-batch of prompts (cyclic with wrapping)
        start = (step * args.batch_size) % n
        idx = [(start + i) % n for i in range(args.batch_size)]
        prompts = [prompts_all[i] for i in idx]

        # 1) tokenize prompts
        query_tensors = [tok(p, return_tensors="pt").input_ids.squeeze(0) for p in prompts]
        query_tensors = [q.to(trainer.accelerator.device) for q in query_tensors]

        # 2) generate responses
        # For seq2seq models like T5, the output is the full generation (no prompt prefix to remove)
        response_tensors = []
        with torch.no_grad():
            for query in query_tensors:
                gen_output = trainer.generate(query, **gen_kwargs)
                # For seq2seq, the output is already just the response
                response = gen_output.squeeze()

                # Handle empty or very short responses - regenerate with higher temperature or use fallback
                # Need at least 2 tokens to avoid masking issues in PPO trainer
                max_retries = 3
                retry_count = 0
                while len(response) < 2 and retry_count < max_retries:
                    # Generate at least something with higher temperature
                    fallback_kwargs = {**gen_kwargs, "temperature": min(1.5 + retry_count * 0.2, 2.0), "min_new_tokens": 5}
                    gen_output = trainer.generate(query, **fallback_kwargs)
                    response = gen_output.squeeze()
                    retry_count += 1

                # If still too short after retries, create a minimal valid response
                if len(response) < 2:
                    # Create a minimal response with at least 2 tokens
                    fallback_text = "The sentence is difficult."
                    response = tok(fallback_text, return_tensors="pt").input_ids.squeeze(0).to(trainer.accelerator.device)

                response_tensors.append(response)

        # 3) decode responses for reward computation
        cleaned = [tok.decode(r, skip_special_tokens=True).strip() for r in response_tensors]

        # Final safeguard: replace any empty strings with a minimal fallback
        cleaned = [s if s else "." for s in cleaned]

        # 4) compute rewards (external scorers)
        rewards_list = rewarder.reward(cleaned, idx)
        rewards = [torch.tensor(r) for r in rewards_list]

        # 5) feed PPO step (query tensors, response tensors, reward tensors)
        stats = trainer.step(query_tensors, response_tensors, rewards)

        # 6) Log metrics to wandb and console
        mean_reward = sum(rewards_list) / len(rewards_list)
        min_reward = min(rewards_list)
        max_reward = max(rewards_list)

        # Compute response length statistics
        response_lengths = [len(r) for r in response_tensors]
        mean_response_len = sum(response_lengths) / len(response_lengths)

        # Prepare wandb logging dict with rewards and generation stats
        log_dict = {
            "reward/mean": mean_reward,
            "reward/min": min_reward,
            "reward/max": max_reward,
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
    demo = DEFAULT_INSTRUCTION.format(seed="He saw her duck.")
    ids = tok([demo], return_tensors="pt").to(trainer.accelerator.device)
    with torch.no_grad():
        gen = trainer.model.generate(**ids, **gen_kwargs)
    final_sample = tok.batch_decode(gen, skip_special_tokens=True)[0]
    print("Sample:", final_sample)

    # Log final model to wandb
    if use_wandb:
        # Log final sample
        wandb.log({
            "final/sample_output": wandb.Html(f"<pre>{final_sample}</pre>"),
        })

        # Save model as artifact
        artifact = wandb.Artifact(
            name=f"ppo-model-{wandb.run.id}",
            type="model",
            description=f"PPO fine-tuned {args.base_model} for MT-breaking"
        )
        artifact.add_dir(args.save_dir)
        wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    main()
