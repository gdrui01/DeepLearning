import os, argparse, random
from dataclasses import dataclass
from typing import List

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
    "Make the following sentence harder to translate while keeping it grammatically correct and natural.\n\n"
    "'{seed}'\n\nRewrite it as one grammatical English sentence. Output only the edited sentence, no other text."
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
                "learning_rate": 5e-6,
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
        learning_rate=5e-6,            # Lower LR for more stable training
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

                # Handle empty responses - regenerate with higher temperature or use fallback
                if len(response) == 0:
                    # Generate at least something with higher temperature
                    fallback_kwargs = {**gen_kwargs, "temperature": 1.0, "min_new_tokens": 5}
                    gen_output = trainer.generate(query, **fallback_kwargs)
                    response = gen_output.squeeze()

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

        # Extract PPO stats
        ppo_loss = stats.get('ppo/loss/total', 0.0) if isinstance(stats, dict) else 0.0
        kl_div = stats.get('objective/kl', 0.0) if isinstance(stats, dict) else 0.0
        entropy = stats.get('objective/entropy', 0.0) if isinstance(stats, dict) else 0.0
        approx_kl = stats.get('objective/kl_coef', 0.0) if isinstance(stats, dict) else 0.0

        # Compute response length statistics
        response_lengths = [len(r) for r in response_tensors]
        mean_response_len = sum(response_lengths) / len(response_lengths)

        if use_wandb:
            wandb.log({
                "step": step + 1,
                "reward/mean": mean_reward,
                "reward/min": min_reward,
                "reward/max": max_reward,
                "ppo/loss": ppo_loss,
                "ppo/kl_divergence": kl_div,
                "ppo/entropy": entropy,
                "ppo/kl_coef": approx_kl,
                "generation/mean_response_length": mean_response_len,
                "generation/sample_text": wandb.Html(f"<pre>{cleaned[0]}</pre>"),
            })

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
