import os, argparse, random
from dataclasses import dataclass
from typing import List

import torch
from tqdm import trange

from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from .translate import MTEnDe
from .scorers import COMETQE, LMVerifier
from .constraints import constraint_score
from .losses import method2_loss


DEFAULT_BASE = "gpt2"
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
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    if tok.pad_token is None:
        # Add a distinct pad token instead of reusing eos_token
        tok.add_special_tokens({'pad_token': '[PAD]'})

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.base_model)
    # Resize embeddings if we added a new pad token
    policy.pretrained_model.resize_token_embeddings(len(tok))
    # TRL will create a frozen reference model internally

    ppo_config = PPOConfig(
        model_name=args.base_model,
        learning_rate=1e-5,
        batch_size=args.batch_size,    # samples used per PPO step
        mini_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        cliprange=0.2,
        kl_penalty="kl",
        init_kl_coef=0.05,
        target_kl=0.15,
        seed=args.seed,
        accelerator_kwargs={"mixed_precision": "fp16" if torch.cuda.is_available() else "no"},
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
        # sample a mini-batch of prompts (cyclic)
        start = (step * args.batch_size) % n
        idx = list(range(start, min(start + args.batch_size, n)))
        prompts = [prompts_all[i] for i in idx]

        # 1) tokenize prompts
        query_tensors = [tok(p, return_tensors="pt").input_ids.squeeze(0) for p in prompts]
        query_tensors = [q.to(trainer.accelerator.device) for q in query_tensors]

        # 2) generate responses
        response_tensors = []
        with torch.no_grad():
            for query in query_tensors:
                response = trainer.generate(query, **gen_kwargs)
                response_tensors.append(response.squeeze())

        # 3) decode responses for reward computation
        responses_decoded = [tok.decode(r, skip_special_tokens=True) for r in response_tensors]

        # Extract just the generated text (remove prompt)
        cleaned = []
        for p, r_full in zip(prompts, responses_decoded):
            r_full = r_full.strip()
            # Try to extract just the response part
            cand = r_full[len(p):].strip() if r_full.startswith(p) else r_full
            cleaned.append(cand if cand else r_full)

        # 4) compute rewards (external scorers)
        rewards_list = rewarder.reward(cleaned, idx)
        rewards = [torch.tensor(r) for r in rewards_list]

        # 5) feed PPO step (query tensors, response tensors, reward tensors)
        trainer.step(query_tensors, response_tensors, rewards)

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
    print("Sample:", tok.batch_decode(gen, skip_special_tokens=True)[0])
    

if __name__ == "__main__":
    main()
