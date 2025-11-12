import os, json, math, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


DEFAULT_BASE = "gpt2"
DEFAULT_PROMPT_TMPL = (
    "Make the following sentence harder to translate while keeping it grammatical and natural.\n\n"
    "'{seed}'\n\nRewrite it as one grammatical English sentence."
)


# ---------- data loading ----------

def load_pairs_from_jsonl(path: str,
                          top_k: int,
                          require_positive_delta: bool = True) -> List[Dict]:
    """
    Read method2_results.jsonl and select top_k examples with the lowest loss.
    Optionally keep only examples where delta_de > 0.
    Returns a list of dicts with 'prompt' and 'target'.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if require_positive_delta and r.get("delta_de", 0.0) <= 0:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit("No usable rows found. Check your results file or disable --require_positive_delta.")

    rows = sorted(rows, key=lambda x: float(x["L"]))[:top_k]
    pairs = []
    for r in rows:
        seed = r["orig"].strip()
        edit = r["edit"].strip()
        prompt = DEFAULT_PROMPT_TMPL.format(seed=seed)
        if not seed or not edit:
            continue
        pairs.append({"prompt": prompt, "target": edit})
    if not pairs:
        raise SystemExit("No (prompt, target) pairs after filtering.")
    return pairs


# ---------- tokenization with loss masking ----------

@dataclass
class SFTFormatter:
    tokenizer: AutoTokenizer
    max_length: int = 256

    def __call__(self, example: Dict) -> Dict:
        """
        Build input_ids and labels such that the loss is computed ONLY on the target segment.
        labels for the prompt tokens are set to -100.
        """
        prompt = example["prompt"].strip()
        target = example["target"].strip()

        # Full text = prompt + newline + target
        full_text = prompt + "\n\n" + target

        enc_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        enc_prompt = self.tokenizer(
            prompt + "\n\n",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        input_ids = enc_full["input_ids"]
        labels = input_ids.copy()

        prompt_len = min(len(enc_prompt["input_ids"]), len(labels))
        # mask prompt tokens
        for i in range(prompt_len):
            labels[i] = -100

        return {"input_ids": input_ids, "labels": labels}


def make_hf_dataset(pairs: List[Dict], tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    fmt = SFTFormatter(tokenizer, max_length=max_length)
    ds = Dataset.from_list(pairs)
    ds = ds.map(fmt, remove_columns=["prompt", "target"])
    return ds


# ---------- LoRA SFT trainer ----------

def build_lora_model(base_model: str, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["c_attn", "c_proj"],  # good defaults for GPT-2 family
    )
    model = get_peft_model(model, peft_cfg)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/method2_results.jsonl",
                    help="JSONL produced by src.method2")
    ap.add_argument("--top_k", type=int, default=300,
                    help="take lowest-loss K examples (after filtering)")
    ap.add_argument("--require_positive_delta", action="store_true", default=True,
                    help="keep only rows with delta_de > 0")
    ap.add_argument("--base_model", type=str, default=DEFAULT_BASE)
    ap.add_argument("--outdir", type=str, default="checkpoints/gpt2-lora-sft")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) select pairs
    pairs = load_pairs_from_jsonl(
        args.data,
        top_k=args.top_k,
        require_positive_delta=args.require_positive_delta,
    )

    # 2) build model + tokenizer with LoRA
    model, tok = build_lora_model(args.base_model)

    # 3) dataset
    ds = make_hf_dataset(pairs, tok, max_length=args.max_length)

    # 4) training args
    use_fp16 = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=False,
        report_to=[],  # no wandb by default
        remove_unused_columns=False,  # we already formatted the inputs
    )

    # 5) train
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.outdir)  # saves LoRA adapter weights

    # 6) quick sanity: show how to use the adapter for generation
    print("\nTraining complete. Example usage:")
    print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"from peft import PeftModel")
    print(f'model = AutoModelForCausalLM.from_pretrained(\"{args.base_model}\")')
    print(f'tok = AutoTokenizer.from_pretrained(\"{args.base_model}\")')
    print(f'model = PeftModel.from_pretrained(model, \"{args.outdir}\")')
    print(f'text = \"{DEFAULT_PROMPT_TMPL.format(seed='He saw her duck.')}\"')
    print(f'ids = tok([text], return_tensors=\"pt\").to(model.device)')
    print(f'out = model.generate(**ids, do_sample=True, top_p=0.95, temperature=0.9, max_new_tokens=40)')
    print(f'print(tok.batch_decode(out, skip_special_tokens=True)[0])')


if __name__ == "__main__":
    main()
