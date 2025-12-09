from transformers import AutoModelForCausalLM, AutoTokenizer
from sentinel_metric import download_model as sentinel_download_model, load_from_checkpoint as sentinel_load_from_checkpoint
import numpy as np

import torch
import gc
import argparse
from pathlib import Path


torch.cuda.empty_cache()
gc.collect()

class Sentinel:
    def __init__(self):
        ckpt = sentinel_download_model("Prosho/sentinel-src-25")
        self.model = sentinel_load_from_checkpoint(ckpt)

    def difficulty(self, sentences):
        data = [{"src": s} for s in sentences]
        out = self.model.predict(data, batch_size = 8, gpus = 1)
        return [1.0 - score for score in out.scores]

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test model inference with optional checkpoint loading")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint directory (e.g., checkpoints/checkpoint-1000)"
)
parser.add_argument(
    "--base-model",
    type=str,
    default="Qwen/Qwen3-0.6B-Base",
    help="Base model name or path"
)
args = parser.parse_args()

model_name = args.base_model

# Determine which model to load
if args.checkpoint:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model_path = str(checkpoint_path)
else:
    print(f"Loading base model: {model_name}")
    model_path = model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto"
)
qe = Sentinel()

seed = "He read a novel."

prompt = (
    """
    Rewrite the following sentence to be extremely difficult for machine translation using idioms, ambiguity, and wordplay while keeping it grammatically correct English. Return only plain text without any formatting, markdown, or special characters. Output only a single edited sentence:

    {seed}
    """
)

model_inputs = tokenizer(
    prompt.format(seed=seed),
    return_tensors="pt",
).to(model.device)

print("Prompting model...")
generated_ids = model.generate(
    **model_inputs,
    # temperature=0.4,
    max_new_tokens=45
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
diff_seed = qe.difficulty([seed])
diff_cont = qe.difficulty([content])
delta = np.array(diff_cont) - np.array(diff_seed)
print("seed:", seed)
print("content:", content)
print("seed difficulty:", diff_seed)
print("content difficulty:", diff_cont)
print("difficulty delta:", delta)
