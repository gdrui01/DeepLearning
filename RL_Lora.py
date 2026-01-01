import os
import torch
import numpy as np
import wandb
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from sentence_transformers import SentenceTransformer, util

from sentinel_metric import download_model as sentinel_download_model
from sentinel_metric import load_from_checkpoint as sentinel_load_from_checkpoint
from src.constraints import constraint_score
from src.scorers import LMVerifier
import logging
import warnings
from datetime import datetime

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")
warnings.filterwarnings("ignore", category=UserWarning, module="trl")



if os.path.exists("wandb.key"):
    print("Found wandb.key, logging in...")
    with open("wandb.key") as f:
        wandb.login(key=f.read().strip())
else:
    print("No wandb.key found")


def calculate_final_reward(
    sentinel_diff: float, 
    semantic_sim: float, 
    ppl_score: float, 
    constraints: dict
) -> tuple[float, str]:
    # Gatekeepers - keep meaning
    if semantic_sim < 0.65: # todo tune threshold
        return -2.0, "fail_semantic"
        
    # Length/Sanity Constraints
    if constraints['char_score'] < 0.5 or constraints['len_score'] < 0.5:
        return -2.0, "fail_constraint"
        
    # Optimizers, fluency check
    fluency_penalty = 0.0
    status = "success"
    
    if ppl_score < 0.3: 
        fluency_penalty = -1.0
        status = "success_with_fluency_penalty" 
        
    # Core Reward (Sentinel Difference)
    # We want to maximize difficulty increase
    difficulty_reward = sentinel_diff * 15.0 # Todo maybe tune multiplier
    
    total_reward = difficulty_reward + fluency_penalty
    return np.clip(total_reward, -5.0, 5.0), status

class Sentinel:
    def __init__(self, device="cuda:0"):
        print("Loading Sentinel...")
        ckpt = sentinel_download_model("Prosho/sentinel-src-25")
        self.model = sentinel_load_from_checkpoint(ckpt)

    def difficulty(self, sentences: list[str]) -> list[float]:
        data = [{"src": s} for s in sentences]
        out = self.model.predict(data, batch_size=8, gpus=1) 
        return out.scores


config = PPOConfig(
    model_name="meta-llama/Llama-3.2-3B-Instruct", #Todo switch to 7B at some point
    learning_rate=1e-6, # Todo maybe too large
    batch_size=64,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    target_kl=0.1,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
    kl_penalty="kl",
    log_with="wandb",
    tracker_project_name="mt-breaker" 
)


device = "cuda" if torch.cuda.is_available() else "cpu"

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)


model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map="auto",
    peft_config=lora_config,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token


print("Loading Reward Models...")
sentinel = Sentinel()
verifier = LMVerifier()
semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=device) #Todo maybe switch to all-mpnet-base-v2


def build_dataset(dataset_name="matafix/english-sentences-with-sentinel"):
    print(f"Loading dataset {dataset_name}...")
    hf_ds = load_dataset(dataset_name, split="train")
    
    # filter long sentences (we keep 99%)
    original_len = len(hf_ds)
    hf_ds = hf_ds.filter(lambda example: len(example['text']) < 355)
    print(f"Filtered: {original_len} -> {len(hf_ds)}")

    ds_data = {
        "input_ids": [],
        "original_text": [],
        "baseline_difficulty": []
    }

    instruction_template = (
        "Rewrite the input sentence to be linguistically complex and difficult to machine translate "
        "(using idioms, rare vocabulary, or nested clauses) while strictly preserving the original meaning.\n\n"
        "Constraints:\n"
        "1. Output ONLY the rewritten sentence.\n"
        "2. Do not provide introductory text or explanations.\n"
        "3. The output must be a single sentence of comparable length to the original.\n\n"
        "Original: {sent}\n"
        "Rewritten:"
    )

    print("Pre-tokenizing dataset (this might take a moment)...")
    for row in hf_ds:
        sent = row['text']
        base_score = row['sentinel_score']
        
        full_prompt = instruction_template.format(sent=sent)
        
        # Tokenize to list of ints (not tensors yet)
        tokenized = tokenizer(full_prompt, add_special_tokens=True, truncation=True, max_length=512)
        
        ds_data["input_ids"].append(tokenized["input_ids"])
        ds_data["original_text"].append(sent)
        ds_data["baseline_difficulty"].append(base_score)

    ds = Dataset.from_dict(ds_data)
    return ds

dataset = build_dataset()


def manual_collator(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = [d[key] for d in data]
    return batch


train_dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=manual_collator,
    drop_last=True
)

wandb.init(project=config.tracker_project_name, config=config.to_dict())

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=manual_collator,
)


generation_kwargs = {
    "min_length": 5, "top_k": 0.0, "top_p": 0.9, "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 128,
}

LOG_FREQ = 10
MAX_STEPS = 10000 # we stop earlier manually when we see collapse
all_history_rows = []
print(f"Starting PPO Training (Max Steps: {MAX_STEPS})...")

for step, batch in enumerate(train_dataloader):
    if step >= MAX_STEPS:
        print("Reached max steps. Stopping.")
        break
        

    query_tensors = [torch.tensor(ids).to(device) for ids in batch["input_ids"]]
    
    response_tensors = ppo_trainer.generate(
        query_tensors, 
        return_prompt=False, 
        **generation_kwargs
    )
    raw_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    batch["response"] = [r.strip() for r in raw_responses]

    current_difficulties = sentinel.difficulty(batch["response"])
    difficulty_rewards = [base - cur for cur, base in zip(current_difficulties, batch["baseline_difficulty"])]

    orig_embeddings = semantic_model.encode(batch["original_text"], convert_to_tensor=True)
    new_embeddings = semantic_model.encode(batch["response"], convert_to_tensor=True)
    cosine_scores = util.cos_sim(orig_embeddings, new_embeddings).diagonal()

    constraint_scores = constraint_score(batch["response"], batch["original_text"])
    verifier_scores = verifier.score(batch["response"])

    rewards = []
    status_list = []
    status_counts = {"fail_semantic": 0, "fail_constraint": 0, "success": 0, "success_with_fluency_penalty": 0}
    
    for diff_r, sem_score, constr_score, verifier_score in zip(difficulty_rewards, cosine_scores, constraint_scores, verifier_scores):
        final_reward, status_code = calculate_final_reward(diff_r, sem_score.item(), verifier_score, constr_score)
        rewards.append(torch.tensor(final_reward, dtype=torch.float32))
        status_list.append(status_code)
        
        if status_code in status_counts:
            status_counts[status_code] += 1
        elif "success" in status_code:
            status_counts["success"] += 1

    batch_size_val = len(rewards)
    custom_metrics = {
        "gatekeeper/semantic_fail_rate": status_counts["fail_semantic"] / batch_size_val,
        "gatekeeper/constraint_fail_rate": status_counts["fail_constraint"] / batch_size_val,
        "gatekeeper/success_rate": (status_counts["success"] + status_counts["success_with_fluency_penalty"]) / batch_size_val,
        "sentinel/mean_raw_diff": np.mean(difficulty_rewards),
        "semantics/mean_score": torch.stack(list(cosine_scores)).mean().item()
    }


    if step % LOG_FREQ == 0:
        current_rows = []
        for i in range(min(4, batch_size_val)):
            current_rows.append([
                step,
                batch["original_text"][i],
                batch["response"][i],
                difficulty_rewards[i],       
                cosine_scores[i].item(),    
                status_list[i],              
                rewards[i].item()            
            ])
        
        all_history_rows.extend(current_rows)
        
        wandb.log({"detailed_log": wandb.Table(
            columns=["Step", "Original", "Rewritten", "Sentinel Diff", "Sem Score", "Status", "Reward"], 
            data=all_history_rows
        )})
        

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    stats.update(custom_metrics)

    batch["query"] = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
    ppo_trainer.log_stats(stats, batch, rewards)
    
    print(f"Step {step}: Mean Reward: {torch.stack(rewards).mean().item():.4f} | Success: {custom_metrics['gatekeeper/success_rate']:.2%}")
    
    if step % 50 == 0 and step > 0:
        save_path = f"checkpoints/step_{step}"
    
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Saving checkpoint to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


model.save_pretrained("mt-breaker")
print("Training Complete.")