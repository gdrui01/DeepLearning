import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util


from sentinel_metric import download_model as sentinel_download_model
from sentinel_metric import load_from_checkpoint as sentinel_load_from_checkpoint


BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
FINETUNED_CHECKPOINT = "../checkpoints/step_500"
DATASET_NAME = "matafix/english-sentences-with-sentinel"
NUM_SAMPLES = 100 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_LOG = "detailed_results_filtered.log" 

# filtering in case finetuned model starts writing too much. Evaluating only on good outputs
FORBIDDEN_PHRASES = [
    "Original:", "Rewritten:", "Note:", "Here is", "In this sentence", 
    "The revised sentence", "Please provide", "Example:", "output format", "policy"
]

def validate_response(original, response):
    """Returns (True/False, reason)"""
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in response.lower():
            return False, f"Contains '{phrase}'"
    
    # also filter out too long/short responses
    if len(response) > 2.5 * len(original):
        return False, "Too Long (>2.5x)"
    if len(response) < 0.5 * len(original):
        return False, "Too Short (<0.5x)"
        
    return True, "Valid"

class Sentinel:
    def __init__(self, device="cuda:0"):
        print("Loading Sentinel...")
        ckpt = sentinel_download_model("Prosho/sentinel-src-25")
        self.model = sentinel_load_from_checkpoint(ckpt)

    def difficulty(self, sentences: list[str]) -> list[float]:
        clean_sentences = [s if s.strip() else "EMPTY" for s in sentences]
        data = [{"src": s} for s in clean_sentences]
        out = self.model.predict(data, batch_size=8, gpus=1) 
        return out.scores


def get_prompt(original_text):
    return (
        "Rewrite the input sentence to be linguistically complex and difficult to machine translate "
        "(using idioms, rare vocabulary, or nested clauses) while strictly preserving the original meaning.\n\n"
        "Constraints:\n"
        "1. Output ONLY the rewritten sentence.\n"
        "2. Do not provide introductory text or explanations.\n"
        "3. The output must be a single sentence of comparable length to the original.\n\n"
        f"Original: {original_text}\n"
        "Rewritten:"
    )

def generate_batch(model, tokenizer, texts, batch_size=8):
    model.eval()
    responses = []
    
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch_texts = texts[i : i + batch_size]
        prompts = [get_prompt(t) for t in batch_texts]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:] 
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        responses.extend([r.strip() for r in decoded])
        
    return responses


print(f"Loading dataset: {DATASET_NAME}...")
ds = load_dataset(DATASET_NAME, split="train")
ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))
original_sentences = ds["text"]

results_df = pd.DataFrame({"original": original_sentences})


print("Loading Sentinel & Semantic Models...")
sentinel = Sentinel()
semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)


results_df["score_original"] = sentinel.difficulty(original_sentences)


print(f"Loading Base Model: {BASE_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
results_df["response_base"] = generate_batch(base_model, tokenizer, original_sentences)
del base_model
torch.cuda.empty_cache()


print(f"Loading Finetuned Checkpoint: {FINETUNED_CHECKPOINT}...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, FINETUNED_CHECKPOINT)
results_df["response_finetuned"] = generate_batch(model, tokenizer, original_sentences)
del model
torch.cuda.empty_cache()


print("Applying Filters...")
valid_flags = []
reasons = []

for idx, row in results_df.iterrows():
    is_valid, reason = validate_response(row['original'], row['response_finetuned'])
    valid_flags.append(is_valid)
    reasons.append(reason)

results_df["is_valid"] = valid_flags
results_df["reason"] = reasons

# calculate metrics for all, subset later
print("Calculating Scores...")

results_df["score_base"] = sentinel.difficulty(results_df["response_base"].tolist())
results_df["score_finetuned"] = sentinel.difficulty(results_df["response_finetuned"].tolist())

embeddings_orig = semantic_model.encode(results_df["original"].tolist())
embeddings_base = semantic_model.encode(results_df["response_base"].tolist())
embeddings_fine = semantic_model.encode(results_df["response_finetuned"].tolist())

results_df["sem_base"] = [util.cos_sim(o, b).item() for o, b in zip(embeddings_orig, embeddings_base)]
results_df["sem_finetuned"] = [util.cos_sim(o, f).item() for o, f in zip(embeddings_orig, embeddings_fine)]

# creating valid subset to compare only where finetuned model passed filters
valid_df = results_df[results_df["is_valid"] == True].copy()

unique_finetuned = len(set(valid_df["response_finetuned"]))
if len(valid_df) > 0:
    uniqueness_ratio = unique_finetuned / len(valid_df)
else:
    uniqueness_ratio = 0.0


print(f"Writing readable logs to {OUTPUT_LOG}...")
with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    f.write("================================================================================\n")
    f.write(f"FILTERED EVALUATION REPORT | {len(valid_df)}/{len(results_df)} Valid Samples\n")
    f.write("================================================================================\n\n")
    
    for i, row in results_df.iterrows():
        status = "[VALID]" if row['is_valid'] else f"[INVALID: {row['reason']}]"
        
        f.write(f"SAMPLE {i+1} {status}\n")
        f.write("-" * 80 + "\n")
        f.write(f"[ORIGINAL]: {row['original']}\n")
        
        f.write(f"[BASE MODEL] (Score: {row['score_base']:.4f})\n")
        f.write(f"{row['response_base']}\n")
        
        f.write(f"[FINETUNED] (Score: {row['score_finetuned']:.4f})\n")
        f.write(f"{row['response_finetuned']}\n")
        
        f.write("\n")

if len(valid_df) > 0:
    print("Plotting results (Valid Samples Only)...")
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.kdeplot(valid_df["score_original"], label="Original", fill=True, alpha=0.3)
    sns.kdeplot(valid_df["score_base"], label="Base Llama", fill=True, alpha=0.3)
    sns.kdeplot(valid_df["score_finetuned"], label="Finetuned", fill=True, alpha=0.3, color='green')
    plt.title("Difficulty Score (Valid Only)")
    plt.xlabel("Sentinel Score (LOWER = Harder)")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.kdeplot(valid_df["sem_base"], label="Base Llama", fill=True, alpha=0.3)
    sns.kdeplot(valid_df["sem_finetuned"], label="Finetuned", fill=True, alpha=0.3, color='green')
    plt.axvline(0.6, color='red', linestyle='--', label="Min Threshold")
    plt.title("Semantic Preservation (Valid Only)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("evaluation_results_filtered.png")
    print("Plot saved.")
    

    print("\n" + "="*40)
    print("FILTERED FINAL REPORT (VALID SAMPLES ONLY)")
    print("="*40)
    print(f"Total Samples:       {len(results_df)}")
    print(f"Valid Samples:       {len(valid_df)} ({len(valid_df)/len(results_df):.1%})")
    print(f"Uniqueness (Valid):  {uniqueness_ratio:.2%}")
    print("-" * 40)
    # stats below are calculated only on valid_df
    print(f"Mean Sentinel (Orig):        {valid_df['score_original'].mean():.4f}")
    print(f"Mean Sentinel (Base):        {valid_df['score_base'].mean():.4f}")
    print(f"Mean Sentinel (Finetuned):   {valid_df['score_finetuned'].mean():.4f}  <-- lower means harder to translate")
    print("-" * 40)
    print(f"Mean Semantics (Base):       {valid_df['sem_base'].mean():.4f}")
    print(f"Mean Semantics (Finetuned):  {valid_df['sem_finetuned'].mean():.4f}")
    print("="*40)

else:
    print("ALL SAMPLES FAILED FILTER. No stats to report.")