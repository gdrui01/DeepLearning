from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import gc

torch.cuda.empty_cache()
gc.collect()

model_name = "Qwen/Qwen3-0.6B-Base" # "gpt2"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token # necessary for gpt2
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = """Rewrite this sentence to be extremely difficult for machine translation using idioms, ambiguity, and wordplay, while keeping it grammatically correct English: "They walked towards the park's entrance."

Only return the single edited sentence."""
# prompt = """Easy to translate sentence: "It was a rainy day."
#         Hard to translate sentence:
# """
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
# )
model_inputs = tokenizer(
    prompt,
    return_tensors="pt",
    # padding=True,
    # truncation=True,
    # max_length=1024
).to(model.device)

# conduct text completion
print("Prompting model...")
generated_ids = model.generate(
    **model_inputs,
    # pad_token_id=tokenizer.pad_token_id,
    # eos_token_id=tokenizer.eos_token_id,
    # do_sample=True,
    # temperature=0.4,
    max_new_tokens=100 # 1024 # 32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# # parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
print("content:", content)
