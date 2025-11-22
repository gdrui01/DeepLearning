from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
print('Default Qwen3 tokenizer settings:')
print(f'  pad_token: {tok.pad_token}')
print(f'  pad_token_id: {tok.pad_token_id}')
print(f'  eos_token: {tok.eos_token}')
print(f'  eos_token_id: {tok.eos_token_id}')
print(f'  padding_side: {tok.padding_side}')
print(f'  vocab_size: {tok.vocab_size}')
