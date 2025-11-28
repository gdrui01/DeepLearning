from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List

DEFAULT_BASE = "gpt2"

class EditGenerator:
    """
    Small causal LM used to *edit* a sentence given an instruction prompt.
    """
    def __init__(self, model_name: str = DEFAULT_BASE, device: str | None = None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

    @torch.inference_mode()
    def edit(self, seeds: List[str], instruction: str,
             max_new_tokens: int = 45,
             temperature: float = 0.9,
             top_p: float = 0.95,
             num_return_sequences: int = 1) -> List[str]:
        """
        Given seeds and an instruction, produce edited sentences.
        Max 45 tokens corresponds to ~30 words (1.5 tokens/word ratio).
        """
        prompts = [f"{instruction}\n\n'{s}'\n\nRewrite it as one grammatical English sentence. Output only the edited sentence, no other text." for s in seeds]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model.generate(**enc,
                                  do_sample=True,
                                  temperature=temperature,
                                  top_p=top_p,
                                  max_new_tokens=max_new_tokens,
                                  num_return_sequences=num_return_sequences,
                                  eos_token_id=self.tok.eos_token_id,
                                  pad_token_id=self.tok.pad_token_id)
        texts = self.tok.batch_decode(out, skip_special_tokens=True)
        # keep only the continuation; if stripping fails, return full text
        results = []
        expanded_prompts = sum([[p]*num_return_sequences for p in prompts], [])
        for p, t in zip(expanded_prompts, texts):
            t = t.strip()
            cand = t[len(p):].strip() if t.startswith(p) else t
            results.append(cand if cand else t)
        return results
