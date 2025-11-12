from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List

MT_MODEL = "Helsinki-NLP/opus-mt-en-de"

class MTEnDe:
    """
    Simple EN→DE translator using OPUS-MT.
    """
    def __init__(self, name: str = MT_MODEL, device: str | None = None):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

    @torch.inference_mode()
    def translate(self, sentences: List[str], max_new_tokens: int = 64) -> List[str]:
        if not sentences:
            return []
        enc = self.tok(sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4)
        return self.tok.batch_decode(out, skip_special_tokens=True)
