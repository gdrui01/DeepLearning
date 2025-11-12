import math
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from comet import download_model, load_from_checkpoint

# -------- COMET QE (reference-free) --------

QE_MODEL = "wmt21-comet-qe-da"  # public, supported in unbabel-comet 2.2.x

class COMETQE:
    """
    Wrap COMET QE to compute difficulty = 1 - QE(src, mt).
    """
    def __init__(self, model_name: str = QE_MODEL):
        ckpt = download_model(model_name)   # handles local cache
        self.model = load_from_checkpoint(ckpt)
        try:
            if hasattr(self.model, "trainer"):
                self.model.trainer.logger = False
        except Exception:
            pass

    def _build(self, src_list: List[str], mt_list: List[str]) -> List[Dict[str, str]]:
        data = []
        for s, t in zip(src_list, mt_list):
            s = (s or "").strip()
            t = (t or "").strip()
            if s and t:
                data.append({"src": s, "mt": t})
        return data

    def difficulty(self, src_list: List[str], mt_list: List[str]) -> List[float]:
        data = self._build(src_list, mt_list)
        if not data:
            return [0.0] * len(src_list)

        # Force CPU first for stability with lightning signatures
        try:
            pred = self.model.predict(data, batch_size=16, accelerator="cpu", devices=1)
            scores = pred.scores if hasattr(pred, "scores") else pred
        except TypeError:
            pred = self.model.predict(data, batch_size=16, gpus=0)
            scores = pred.scores if hasattr(pred, "scores") else pred

        # map back (difficulty = 1 - qe)
        it = iter(float(s) for s in scores)
        out = []
        for s, t in zip(src_list, mt_list):
            if (s or "").strip() and (t or "").strip():
                out.append(1.0 - next(it))
            else:
                out.append(0.0)
        return out

# -------- Verifier (LM perplexity) --------

VERIFIER_LM = "distilgpt2"

class LMVerifier:
    """
    Return higher-is-better 'naturalness' via negative perplexity.
    We use a small LM to keep it light.
    """
    def __init__(self, name: str = VERIFIER_LM, device: str | None = None):
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

    @torch.inference_mode()
    def score(self, sentences: List[str]) -> List[float]:
        enc = self.tok(sentences, return_tensors="pt", padding=True).to(self.device)
        out = self.model(**enc, labels=enc["input_ids"])
        # one scalar loss for the batch; broadcast for simplicity
        loss = float(out.loss)
        ppl = math.exp(loss)
        return [-ppl] * len(sentences)  # higher is better
