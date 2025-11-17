import math
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from comet import download_model, load_from_checkpoint

# -------- COMET QE (reference-free) --------

QE_MODEL = "Unbabel/wmt22-cometkiwi-da"  # public, supported in unbabel-comet 2.2.x

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

    def _get_qe_scores(self, src_list: List[str], mt_list: List[str]) -> List[float]:
        """Get raw QE scores (higher is better)."""
        data = self._build(src_list, mt_list)
        if not data:
            return [0.0] * len(src_list)

        # Force CPU first for stability with lightning signatures
        try:
            pred = self.model.predict(data, batch_size=16, accelerator="cpu", devices=[1])
            scores = pred.scores if hasattr(pred, "scores") else pred
        except (TypeError, AssertionError):
            pred = self.model.predict(data, batch_size=16, gpus=0)
            scores = pred.scores if hasattr(pred, "scores") else pred

        # Handle different return types from COMET
        if isinstance(scores, list):
            score_list = [float(s) for s in scores]
        elif hasattr(scores, "__iter__") and not isinstance(scores, str):
            score_list = [float(s) for s in scores]
        else:
            score_list = [float(scores)]

        # map back to original input order
        it = iter(score_list)
        out = []
        for s, t in zip(src_list, mt_list):
            if (s or "").strip() and (t or "").strip():
                out.append(next(it))
            else:
                out.append(0.0)
        return out

    def difficulty(self, src_list: List[str], mt_list: List[str]) -> List[float]:
        """Compute difficulty = 1 - QE(src, mt). Lower difficulty is better."""
        qe_scores = self._get_qe_scores(src_list, mt_list)
        return [1.0 - score for score in qe_scores]

def test_comet_qe():
    qe = COMETQE()
    src = ["The cat sat on the mat."]    
    # Test good translation (correct German translation)
    mt_good = ["Die Katze saß auf der Matte."]
    difficulty_good = qe.difficulty(src, mt_good)
    print(f"Good mt: {difficulty_good}")

    # Test good translation, but with a typo
    mt_good_typo = ["Die Katze sas auf der Matte."]
    difficulty_good_typo = qe.difficulty(src, mt_good_typo)
    print(f"Good mt_typo: {difficulty_good_typo}")
    
    # Test bad translation (wrong meaning)
    mt_bad = ["Der Hund rannte im Garten."] 
    difficulty_bad = qe.difficulty(src, mt_bad)
    print(f"Bad mt: {difficulty_bad}")
    
    # Note: The "bad" translation is grammatically valid German (just wrong meaning),
    # so the difference might be smaller than expected. For a more dramatic difference,
    # try a translation with grammatical errors or nonsensical output.
    print("Test complete.")

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
    def score(self, sentences: List[str], scale: float = 1.0) -> List[float]:
        # Handle empty or invalid sentences
        valid_sentences = [s if s and s.strip() else "." for s in sentences]

        enc = self.tok(valid_sentences, return_tensors="pt", padding=True).to(self.device)

        # Handle edge case of empty tokenization
        if enc["input_ids"].numel() == 0:
            return [0.0] * len(sentences)

        out = self.model(**enc, labels=enc["input_ids"])
        # one scalar loss for the batch; broadcast for simplicity
        loss = float(out.loss)
        print(f"Loss: {loss}")
        ppl = min(1.0, math.exp(-((loss-5.0)/scale)))
        return [ppl] * len(sentences)  # higher is better

def test_verifier():
    verifier = LMVerifier()
    good_score = verifier.score(["Because the storm had intensified rapidly, the coastal town evacuated all residents."])
    print(f"Good score: {good_score}") # 1.0
    bad_score = verifier.score(["Tables green upside flipping extreme sword fight."])
    print(f"Bad score: {bad_score}") # 0.004
    very_bad_score = verifier.score(["Jhgjasd dsafjklhref, fkssgh sjh sfkhas fjhfe."])
    print(f"Very bad score: {very_bad_score}") # 0.35 (!!) (Although gibberish, the score
    # is still higher than the bad score, because the model splits the sentence into
    # much smaller tokens that are still "somewhat" meaningful.)

if __name__ == "__main__":
    # test_comet_qe()
    test_verifier()