import math
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentinel_metric import download_model as sentinel_download_model, load_from_checkpoint as sentinel_load_from_checkpoint

from comet import download_model as comet_download_model, load_from_checkpoint as comet_load_from_checkpoint



# cloned locally instead of pip install because of issues with sentinel_metric
# https://huggingface.co/Prosho/sentinel-src-25
class Sentinel:
    def __init__(self):
        ckpt = sentinel_download_model("Prosho/sentinel-src-25")
        self.model = sentinel_load_from_checkpoint(ckpt)

    def difficulty(self, sentences: List[str]) -> List[float]:
        data = [{"src": s} for s in sentences]
        out = self.model.predict(data, batch_size = 8, gpus = 1)
        return out.scores
    
    # depending on how we use in pipeline need to adjust behavior for a whole batch vs single example
    # using tanh to map also to negative values and penalize decreases in difficulty
    def get_difference_tanh(self, de_before, de_after):
        delta = de_before - de_after
        score = math.tanh(delta) # apply on delta and not individual scores such that same delta always maps to same score
        return score



# -------- COMET QE (reference-free) --------

QE_MODEL = "wmt21-comet-qe-da"
#QE_MODEL = "Unbabel/wmt22-cometkiwi-da"
class COMETQE:
    """
    Wrap COMET QE to compute difficulty = 1 - QE(src, mt).
    """
    def __init__(self, model_name: str = QE_MODEL, accelerator_preference: str = "gpu"):
        ckpt = comet_download_model(model_name)   # handles local cache
        self.model = comet_load_from_checkpoint(ckpt)
        self.accelerator_preference = accelerator_preference
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

        # Try GPU first, fall back to CPU if needed
        # Build ordered list of accelerator configs to try
        pref = (self.accelerator_preference or "gpu").lower()
        attempts = []
        if pref == "cpu":
            attempts.extend([
                {"accelerator": "cpu", "devices": 1},
                {"gpus": 0},
            ])
        else:
            attempts.extend([
                {"accelerator": "gpu", "devices": 1},
                {"gpus": 1},
            ])
        # Always add CPU fallbacks last
        attempts.extend([
            {"accelerator": "cpu", "devices": 1},
            {"gpus": 0},
        ])

        last_err = None
        for cfg in attempts:
            try:
                pred = self.model.predict(data, batch_size=16, **cfg)
                scores = pred.scores if hasattr(pred, "scores") else pred
                break
            except (TypeError, AssertionError, ValueError, RuntimeError) as e:
                last_err = e
        else:
            raise RuntimeError("COMET QE prediction failed for all accelerator configs") from last_err

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
# from https://www.geeksforgeeks.org/nlp/perplexity-for-llm-evaluation/
class LMVerifier:
    """
    Return higher-is-better 'naturalness' via perplexity. We map perplexity to [0,1].
    Change P_min and P_max to adjust the mapping.
    We use a small LM to keep it light.
    """
    def __init__(self, name: str = VERIFIER_LM, device: str | None = None,
                P_min: float = 40.0,   # "good" perplexity
                P_max: float = 200.0,  # "bad" perplexity maybe change to 300
                ):
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device
        self.log_P_min = math.log(P_min)
        self.log_P_max = math.log(P_max)

    @torch.inference_mode()
    def score(self, sentences: List[str], scale: float = 1.0) -> List[float]:
        # Handle empty or invalid sentences
        valid_sentences = [s if s and s.strip() else "." for s in sentences]

        enc = self.tok(
            valid_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        if enc["input_ids"].numel() == 0:
            return [0.0] * len(sentences)

        input_ids = enc["input_ids"] # [B, L]
        attention_mask = enc["attention_mask"] # [B, L]

        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits # [B, L, V]

        # shift for next-token prediction
        shift_logits = logits[:, :-1, :]  # [B, L-1, V]
        shift_labels = input_ids[:, 1:] # [B, L-1]
        shift_mask = attention_mask[:, 1:] # [B, L-1]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1) # [B, L-1]

        # mask out padding
        token_log_probs = token_log_probs * shift_mask

        lengths = shift_mask.sum(dim=-1)
        # maybe not necessary
        lengths = torch.clamp(lengths, min=1)

        # negative log-likelihood per sentence
        nll = -(token_log_probs.sum(dim=-1) / lengths)

        # calculated perplexity
        ppl = torch.exp(nll)
        scores = []


        # penalize if model creates multiple sentences?
        for p in ppl.cpu().tolist():
            p = max(1e-6, float(p))
            lp = math.log(p)

            # linear mapping in log space
            raw = (self.log_P_max - lp) / (self.log_P_max - self.log_P_min)
            s = max(0.0, min(1.0, raw))

            # included threshold map to not penalize (we could have a sentence that makes sense but is hard to translate which we don't want to penalize)
            #if s >= 0.3:
            #    s = 1.0
            scores.append(s)

        return scores

def test_verifier():
    verifier = LMVerifier()
    good_score = verifier.score(["Because the storm had intensified rapidly, the coastal town evacuated all residents."])
    print(f"Good score: {good_score}")
    bad_score = verifier.score(["Tables green upside flipping extreme sword fight."])
    print(f"Bad score: {bad_score}")
    very_bad_score = verifier.score(["Jhgjasd dsafjklhref, fkssgh sjh sfkhas fjhfe."])
    print(f"Very bad score: {very_bad_score}")

if __name__ == "__main__":
    # test_comet_qe()
    test_verifier()