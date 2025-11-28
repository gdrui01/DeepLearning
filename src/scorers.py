import math
import re
from typing import List, Dict
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from comet import download_model, load_from_checkpoint
from langdetect import detect, LangDetectException

# -------- COMET QE (reference-free) --------

QE_MODEL = "Unbabel/wmt22-cometkiwi-da"  # public, supported in unbabel-comet 2.2.x

class COMETQE:
    """
    Wrap COMET QE to compute difficulty = 1 - QE(src, mt).
    """
    def __init__(self, model_name: str = QE_MODEL, accelerator_preference: str = "gpu"):
        ckpt = download_model(model_name)   # handles local cache
        self.model = load_from_checkpoint(ckpt)
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

class LMVerifier:
    """
    Return higher-is-better 'naturalness' via negative perplexity.
    We use a small LM to keep it light.

    Additionally validates:
    - Output is in English
    - Output is exactly one sentence
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

    def _is_english(self, text: str) -> bool:
        """Check if text is in English using language detection."""
        if not text or not text.strip():
            return False
        try:
            lang = detect(text)
            return lang == "en"
        except LangDetectException:
            # If detection fails, assume it's not valid English
            return False

    def _count_sentences(self, text: str) -> int:
        """Count sentences using regex pattern matching."""
        if not text or not text.strip():
            return 0
        # Split on common sentence terminators (., !, ?)
        # Use regex to match sentence endings followed by space or end of string
        sentences = re.split(r'[.!?]+\s+|\s*[.!?]+$', text.strip())
        # Filter out empty strings
        sentences = [s for s in sentences if s.strip()]
        return len(sentences)

    @torch.inference_mode()
    def score(self, sentences: List[str], scale: float = 1.0) -> List[float]:
        """
        Score sentences based on perplexity, language, and sentence count.
        Returns 0.0 if:
        - Text is not in English
        - Text contains more than one sentence
        Otherwise returns perplexity-based score.
        """
        scores = []

        for sentence in sentences:
            pen = False
            sent_score = 0.0
            # Check if empty or invalid
            if not sentence or not sentence.strip():
                sent_score -= 0.1
                pen = True
                # scores.append(0.0)
                # continue

            # Validate it's English
            if not self._is_english(sentence):
                sent_score -= 1.0
                pen = True
                # print(f"Not English: {sentence[:50]}...")
                # scores.append(0.0)
                # continue

            # Validate it's a single sentence
            sentence_count = self._count_sentences(sentence)
            if sentence_count != 1:
                sent_score -= (np.abs(sentence_count - 1)) * 0.1
                pen = True
                # print(f"Expected 1 sentence, got {sentence_count}: {sentence[:50]}...")
                # scores.append(0.0)
                # continue

            # If all validations pass, compute perplexity score
            enc = self.tok([sentence], return_tensors="pt", padding=True).to(self.device)

            if enc["input_ids"].numel() == 0:
                scores.append(-10.0)
                continue

            if pen is False:
                out = self.model(**enc, labels=enc["input_ids"])
                loss = float(out.loss)
                print(f"Loss: {loss}")
                ppl = min(1.0, math.exp(-((loss-5.0)/scale)))
                sent_score += ppl
            scores.append(sent_score)

        return scores

def test_verifier():
    verifier = LMVerifier()

    print("\n=== Testing good English single sentence ===")
    good_score = verifier.score(["Because the storm had intensified rapidly, the coastal town evacuated all residents."])
    print(f"Good score: {good_score}") # Should pass all checks

    print("\n=== Testing bad English (nonsensical but English words) ===")
    bad_score = verifier.score(["Tables green upside flipping extreme sword fight."])
    print(f"Bad score: {bad_score}") # Should get low perplexity score

    print("\n=== Testing gibberish ===")
    very_bad_score = verifier.score(["Jhgjasd dsafjklhref, fkssgh sjh sfkhas fjhfe."])
    print(f"Very bad score: {very_bad_score}") # Should fail language check

    print("\n=== Testing multiple sentences ===")
    multi_sentence = verifier.score(["The cat sat on the mat. The dog ran in the yard."])
    print(f"Multiple sentences score: {multi_sentence}") # Should be 0.0 - too many sentences

    print("\n=== Testing non-English (German) ===")
    german_score = verifier.score(["Die Katze saß auf der Matte."])
    print(f"German score: {german_score}") # Should be 0.0 - not English

    print("\n=== Testing empty string ===")
    empty_score = verifier.score([""])
    print(f"Empty score: {empty_score}") # Should be 0.0

    print("\n=== Testing three sentences ===")
    three_sentences = verifier.score(["First sentence here. Second one too. And a third."])
    print(f"Three sentences score: {three_sentences}") # Should be 0.0

if __name__ == "__main__":
    # test_comet_qe()
    test_verifier()