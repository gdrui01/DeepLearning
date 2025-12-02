import math
import re
from typing import List, Dict
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from langdetect import detect, LangDetectException
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
        return [1.0 - score for score in out.scores]

# -------- COMET QE (reference-free) --------

#QE_MODEL = "wmt21-comet-qe-da"
QE_MODEL = "Unbabel/wmt22-cometkiwi-da"  # public, supported in unbabel-comet 2.2.x

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
    sen = Sentinel()
    src_easy = ["The sky was cloudy."] #["The cat sat on the mat."]
    src_harder = ["Going back up tomorrow; we’re doing stalls and coffin corner practice drills to nail it on the checkride."] #["He read a novel that was written by an author who was known for his ability to create intricate and complex characters."]    
    # Test good translation (correct German translation)
    mt_easy = ["Der Himmel war bewölkt."] #["Die Katze saß auf der Matte."]
    mt_harder = ["Morgen geht es wieder hoch; wir machen Übungen zu Strömungsabrissen und dem kritischen Geschwindigkeitsbereich, um das bei der Prüfungsfahrt perfekt hinzubekommen."] #["Er las einen Roman, der von einem Autor geschrieben wurde, der für seine Fähigkeit bekannt war, komplizierte und komplexe Charaktere zu erschaffen."]
    difficulty_qe_easy = qe.difficulty(src_easy, mt_easy)
    difficulty_qe_harder = qe.difficulty(src_harder, mt_harder)
    difficulty_sen_easy = sen.difficulty(src_easy)
    difficulty_sen_harder = sen.difficulty(src_harder)
    print("The higher the score, the harder the translation.")
    print(f"QE score easy: {difficulty_qe_easy}")
    print(f"QE score harder: {difficulty_qe_harder}")
    print(f"Sentinel score easy: {difficulty_sen_easy}")
    print(f"Sentinel score harder: {difficulty_sen_harder}")

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
    test_comet_qe()
    # test_verifier()