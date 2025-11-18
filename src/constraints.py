import re
from typing import List

def constraint_score(sentences: List[str],
                     min_len: int = 4,
                     max_len: int = 40) -> List[float]:
    """
    Simple constraint signal: length window + lexical diversity.
    Returns higher-is-better scores.
    """
    scores = []
    for s in sentences:
        toks = re.findall(r"\w+|\S", s)
        alpha = [t for t in toks if re.match(r"^[A-Za-z]+$", t)]
        n = len(toks)
        length_ok = (min_len <= n <= max_len)
        uniq_ratio = len(set(w.lower() for w in alpha)) / max(1, len(alpha))
        # base on diversity around 0.5, add length bonus
        score = (uniq_ratio - 0.5) + (0.5 if length_ok else -0.5)
        score = min(1.0, max(0.0, score)) # making sure the score is between 0 and 1
        scores.append(score)
    return scores

if __name__ == "__main__":
    good_score = constraint_score(["The cross-eyed cat sat on the mat, patiently waiting for her owner to return."])
    bad_score = constraint_score(["I am going out."])
    print(f"Good score: {good_score}, Bad score: {bad_score}")
