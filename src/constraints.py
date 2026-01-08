import re
from collections import Counter
from typing import List

def constraint_score(sentences: List[str], ref_sentences: List[str]) -> List[dict]:
    """
    Returns a dict of sub-scores for analysis, rather than a single float.
    Pass 'original_text' as ref_sentences to compare lengths.
    """
    results = []
    
    for s, ref in zip(sentences, ref_sentences):
        # Basic Tokenization
        toks = re.findall(r"\w+|\S", s)
        alpha = [t for t in toks if re.match(r"^[A-Za-z]+$", t)]
        n = len(alpha)
        
        # 0. Fail safe for empty strings
        if n == 0:
            results.append({'len_score': 0, 'char_score': 0, 'total': 0})
            continue

        # 1. Relative Length Score (Better than absolute min/max)
        # We want the rewrite to be roughly the same length as the original
        ref_len = len(re.findall(r"[A-Za-z]+", ref))
        ratio = n / max(1, ref_len)
        
        # Penalty if length matches poorly (e.g. < 50% or > 200% of original)
        if 0.5 <= ratio <= 2.0:
            len_score = 1.0
        else:
            len_score = 0.0 

        # 2. Diversity / Repetition
        counts = Counter(w.lower() for w in alpha)
        # Check if any single word appears more than 40% of the time (repetition loop)
        most_common_ratio = counts.most_common(1)[0][1] / n
        if most_common_ratio > 0.4 and n > 5:
            div_score = 0.0 # Strict fail
        else:
            div_score = 1.0

        # 3. Character Sanity
        letters = sum(c.isalpha() for c in s)
        nonspace = sum(not c.isspace() for c in s)
        char_ratio = letters / max(1, nonspace)
        char_score = 1.0 if char_ratio > 0.7 else 0.0 # Thresholding is safer

        results.append({
            'len_score': len_score,
            'div_score': div_score,
            'char_score': char_score
        })
        
    return results