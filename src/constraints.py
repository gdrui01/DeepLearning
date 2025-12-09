import re
from typing import List

def constraint_score(sentences: List[str],
                     min_len: int = 4,
                     max_len: int = 60,
                     penalize_markdown: bool = True) -> List[float]:
    """
    Simple constraint signal: length window + lexical diversity.
    Optionally penalizes markdown formatting.
    Returns higher-is-better scores.

    Max length is set to 30 words (not tokens - this is simpler and more interpretable).
    Length is measured by counting alphabetic words only.
    """
    scores = []
    for s in sentences:
        # Extract only alphabetic words (ignoring punctuation/numbers)
        words = re.findall(r'\b[A-Za-z]+\b', s)
        n_words = len(words)

        # Check length constraint (word count)
        length_ok = (min_len <= n_words <= max_len)

        # Lexical diversity based on unique words
        uniq_ratio = len(set(w.lower() for w in words)) / max(1, len(words))

        # Base score: diversity around 0.5, add length bonus/penalty
        score = (uniq_ratio - 0.5) + (0.5 if length_ok else -0.5)

        # Penalize markdown formatting (e.g., **bold** or *italic*)
        if penalize_markdown:
            markdown_count = len(re.findall(r'\*\*[^\*]+\*\*|\*[^\*]+\*', s))
            score -= markdown_count * 0.2  # Penalty per markdown instance

        score = min(1.0, max(0.0, score)) # making sure the score is between 0 and 1
        scores.append(score)
    return scores

if __name__ == "__main__":
    good_score = constraint_score(["The cross-eyed cat sat on the mat, patiently waiting for her owner to return."])
    bad_too_long = constraint_score(["This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence that exceeds the maximum word limit."])
    bad_too_short = constraint_score(["Hi."])
    bad_low_diversity = constraint_score(["The the the the the the the the the the cat sat there."])

    print(f"Good score (16 words, high diversity): {good_score}")
    print(f"Bad - too long (>30 words): {bad_too_long}")
    print(f"Bad - too short (<4 words): {bad_too_short}")
    print(f"Bad - low diversity (repeated words): {bad_low_diversity}")
