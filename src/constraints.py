import re
import math
from collections import Counter
from typing import List

def constraint_score(
    sentences: List[str],
    min_len: int = 8,
    max_len: int = 200, # we can tune min_len and max_len using the length distribution of the dataset
) -> List[float]:
    scores = []
    for s in sentences:
        toks = re.findall(r"\w+|\S", s)
        alpha = [t for t in toks if re.match(r"^[A-Za-z]+$", t)]
        n = len(alpha)

        if n == 0:
            scores.append(0.0)
            continue

        # 1) length score
        if min_len <= n <= max_len:
            len_score = 1.0
        else:
            if n < min_len:
                rel_dev = (min_len - n) / min_len
                k = 0.5  # tunable: lower = stronger penalty, higher = softer
                len_score = 1.0 / (1.0 + rel_dev / k)
            else:  # n > max_len
                rel_dev = (n - max_len) / max_len

                k = 0.5  # tunable: lower = stronger penalty, higher = softer
                len_score = 1.0 / (1.0 + rel_dev / k)

        # 2) diversity / repetition
        counts = Counter(w.lower() for w in alpha)
        uniq_ratio = len(counts) / n
        max_rep = max(counts.values())
        rep_score = 1.0 - (max_rep - 1) / n
        rep_score = min(1.0, max(0.0, rep_score))
        w_uniq = 0.5
        w_rep = 0.5 # we can tune this
        div_score = w_uniq * uniq_ratio + w_rep * rep_score

        # 3) character sanity: lower score if there are many non-alphabetic characters
        letters = sum(c.isalpha() for c in s)
        nonspace = sum(not c.isspace() for c in s)
        char_ratio = letters / max(1, nonspace)
        char_score = min(1.0, max(0.0, char_ratio)) 

        # 4) mild complexity bonus: should avoid simple sentences with no conjuction like "I am going out today."
        complex_markers = complex_markers = {
            ",", ";",
            "because", "although", "though", "whereas", "while", "since",
            "if", "when", "whenever", "unless", "until", "before", "after",
            "that", "which", "who", "whom", "whose"
        }
        has_complexity = any(tok.lower() in complex_markers for tok in toks)
        complexity_score = 1.0 if has_complexity else 0.8

        # 5) single sentence constraint: penalize if not exactly one sentence
        # Count sentences by finding sentence-ending punctuation (., !, ?)
        sentence_endings = re.findall(r'[.!?]', s)
        num_sentences = len(sentence_endings)
        if num_sentences == 1:
            sentence_count_score = 1.0
        else:
            # Penalize based on deviation from 1 sentence
            deviation = abs(num_sentences - 1)
            k = 0.5  # tunable: controls penalty strength
            sentence_count_score = 1.0 / (1.0 + deviation / k)

        w_len, w_div, w_char, w_comp, w_sent = 3, 3, 2, 2, 3 # we can tune these weights
        score = ( # during training we can look for which of the constraints make sense and if we need all of them
            w_len * len_score +
            w_div * div_score +
            w_char * char_score +
            w_comp * complexity_score +
            w_sent * sentence_count_score
        ) / (w_len + w_div + w_char + w_comp + w_sent)
        print(f"len_score: {len_score}, div_score: {div_score}, char_score: {char_score}, complexity_score: {complexity_score}, sentence_count_score: {sentence_count_score}")

        scores.append(score)

    return scores

if __name__ == "__main__":
    good_score = constraint_score(["The cross-eyed cat sat on the mat, patiently waiting for her owner to return."])
    print(f"Good score: {good_score}")
    print("-------------------------------")
    too_short = constraint_score(["I am going."])
    print(f"Too short(3 words, 8 min_length): {too_short}")
    print("-------------------------------")
    too_long = constraint_score(["Although the experimental setup appeared straightforward to the new interns, the senior researcher patiently walked them through every single calibration step, explaining how minor measurement errors, overlooked cable connections, or slightly misaligned sensors could propagate through the entire data pipeline and ultimately produce translations that sounded fluent on the surface yet subtly distorted critical technical details in ways that would be almost impossible to detect without systematic evaluation."])
    print(f"Too long (68 words, 60 max_length): {too_long}")
    print("-------------------------------")
    low_diverity = constraint_score(["word word word word word word word word word"])
    print(f"Low diversity: {low_diverity}")
    print("-------------------------------")
    few_characters = constraint_score(["!!! ....... ,,,,,,,, 1234 ### 5678 @@@ 9999 %%% The cross eyed cat sat on the mat"])
    print(f"Few characters: {few_characters}")
    print("-------------------------------")
    no_conjunction = constraint_score(["The weather was terrible and the hikers stayed home."])
    print(f"No conjunction: {no_conjunction}")
    print("-------------------------------")
  