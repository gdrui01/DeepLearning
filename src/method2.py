import argparse, json, os
from typing import List
from tqdm import tqdm

from .gen_model import EditGenerator
from .translate import MTEnDe
from .scorers import COMETQE, LMVerifier
from .constraints import constraint_score
from .losses import method2_loss

DEFAULT_INSTRUCTION = "Make the following sentence harder to translate while keeping it grammatical and natural."

def read_seeds(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="data/seeds.txt")
    ap.add_argument("--out", type=str, default="data/method2_results.jsonl")
    ap.add_argument("--k", type=int, default=100, help="number of seeds to process (<= file size)")
    ap.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    ap.add_argument("--x", type=float, default=1.0)
    ap.add_argument("--y", type=float, default=0.3)
    ap.add_argument("--z", type=float, default=0.3)
    ap.add_argument("--f", type=str, default="relu", choices=["relu","sigmoid","none"])
    args = ap.parse_args()

    seeds = read_seeds(args.seeds)[:args.k]
    if not seeds:
        raise SystemExit(f"No sentences found in {args.seeds}")

    gen = EditGenerator()
    mt = MTEnDe()
    qe = COMETQE()
    vf = LMVerifier()

    # 1) Generate edits
    edits = gen.edit(seeds, args.instruction, num_return_sequences=1)

    # 2) Translate originals and edits
    t0 = mt.translate(seeds)
    t1 = mt.translate(edits)

    # 3) Scores
    de_old = qe.difficulty(seeds, t0)      # 1 - QE(s0, t0)
    de_new = qe.difficulty(edits, t1)      # 1 - QE(s,  t)
    cons = constraint_score(edits)
    ver = vf.score(edits)

    # 4) Loss
    losses = method2_loss(de_new, de_old, cons, ver, x=args.x, y=args.y, z=args.z, f=args.f)

    # 5) Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s0, s1, a, b, c, v, L in zip(seeds, edits, de_old, de_new, cons, ver, losses):
            f.write(json.dumps({
                "orig": s0,
                "edit": s1,
                "de_old": float(a),
                "de_new": float(b),
                "delta_de": float(b - a),
                "constraint": float(c),
                "verify": float(v),
                "L": float(L)
            }) + "\n")

    # 6) Print top edits (lowest L)
    rows = [json.loads(line) for line in open(args.out, "r", encoding="utf-8")]
    rows = sorted(rows, key=lambda r: r["L"])[:10]
    print("\nTop 10 (lowest loss):")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. L={r['L']:.3f}  Δde={r['delta_de']:+.3f}  | {r['edit']}")

if __name__ == "__main__":
    main()
