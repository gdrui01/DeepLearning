import math

from datasets import load_dataset
from sentinel_metric import download_model, load_from_checkpoint

OUT_DIR = "../data/high_quality_english_with_sentinel"
DATASET_NAME = "agentlans/high-quality-english-sentences"
DATASET_SPLIT = "train"


def load_sentinel():
    ckpt = download_model("Prosho/sentinel-src-25")
    model = load_from_checkpoint(ckpt)
    return model


def main():

    sentinel = load_sentinel()

    print(f"Loading dataset: {DATASET_NAME} [{DATASET_SPLIT}]")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)


    sentences = ds["text"]
    n = len(sentences)
    print(f"Dataset loaded with {n} sentences.")

    print("Computing Sentinel scores (one big predict call)...")
    data = [{"src": s} for s in sentences]

    out = sentinel.predict(data, batch_size=32, gpus=1)
    scores = [float(x) for x in out.scores]

    if len(scores) != n:
        raise RuntimeError(
            f"Number of computed scores ({len(scores)}) "
            f"does not match number of sentences ({n})."
        )
    ds_with_scores = ds.add_column("sentinel_score", scores)

    print(f"Saving dataset with scores to: {OUT_DIR}")
    ds_with_scores.save_to_disk(OUT_DIR)

    print("First example with score:")
    print(ds_with_scores[0])


if __name__ == "__main__":
    main()
