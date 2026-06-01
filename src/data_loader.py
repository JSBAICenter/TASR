import json
import random
from pathlib import Path

from datasets import load_dataset


def download_and_split(seed: int = 42, n_total: int = 400, n_tune: int = 100) -> None:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    sampled = [ds[i] for i in indices[:n_total]]

    dev_tune = sampled[:n_tune]
    dev_eval = sampled[n_tune:]

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "dev_tune.json", "w") as f:
        json.dump(dev_tune, f)
    with open(data_dir / "dev_eval.json", "w") as f:
        json.dump(dev_eval, f)

    print(f"Saved {len(dev_tune)} tune, {len(dev_eval)} eval questions")

    q = dev_tune[0]
    print(f"\nSample question: {q['question']}")
    print(f"Answer: {q['answer']}")
    print(f"Type: {q['type']}")
    print(f"Num context paragraphs: {len(q['context']['title'])}")


if __name__ == "__main__":
    download_and_split()
