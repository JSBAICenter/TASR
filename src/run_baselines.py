import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_wrapper import evaluate  # noqa: E402
from llm_client import call_llm  # noqa: E402


def run_closed_book(data_path: str, output_path: str) -> dict:
    with open(data_path) as f:
        data = json.load(f)

    predictions = {}
    for item in tqdm(data, desc=f"Closed-book {Path(data_path).stem}"):
        prompt = (
            "Answer the following question with a short phrase "
            "(a few words only). Do not explain.\n"
            f"Question: {item['question']}\n"
            "Answer:"
        )
        answer = call_llm(prompt)
        predictions[item["id"]] = answer

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    return predictions


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results",
                    help="directory for closed_book_{split}.json outputs")
    args = ap.parse_args()
    for split in ["dev_tune", "dev_eval"]:
        path = f"data/{split}.json"
        out = f"{args.out_dir}/closed_book_{split}.json"
        print(f"\nRunning closed-book on {split} -> {out} ...")
        preds = run_closed_book(path, out)
        result = evaluate(preds, path)
        print(f"{split} -> EM: {result['em']:.1f}%, F1: {result['f1']:.1f}%")
