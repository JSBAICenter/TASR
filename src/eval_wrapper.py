import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from hotpot_evaluate_v1 import exact_match_score, f1_score  # noqa: E402


def load_gold(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {item["id"]: item["answer"] for item in data}


def evaluate(predictions: dict, gold_path: str) -> dict:
    gold = load_gold(gold_path)

    total_em, total_f1, count = 0, 0, 0
    for qid, gold_answer in gold.items():
        pred = predictions.get(qid, "")
        em = exact_match_score(pred, gold_answer)
        f1, _, _ = f1_score(pred, gold_answer)
        total_em += em
        total_f1 += f1
        count += 1

    return {"em": total_em / count * 100, "f1": total_f1 / count * 100}


if __name__ == "__main__":
    gold = load_gold("data/dev_tune.json")
    dummy_preds = {qid: "dummy answer" for qid in gold}
    result = evaluate(dummy_preds, "data/dev_tune.json")
    print(f"Dummy eval -> EM: {result['em']:.1f}, F1: {result['f1']:.1f}")
