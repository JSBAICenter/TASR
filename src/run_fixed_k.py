import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_wrapper import evaluate  # noqa: E402
from llm_client import call_llm  # noqa: E402
from retrieval import build_paragraphs, retrieve  # noqa: E402


ANSWER_PROMPT = """Based on the evidence below, answer the question with a short phrase (a few words only).
Then rate your confidence from 1 to 5:
  1 = pure guess, 2 = unlikely correct, 3 = maybe, 4 = fairly confident, 5 = certain

Question: {question}
Evidence:
{evidence_text}

Respond in exactly this format:
Answer: <your answer>
Confidence: <1-5>"""


def parse_answer_confidence(response: str) -> tuple[str, int]:
    ans = re.search(r"Answer:\s*(.+)", response)
    conf = re.search(r"Confidence:\s*(\d)", response)
    answer = ans.group(1).strip() if ans else response.strip().split("\n")[0]
    confidence = int(conf.group(1)) if conf else 3
    return answer, max(1, min(5, confidence))


def format_evidence(passages: list[dict]) -> str:
    parts = []
    for p in passages:
        parts.append(f"[{p['title']}] {p['text']}")
    return "\n\n".join(parts)


def run_fixed_k(data_path: str, k: int, output_path: str) -> dict:
    with open(data_path) as f:
        data = json.load(f)

    predictions = {}
    confidences = {}
    for item in tqdm(data, desc=f"k={k} {Path(data_path).stem}"):
        paragraphs = build_paragraphs(item)
        top_k = retrieve(item["question"], paragraphs, k=k)
        evidence = format_evidence(top_k)
        prompt = ANSWER_PROMPT.format(question=item["question"], evidence_text=evidence)
        response = call_llm(prompt, max_tokens=128)
        answer, confidence = parse_answer_confidence(response)
        predictions[item["id"]] = answer
        confidences[item["id"]] = confidence

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"predictions": predictions, "confidences": confidences}, f, indent=2)

    return predictions


if __name__ == "__main__":
    split = "dev_eval"
    path = f"data/{split}.json"
    rows = []
    for k in [1, 3, 5]:
        out = f"results/fixed_k{k}_{split}.json"
        print(f"\nRunning fixed k={k} on {split}...")
        preds = run_fixed_k(path, k, out)
        result = evaluate(preds, path)
        print(f"k={k} {split} -> EM: {result['em']:.1f}%, F1: {result['f1']:.1f}%")
        rows.append({"k": k, "split": split, **result})

    print("\n=== Fixed-k summary ===")
    print(f"{'k':<4}{'EM':>8}{'F1':>8}")
    for r in rows:
        print(f"{r['k']:<4}{r['em']:>8.1f}{r['f1']:>8.1f}")
