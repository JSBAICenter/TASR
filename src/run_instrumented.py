"""Instrumented k=5 replay (Task 2.6).

Walks `AgentState` for up to MAX_ROUNDS rounds per question under semantics (a):
each round adds the next BM25-ranked paragraph and asks the LLM for
(answer, confidence). No reformulation. One LLM call per round.

Per-round rows are saved to parquet with the columns Day 3 needs:
  qid, question_type, round,
  top_bm25_score, rank2_bm25_score, gap_1_2,
  jaccard_overlap, llm_confidence,
  current_answer, current_em, current_f1

Cache reuse: round-1 prompts match `run_fixed_k.py k=1`; round-3 matches k=3;
round-5 matches k=5 (because evidence list is identical and `format_evidence`
is shared). Fresh LLM calls only at rounds 2 and 4 on dev_eval, plus all
rounds on dev_tune.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentState  # noqa: E402
from eval_wrapper import exact_match_score, f1_score  # noqa: E402
from retrieval import build_paragraphs, tokenize  # noqa: E402


MAX_ROUNDS = 5


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def run_instrumented(data_path: str, output_path: str) -> pd.DataFrame:
    with open(data_path) as f:
        data = json.load(f)

    rows: list[dict] = []
    for item in tqdm(data, desc=f"instrumented {Path(data_path).stem}"):
        qid = item["id"]
        gold = item["answer"]
        qtype = item.get("type", "")
        paragraphs = build_paragraphs(item)

        state = AgentState(question=item["question"], paragraphs=paragraphs)
        # Static per-question signals (BM25 over original question).
        top_score = state.ranked[0]["score"] if state.ranked else 0.0
        rank2_score = state.ranked[1]["score"] if len(state.ranked) > 1 else 0.0
        gap_1_2 = top_score - rank2_score

        prior_tokens: set[str] = set()
        for _ in range(MAX_ROUNDS):
            added = state.search(n=1)
            new_tokens: set[str] = set()
            for p in added:
                new_tokens.update(tokenize(p["title"] + " " + p["text"]))
            overlap = jaccard(new_tokens, prior_tokens)

            ans, conf = state.answer()
            em = exact_match_score(ans, gold)
            f1, _, _ = f1_score(ans, gold)

            rows.append(
                {
                    "qid": qid,
                    "question_type": qtype,
                    "round": state.round_num,
                    "top_bm25_score": top_score,
                    "rank2_bm25_score": rank2_score,
                    "gap_1_2": gap_1_2,
                    "jaccard_overlap": overlap,
                    "llm_confidence": conf,
                    "current_answer": ans,
                    "current_em": float(em),
                    "current_f1": float(f1),
                }
            )

            prior_tokens.update(new_tokens)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    by_round = df.groupby("round").agg(
        n=("qid", "count"),
        em=("current_em", "mean"),
        f1=("current_f1", "mean"),
        conf=("llm_confidence", "mean"),
        overlap=("jaccard_overlap", "mean"),
    )
    by_round["em"] = (by_round["em"] * 100).round(1)
    by_round["f1"] = (by_round["f1"] * 100).round(1)
    by_round["conf"] = by_round["conf"].round(2)
    by_round["overlap"] = by_round["overlap"].round(3)
    print(by_round.to_string())
    print(f"NaN cells: {int(df.isna().sum().sum())}")
    print(f"confidence value counts:\n{df['llm_confidence'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tune", default="results/signal_log_tune.parquet")
    ap.add_argument("--out-eval", default="results/signal_log.parquet")
    args = ap.parse_args()
    splits = [
        ("data/dev_tune.json", args.out_tune),
        ("data/dev_eval.json", args.out_eval),
    ]
    for data_path, out_path in splits:
        df = run_instrumented(data_path, out_path)
        summary(df, Path(data_path).stem)
        print(f"wrote {out_path}: {len(df)} rows")
