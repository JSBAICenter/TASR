"""Cross-encoder reranker signal as a replacement for weak BM25 stats.

For each (qid, round) row, score the question against each paragraph in the
agent's current evidence (top-`round` BM25 paragraphs) using a free, off-the-shelf
ms-marco cross-encoder. Add two signals:

- ce_top_score:        max CE score over the round-k paragraphs
- ce_top_score_z_intra: per-qid z-score of ce_top_score (so we get a "this is
                       the best CE score we've seen for this question so far"
                       signal that doesn't depend on the question's absolute scale)

No new LLM calls. CPU-only. ~22M-param model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval import build_paragraphs, retrieve  # noqa: E402


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (a.std() + b.std()) / 2 + 1e-9
    return abs(a.mean() - b.mean()) / pooled


def load_question_data(data_path: str) -> dict[str, dict]:
    with open(data_path) as f:
        items = json.load(f)
    out: dict[str, dict] = {}
    for item in items:
        paragraphs = build_paragraphs(item)
        ranked = retrieve(item["question"], paragraphs, k=len(paragraphs))
        out[item["id"]] = {"question": item["question"], "ranked": ranked}
    return out


def score_all_pairs(
    model: CrossEncoder, qdata: dict[str, dict], max_round: int = 5, batch_size: int = 64
) -> dict[tuple[str, int], float]:
    """Score (question, paragraph) for every qid and every paragraph it might see."""
    pairs: list[tuple[str, str]] = []
    keys: list[tuple[str, int]] = []
    for qid, meta in qdata.items():
        for r_idx in range(min(max_round, len(meta["ranked"]))):
            pairs.append((meta["question"], meta["ranked"][r_idx]["text"]))
            keys.append((qid, r_idx))
    t0 = time.time()
    print(f"Scoring {len(pairs)} (question, paragraph) pairs with batch_size={batch_size}...", flush=True)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({len(pairs)/elapsed:.1f} pairs/s)", flush=True)
    return {k: float(s) for k, s in zip(keys, scores)}


def add_ce_signals(df: pd.DataFrame, pair_scores: dict[tuple[str, int], float]) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy().reset_index(drop=True)
    ce_top = np.zeros(len(df))
    for i, row in df.iterrows():
        qid = row["qid"]
        r = int(row["round"])
        scores_r = [pair_scores.get((qid, j), -1e9) for j in range(r)]
        ce_top[i] = max(scores_r) if scores_r else 0.0
    df["ce_top_score"] = ce_top
    # z-score within each qid (per-question normalization)
    df["ce_top_score_z_intra"] = df.groupby("qid")["ce_top_score"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9)
    )
    return df


def report(df: pd.DataFrame, label: str) -> None:
    print(f"\n[{label}] Cohen's d (EM-correct vs incorrect):")
    for sig in ["ce_top_score", "ce_top_score_z_intra"]:
        cor = df[df["current_em"] == 1][sig].astype(float)
        inc = df[df["current_em"] == 0][sig].astype(float)
        d = cohens_d(cor, inc)
        print(f"  {sig:30s}  d={d:6.3f}  mean(cor)={cor.mean():8.3f}  mean(inc)={inc.mean():8.3f}")
    print(f"\n[{label}] Per-round mean ce_top_score:")
    print(df.groupby("round")["ce_top_score"].mean().round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-parquet", default="results/signal_log_enriched.parquet")
    ap.add_argument("--tune-parquet", default="results/signal_log_enriched_tune.parquet")
    ap.add_argument("--data-eval", default="data/dev_eval.json")
    ap.add_argument("--data-tune", default="data/dev_tune.json")
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    print(f"Loading CrossEncoder: {args.model}")
    model = CrossEncoder(args.model, max_length=512)
    print("  loaded.")

    for label, parquet, data_path in [
        ("tune", args.tune_parquet, args.data_tune),
        ("eval", args.eval_parquet, args.data_eval),
    ]:
        print(f"\n{'=' * 70}\nProcessing {label}: {parquet}\n{'=' * 70}")
        qdata = load_question_data(data_path)
        pair_scores = score_all_pairs(model, qdata, max_round=5, batch_size=args.batch_size)
        df = pd.read_parquet(parquet)
        df = add_ce_signals(df, pair_scores)
        report(df, label)
        df.to_parquet(parquet, index=False)
        print(f"Wrote {parquet}  {df.shape}")


if __name__ == "__main__":
    main()
