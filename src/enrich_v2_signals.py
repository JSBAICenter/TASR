"""Add §2 (IDF-weighted Jaccard) and §3 (robust median/IQR z-score) signals.

Both are deterministic functions of the BM25 retrieval over the 10-paragraph
distractor pool — recomputable without any LLM re-run. Loads an existing
enriched parquet and writes two new columns:

  - overlap_signal_idf   : IDF-weighted Jaccard of new vs prior evidence tokens.
                           Per-question IDF computed over the 10 distractor
                           paragraphs (not a global corpus). Round 1 == 0 by
                           construction (no prior evidence).
  - top_score_z_robust   : (top_bm25_score - median(pool)) / IQR(pool) per qid,
                           median/IQR over the 10 paragraphs of this question.
                           Drop-in for top_score_z_intra. IQR=0 -> 0.0.

Usage:
  python src/enrich_v2_signals.py
  python src/enrich_v2_signals.py --eval results/signal_log_lp_enriched.parquet \
                                  --tune results/signal_log_lp_enriched_tune.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval import build_paragraphs, retrieve, tokenize  # noqa: E402


def compute_per_qid_signals(data_path: str) -> dict[str, dict]:
    """Replay the deterministic BM25 trajectory and compute new signals.

    Returns {qid: {"robust_z": float, "round_idf_jaccard": {round: value}}}.
    """
    with open(data_path) as f:
        items = json.load(f)
    out: dict[str, dict] = {}
    for item in items:
        qid = item["id"]
        paragraphs = build_paragraphs(item)
        ranked = retrieve(item["question"], paragraphs, k=len(paragraphs))
        scores = np.array([r["score"] for r in ranked])

        # §3: robust z-score (median/IQR) over the 10-paragraph pool.
        med = float(np.median(scores))
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = float(q3 - q1)
        robust_z = 0.0 if iqr == 0 else float((scores[0] - med) / iqr)

        # §2: per-question IDF over the 10 paragraphs.
        # idf[t] = log((N + 1) / (df[t] + 1)) + 1  (smooth, positive).
        N = len(paragraphs)
        token_sets = [set(tokenize(p["title"] + " " + p["text"])) for p in paragraphs]
        df_count: dict[str, int] = {}
        for ts in token_sets:
            for t in ts:
                df_count[t] = df_count.get(t, 0) + 1
        idf = {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df_count.items()}

        # Replay the agent: round r adds paragraph at rank r-1 (0-indexed).
        # IDF-Jaccard at round r = overlap of new-tokens(rank r-1) vs prior union(ranks 0..r-2).
        # round 1 has no prior -> 0.0 by construction.
        round_idf: dict[int, float] = {}
        prior_tokens: set[str] = set()
        # The agent calls state.search(n=1) MAX_ROUNDS=5 times. Use the same ordering.
        # ranked is sorted by score desc; tokens for the rank-i paragraph are token_sets at the
        # index matching ranked[i] back to paragraphs.
        # Easier: re-tokenize ranked[i] directly (same content, just title+text).
        for r_idx in range(min(5, len(ranked))):
            new_tokens = set(tokenize(ranked[r_idx]["title"] + " " + ranked[r_idx]["text"]))
            inter = new_tokens & prior_tokens
            union = new_tokens | prior_tokens
            if not union:
                jacc_idf = 0.0
            else:
                num = sum(idf.get(t, 1.0) for t in inter)
                den = sum(idf.get(t, 1.0) for t in union)
                jacc_idf = float(num / den) if den > 0 else 0.0
            round_idf[r_idx + 1] = jacc_idf
            prior_tokens.update(new_tokens)

        out[qid] = {"robust_z": robust_z, "round_idf_jaccard": round_idf}
    return out


def enrich(parquet_path: str, data_path: str, out_path: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {parquet_path}  ({df.shape})")
    info = compute_per_qid_signals(data_path)

    df["top_score_z_robust"] = df["qid"].map(lambda q: info[q]["robust_z"]).astype(float)
    df["overlap_signal_idf"] = df.apply(
        lambda row: info[row["qid"]]["round_idf_jaccard"].get(int(row["round"]), 0.0), axis=1
    ).astype(float)

    # Sanity checks.
    r1_idf = df.loc[df["round"] == 1, "overlap_signal_idf"]
    assert (r1_idf == 0).all(), f"round-1 overlap_signal_idf has non-zero rows: {(r1_idf != 0).sum()}"
    assert df["overlap_signal_idf"].between(0.0, 1.0).all(), "overlap_signal_idf out of [0,1]"
    print(f"  overlap_signal_idf range: [{df['overlap_signal_idf'].min():.4f}, {df['overlap_signal_idf'].max():.4f}]  "
          f"mean={df['overlap_signal_idf'].mean():.4f}")
    print(f"  top_score_z_robust  range: [{df['top_score_z_robust'].min():+.4f}, {df['top_score_z_robust'].max():+.4f}]  "
          f"mean={df['top_score_z_robust'].mean():+.4f}")

    # Comparison vs incumbents.
    if "overlap_signal" in df.columns:
        raw_mean = float(df["overlap_signal"].mean())
        idf_mean = float(df["overlap_signal_idf"].mean())
        corr = float(df[["overlap_signal", "overlap_signal_idf"]].corr().iloc[0, 1])
        print(f"  raw_jaccard mean={raw_mean:.4f}  vs idf mean={idf_mean:.4f}  corr={corr:+.3f}")
    if "top_score_z_intra" in df.columns:
        corr = float(df[["top_score_z_intra", "top_score_z_robust"]].corr().iloc[0, 1])
        print(f"  z_intra mean={df['top_score_z_intra'].mean():+.4f}  vs z_robust mean={df['top_score_z_robust'].mean():+.4f}  corr={corr:+.3f}")

    out_path = out_path or parquet_path
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/signal_log_lp_enriched.parquet")
    ap.add_argument("--tune", default="results/signal_log_lp_enriched_tune.parquet")
    ap.add_argument("--data-eval", default="data/dev_eval.json")
    ap.add_argument("--data-tune", default="data/dev_tune.json")
    args = ap.parse_args()
    print("=== EVAL ===")
    enrich(args.eval, args.data_eval)
    print("\n=== TUNE ===")
    enrich(args.tune, args.data_tune)


if __name__ == "__main__":
    main()
