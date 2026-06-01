"""Signal correctness-separation analysis.

For each candidate signal: plot correct vs incorrect distributions,
report Cohen's d. For overlap_signal we restrict to round >= 2 because
round-1 is always 0 by construction (would falsely depress separation).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SIGNALS = [
    ("top_score_z_global", "Top BM25 z (global)", None),
    ("top_score_z_intra", "Top BM25 z (intra-pool)", None),
    ("gap", "Score gap rank1-rank2", None),
    ("overlap_signal", "Jaccard overlap (rounds >=2)", lambda df: df["round"] >= 2),
    ("calibrated_conf", "Calibrated confidence", None),
]


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (a.std() + b.std()) / 2 + 1e-9
    return abs(a.mean() - b.mean()) / pooled


def analyze(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    flat = axes.flat
    print("\nSignal separation (Cohen's d, larger = more discriminative):")
    print(f"{'signal':30s} {'d':>6s}  {'mean(correct)':>14s}  {'mean(incorrect)':>16s}")
    for ax, (col, label, mask_fn) in zip(flat, SIGNALS):
        sub = df if mask_fn is None else df[mask_fn(df)]
        correct = sub[sub["current_em"] == 1][col]
        incorrect = sub[sub["current_em"] == 0][col]
        ax.hist(incorrect, bins=30, alpha=0.55, density=True, color="#d62728",
                label=f"Incorrect (n={len(incorrect)})")
        ax.hist(correct, bins=30, alpha=0.55, density=True, color="#2ca02c",
                label=f"Correct (n={len(correct)})")
        d = cohens_d(correct, incorrect)
        ax.set_title(f"{label}   d={d:.2f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        print(f"{label:30s} {d:6.3f}  {correct.mean():>14.4f}  {incorrect.mean():>16.4f}")
    for ax in list(flat)[len(SIGNALS):]:
        ax.axis("off")
    fig.suptitle(f"Signal correctness separation  ({Path(out_path).stem})")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", default="results/signal_log_enriched.parquet")
    ap.add_argument("--out", default="figures/signal_separation.png")
    args = ap.parse_args()
    df = pd.read_parquet(args.enriched)
    analyze(df, args.out)


if __name__ == "__main__":
    main()
