"""Accuracy-vs-calls plot from instrumented trace + closed-book baseline.

X = LLM calls per question (= rounds used). Y = EM or F1 (%).
Each split is one line; closed-book is the point at x=0.
Bar to beat = fixed-k=3 F1 of 65.4% on dev_eval (dashed horizontal line).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def per_round_means(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    grouped = df.groupby("round").agg(
        em=("current_em", "mean"), f1=("current_f1", "mean")
    )
    grouped["em"] = grouped["em"] * 100
    grouped["f1"] = grouped["f1"] * 100
    return grouped.reset_index()


def main() -> None:
    tune = per_round_means("results/signal_log_tune.parquet")
    eval_ = per_round_means("results/signal_log.parquet")

    closed_book = {
        "dev_tune": {"em": 24.0, "f1": 30.4},
        "dev_eval": {"em": 25.3, "f1": 33.6},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, metric, ylabel in zip(axes, ["em", "f1"], ["Exact Match (%)", "F1 (%)"]):
        ax.plot(
            [0] + tune["round"].tolist(),
            [closed_book["dev_tune"][metric]] + tune[metric].tolist(),
            marker="o", label="dev_tune (100q)", color="#1f77b4",
        )
        ax.plot(
            [0] + eval_["round"].tolist(),
            [closed_book["dev_eval"][metric]] + eval_[metric].tolist(),
            marker="s", label="dev_eval (300q)", color="#d62728",
        )
        if metric == "f1":
            ax.axhline(65.4, ls="--", color="grey", lw=1, label="fixed-k=3 bar (F1 65.4)")
        ax.set_xlabel("LLM calls per question (= rounds)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(0, 6))
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Accuracy vs. calls per question (full-budget trace, no stopping rule yet)")
    fig.tight_layout()
    out = Path("figures/pareto_day2.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
