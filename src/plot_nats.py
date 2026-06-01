"""Generate figures for nats.md explainer.

Three figures, all using real project data:
  figures/nats_confidence_collapse.png - 3-panel confidence distributions
  figures/nats_entropy_scale.png       - horizontal entropy bar chart
  figures/nats_margin_reliability.png  - margin histogram + reliability curve
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def shannon_entropy(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main() -> None:
    Path("figures").mkdir(exist_ok=True)

    df_canon = pd.read_parquet("results/signal_log.parquet")
    df_confirst = pd.read_parquet("results/signal_log_confirst.parquet")
    df_smoke = pd.read_parquet("results/smoke_logprobs.parquet")

    counts_canon = df_canon["llm_confidence"].value_counts().sort_index().reindex([1, 2, 3, 4, 5], fill_value=0)
    counts_confirst = df_confirst["llm_confidence"].value_counts().sort_index().reindex([1, 2, 3, 4, 5], fill_value=0)
    p_canon = counts_canon / counts_canon.sum()
    p_confirst = counts_confirst / counts_confirst.sum()
    p_uniform = np.array([0.2] * 5)

    H_canon = shannon_entropy(p_canon.values)
    H_confirst = shannon_entropy(p_confirst.values)
    H_uniform = shannon_entropy(p_uniform)

    # === Figure 1: 3-panel distribution comparison ===
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    panels = [
        (p_canon.values, f"Canonical prompt\n(n={int(counts_canon.sum())})", H_canon),
        (p_confirst.values, f"Confidence-first prompt\n(n={int(counts_confirst.sum())})", H_confirst),
        (p_uniform, "Uniform reference\n(max-entropy on 5 buckets)", H_uniform),
    ]
    for ax, (p_vals, name, H) in zip(axes, panels):
        ax.bar([1, 2, 3, 4, 5], p_vals, color="steelblue", edgecolor="black")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Self-reported confidence (1-5)")
        ax.set_title(f"{name}\nH = {H:.3f} nats")
        # Headroom above the tallest bar so its percent label clears the top spine
        ax.set_ylim(0, 1.12)
        for i, v in enumerate(p_vals):
            ax.text(i + 1, v + 0.02, f"{v * 100:.1f}%", ha="center", fontsize=9)
    axes[0].set_ylabel("Fraction of rows")
    fig.suptitle("Confidence-value distributions: same model, three states")
    fig.tight_layout()
    fig.savefig("figures/nats_confidence_collapse.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote figures/nats_confidence_collapse.png  (H_canon={H_canon:.3f}, H_confirst={H_confirst:.3f}, H_uniform={H_uniform:.3f})")

    # === Figure 2: entropy on the [0, ln(5)] scale ===
    fig, ax = plt.subplots(figsize=(10, 3.8))
    levels = [
        ("Fully collapsed\n(100% on one value)", 0.0, "#8B0000"),
        (f"Canonical (Day 2)\n(96.5% on 5)", H_canon, "#FF8C00"),
        (f"1.0 nat threshold\n(Day 2 decision rule)", 1.0, "#FFD700"),
        (f"Confidence-first\n(62.7% on 5)", H_confirst, "#2E8B57"),
        ("Uniform (max)\n(20% on each)", H_uniform, "#006400"),
    ]
    labels, values, colors = zip(*levels)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Entropy (nats)")
    ax.set_xlim(0, 1.8)
    ax.axvline(math.log(5), ls="--", color="black", lw=1, label=f"ln(5) = {math.log(5):.3f} (theoretical max)")
    ax.axvline(1.0, ls=":", color="red", lw=1)
    ax.set_title("Shannon entropy of the confidence distribution, in nats")
    ax.legend(loc="lower right", fontsize=9)
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/nats_entropy_scale.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote figures/nats_entropy_scale.png")

    # === Figure 3: margin distribution + reliability ===
    df_smoke_clean = df_smoke.dropna(subset=["answer_token_margin"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel A: histogram by EM
    ax = axes[0]
    bins = np.linspace(0, 13, 27)
    em0 = df_smoke_clean[df_smoke_clean["em"] == 0]["answer_token_margin"]
    em1 = df_smoke_clean[df_smoke_clean["em"] == 1]["answer_token_margin"]
    ax.hist(em0, bins=bins, alpha=0.6, label=f"em=0 (wrong)  n={len(em0)}", color="#DC143C", edgecolor="black")
    ax.hist(em1, bins=bins, alpha=0.6, label=f"em=1 (correct)  n={len(em1)}", color="#4682B4", edgecolor="black")
    ax.axvline(em0.mean(), ls="--", color="#DC143C", lw=2)
    ax.axvline(em1.mean(), ls="--", color="#4682B4", lw=2)
    ax.text(em0.mean() + 0.15, ax.get_ylim()[1] * 0.85, f"mean = {em0.mean():.2f}", color="#DC143C", fontsize=9)
    ax.text(em1.mean() + 0.15, ax.get_ylim()[1] * 0.95, f"mean = {em1.mean():.2f}", color="#4682B4", fontsize=9)
    ax.set_xlabel("Logit margin at answer token (nats)")
    ax.set_ylabel("Count")
    ax.set_title("Logit margin distribution by correctness\n(smoke test, n=100 = 20 dev_tune q x 5 rounds)")
    ax.legend(loc="upper right", fontsize=9)

    # Panel B: reliability
    ax = axes[1]
    df_smoke_clean["bucket"] = pd.qcut(df_smoke_clean["answer_token_margin"], q=5, labels=False, duplicates="drop")
    g = df_smoke_clean.groupby("bucket").agg(
        margin_mean=("answer_token_margin", "mean"),
        em_rate=("em", "mean"),
        n=("em", "size"),
    ).reset_index()
    ax.plot(g["margin_mean"], g["em_rate"], color="steelblue", alpha=0.6, lw=2)
    ax.scatter(g["margin_mean"], g["em_rate"], s=g["n"] * 8, color="steelblue", edgecolor="black", zorder=5)
    # Place labels: rightmost bucket goes left of the marker so it doesn't cross the plot edge
    x_max = g["margin_mean"].max()
    for _, row in g.iterrows():
        if row["margin_mean"] == x_max:
            xytext, ha = (-10, 5), "right"
        else:
            xytext, ha = (10, 5), "left"
        ax.annotate(
            f"n={int(row.n)}",
            (row["margin_mean"], row["em_rate"]),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            fontsize=8,
        )
    ax.set_xlabel("Mean logit margin in bucket (nats)")
    ax.set_ylabel("P(em = 1) in bucket")
    ax.set_title("Margin -> correctness reliability\n(5 equal-size buckets)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=min(0, g["margin_mean"].min() - 0.5), right=x_max + 1.0)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/nats_margin_reliability.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote figures/nats_margin_reliability.png")
    print()
    print(f"Smoke test summary:  mean(em=1) = {em1.mean():.2f}  mean(em=0) = {em0.mean():.2f}  "
          f"separation = {em1.mean() - em0.mean():.2f} nats")


if __name__ == "__main__":
    main()
