"""Side-by-side comparison: canonical vs confidence-first prompt.

Reads two enriched parquets, prints a comparison table covering:
- confidence distribution
- per-round EM/F1
- oracle bound
- Cohen's d for each signal
- Brier score for calibrated_conf on each split's tune

Also writes a 2x2 figure overlaying the two prompts' EM/F1 curves and
confidence histograms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulator import fixed_k_rule, oracle_rule, simulate  # noqa: E402


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (a.std() + b.std()) / 2 + 1e-9
    return abs(a.mean() - b.mean()) / pooled


def per_round_table(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    def agg(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("round").agg(
            em=("current_em", "mean"),
            f1=("current_f1", "mean"),
        ) * 100
        return g.round(2)
    a = agg(df_a).rename(columns=lambda c: f"{label_a}_{c}")
    b = agg(df_b).rename(columns=lambda c: f"{label_b}_{c}")
    return a.join(b)


def confidence_summary(df: pd.DataFrame) -> dict:
    counts = df["llm_confidence"].value_counts().sort_index().to_dict()
    n = len(df)
    pct_5 = float(counts.get(5, 0)) / n * 100
    # Shannon entropy in nats
    p = df["llm_confidence"].value_counts(normalize=True)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return {"counts": counts, "pct_at_5": pct_5, "entropy_nats": entropy, "n": n}


def signal_d_table(df: pd.DataFrame, signals: list[str], round_mask: dict[str, callable] | None = None) -> pd.DataFrame:
    rows = []
    for s in signals:
        sub = df if not round_mask or s not in round_mask else df[round_mask[s](df)]
        cor = sub[sub["current_em"] == 1][s]
        inc = sub[sub["current_em"] == 0][s]
        rows.append({
            "signal": s,
            "d": round(cohens_d(cor, inc), 3),
            "mean_correct": round(cor.mean(), 4),
            "mean_incorrect": round(inc.mean(), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default="results/signal_log_enriched.parquet")
    ap.add_argument("--canonical-tune", default="results/signal_log_enriched_tune.parquet")
    ap.add_argument("--confirst", default="results/signal_log_confirst_enriched.parquet")
    ap.add_argument("--confirst-tune", default="results/signal_log_confirst_enriched_tune.parquet")
    ap.add_argument("--out-fig", default="figures/prompt_comparison.png")
    args = ap.parse_args()

    df_can = pd.read_parquet(args.canonical)
    df_can_t = pd.read_parquet(args.canonical_tune)
    df_cfr = pd.read_parquet(args.confirst)
    df_cfr_t = pd.read_parquet(args.confirst_tune)

    print("=" * 72)
    print("PROMPT COMPARISON: canonical (Answer-first) vs confidence-first")
    print("=" * 72)

    # 1) Confidence distribution
    print("\n[1] Confidence distribution on dev_eval")
    for label, df in [("canonical", df_can), ("confirst ", df_cfr)]:
        s = confidence_summary(df)
        counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(s["counts"].items()))
        print(f"  {label}  n={s['n']}  pct@5={s['pct_at_5']:5.1f}%  entropy={s['entropy_nats']:.3f} nats  counts={{{counts_str}}}")

    # 2) Per-round EM/F1
    print("\n[2] Per-round EM/F1 on dev_eval")
    tbl = per_round_table(df_can, df_cfr, "can", "cfr")
    print(tbl.to_string())

    # 3) Pareto + oracle via simulator
    print("\n[3] Fixed-k + oracle on dev_eval")
    print(f"{'rule':<20} {'can_EM':>7} {'can_F1':>7} {'can_calls':>10}    {'cfr_EM':>7} {'cfr_F1':>7} {'cfr_calls':>10}")
    for k in [1, 2, 3, 4, 5]:
        a = simulate(df_can, fixed_k_rule(k))["metrics"]
        b = simulate(df_cfr, fixed_k_rule(k))["metrics"]
        print(f"  fixed k={k:<3}        {a['em']:7.2f} {a['f1']:7.2f} {a['avg_calls']:10.2f}    {b['em']:7.2f} {b['f1']:7.2f} {b['avg_calls']:10.2f}")
    a = simulate(df_can, oracle_rule())["metrics"]
    b = simulate(df_cfr, oracle_rule())["metrics"]
    print(f"  oracle              {a['em']:7.2f} {a['f1']:7.2f} {a['avg_calls']:10.2f}    {b['em']:7.2f} {b['f1']:7.2f} {b['avg_calls']:10.2f}")

    # 4) Signal separation (Cohen's d)
    print("\n[4] Signal separation (Cohen's d) on dev_eval — correct vs incorrect rows")
    signals = ["top_score_z_global", "top_score_z_intra", "gap", "overlap_signal", "calibrated_conf"]
    mask = {"overlap_signal": lambda df: df["round"] >= 2}
    a = signal_d_table(df_can, signals, mask).rename(columns={"d": "can_d", "mean_correct": "can_mean_correct", "mean_incorrect": "can_mean_incorrect"})
    b = signal_d_table(df_cfr, signals, mask).rename(columns={"d": "cfr_d", "mean_correct": "cfr_mean_correct", "mean_incorrect": "cfr_mean_incorrect"})
    merged = a.merge(b, on="signal")
    print(merged.to_string(index=False))

    # 5) Brier (already in signals.py output; recompute here for both tunes)
    print("\n[5] Brier of calibrated_conf on tune (lower = better; const-baseline = p*(1-p))")
    for label, df in [("canonical_tune", df_can_t), ("confirst_tune", df_cfr_t)]:
        y = df["current_em"].values.astype(float)
        p = df["calibrated_conf"].values.astype(float)
        brier = float(np.mean((p - y) ** 2))
        const_baseline = float(y.mean() * (1 - y.mean()))
        print(f"  {label:18s}  Brier={brier:.4f}   const_baseline={const_baseline:.4f}   delta={(const_baseline - brier):+.4f}")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, ylabel in zip([axes[0, 0], axes[0, 1]], ["em", "f1"], ["EM (%)", "F1 (%)"]):
        for df, label, color in [(df_can, "canonical", "#1f77b4"), (df_cfr, "confirst", "#d62728")]:
            g = df.groupby("round").agg(em=("current_em", "mean"), f1=("current_f1", "mean")) * 100
            ax.plot(g.index, g[metric], marker="o", label=label, color=color)
        ax.set_xlabel("rounds (= LLM calls)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 6))
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f"dev_eval {metric.upper()} per round")
    for ax, df, label, color in zip([axes[1, 0], axes[1, 1]], [df_can, df_cfr], ["canonical", "confirst"], ["#1f77b4", "#d62728"]):
        cs = df["llm_confidence"].value_counts().sort_index()
        ax.bar(cs.index.astype(str), cs.values, color=color)
        ax.set_xlabel("self-rated confidence")
        ax.set_ylabel("count")
        ax.set_title(f"{label}: confidence distribution (n={len(df)})")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Prompt comparison: canonical (Answer-first) vs confidence-first")
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=140)
    print(f"\nWrote {args.out_fig}")


if __name__ == "__main__":
    main()
