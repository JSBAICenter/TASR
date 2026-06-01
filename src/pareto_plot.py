"""Headline Pareto plot: F1 vs avg retrieval rounds.

F1 vs avg_calls is the main figure. Plot:
- Fixed-k=1..5 as the naive Pareto frontier (line + markers)
- Oracle as an upper-bound point
- Our headline rule (`answer_stable == 1`) as a single point
- 95% bootstrap CI shown as error bars on every point (paired qid resampling)

Also writes a small sensitivity table: F1, F1-per-call, F1 - lambda*calls for
lambda in {0.5, 1.0, 2.0}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def per_question_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = df.eval(rule)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def per_question_fixed_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    sub = df[df["round"] == k][["qid", "current_em", "current_f1", "round"]].copy()
    return sub.sort_values("qid").reset_index(drop=True)


def per_question_oracle(df: pd.DataFrame) -> pd.DataFrame:
    """For each qid, pick the round with the highest current_f1 (break ties by lowest round)."""
    df = df.sort_values(["qid", "round"]).copy()
    idx = df.groupby("qid", sort=False)["current_f1"].idxmax()
    chosen = df.loc[idx, ["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)
    return chosen


def bootstrap_point(pq: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pq)
    f1 = pq["current_f1"].values
    em = pq["current_em"].values
    r = pq["round"].values

    f1_b = np.zeros(n_boot)
    em_b = np.zeros(n_boot)
    r_b = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_b[i] = f1[idx].mean()
        em_b[i] = em[idx].mean()
        r_b[i] = r[idx].mean()

    def ci(arr, scale=1.0):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return float(lo * scale), float(hi * scale)

    return {
        "f1_mean": float(f1.mean() * 100),
        "f1_ci95": ci(f1_b, 100),
        "em_mean": float(em.mean() * 100),
        "em_ci95": ci(em_b, 100),
        "calls_mean": float(r.mean()),
        "calls_ci95": ci(r_b, 1),
        "n": int(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/signal_log_enriched.parquet")
    ap.add_argument("--rule", default="answer_stable == 1")
    ap.add_argument("--rule-name", default="ours (answer_stable)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-fig", default="figures/pareto.png")
    ap.add_argument("--out-json", default="results/pareto_data.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.eval)
    print(f"Loaded {len(df)} rows over {df['qid'].nunique()} questions from {args.eval}")

    points: dict[str, dict] = {}

    for k in [1, 2, 3, 4, 5]:
        pq = per_question_fixed_k(df, k)
        points[f"fixed_k={k}"] = bootstrap_point(pq, args.n_boot, args.seed)

    pq_oracle = per_question_oracle(df)
    points["oracle"] = bootstrap_point(pq_oracle, args.n_boot, args.seed)

    pq_ours = per_question_rule(df, args.rule)
    points["ours"] = bootstrap_point(pq_ours, args.n_boot, args.seed)

    # Sensitivity: F1/call and F1 - lambda * calls
    sensitivity = {}
    for name, pt in points.items():
        sensitivity[name] = {
            "f1": round(pt["f1_mean"], 3),
            "avg_calls": round(pt["calls_mean"], 3),
            "f1_per_call": round(pt["f1_mean"] / pt["calls_mean"], 3),
            **{
                f"f1_minus_{lam}calls": round(pt["f1_mean"] - lam * pt["calls_mean"], 3)
                for lam in [0.5, 1.0, 2.0, 3.0]
            },
        }

    print("\n" + "=" * 80)
    print(f"{'method':<22} {'F1':>7} {'CI95':>17} {'calls':>7} {'CI95':>13}")
    print("-" * 80)
    order = ["fixed_k=1", "fixed_k=2", "fixed_k=3", "fixed_k=4", "fixed_k=5", "ours", "oracle"]
    for name in order:
        p = points[name]
        print(
            f"{name:<22} {p['f1_mean']:7.2f}  [{p['f1_ci95'][0]:5.2f},{p['f1_ci95'][1]:6.2f}]  "
            f"{p['calls_mean']:7.2f}  [{p['calls_ci95'][0]:.2f},{p['calls_ci95'][1]:.2f}]"
        )

    print("\nSensitivity table:")
    print(f"{'method':<22} {'F1':>7} {'calls':>7} {'F1/call':>9} {'F1-0.5c':>9} {'F1-1c':>9} {'F1-2c':>9} {'F1-3c':>9}")
    for name in order:
        s = sensitivity[name]
        print(
            f"{name:<22} {s['f1']:7.2f} {s['avg_calls']:7.2f} {s['f1_per_call']:9.2f} "
            f"{s['f1_minus_0.5calls']:9.2f} {s['f1_minus_1.0calls']:9.2f} "
            f"{s['f1_minus_2.0calls']:9.2f} {s['f1_minus_3.0calls']:9.2f}"
        )

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Fixed-k line
    fk_x = [points[f"fixed_k={k}"]["calls_mean"] for k in [1, 2, 3, 4, 5]]
    fk_y = [points[f"fixed_k={k}"]["f1_mean"] for k in [1, 2, 3, 4, 5]]
    fk_xerr = np.array([[points[f"fixed_k={k}"]["calls_mean"] - points[f"fixed_k={k}"]["calls_ci95"][0],
                          points[f"fixed_k={k}"]["calls_ci95"][1] - points[f"fixed_k={k}"]["calls_mean"]]
                         for k in [1, 2, 3, 4, 5]]).T
    fk_yerr = np.array([[points[f"fixed_k={k}"]["f1_mean"] - points[f"fixed_k={k}"]["f1_ci95"][0],
                          points[f"fixed_k={k}"]["f1_ci95"][1] - points[f"fixed_k={k}"]["f1_mean"]]
                         for k in [1, 2, 3, 4, 5]]).T

    ax.errorbar(fk_x, fk_y, xerr=fk_xerr, yerr=fk_yerr, fmt="o-", color="#7f7f7f",
                 lw=1.5, ms=6, capsize=3, label="fixed-k (naive baseline)", zorder=2)
    for k, x, y in zip([1, 2, 3, 4, 5], fk_x, fk_y):
        ax.annotate(f"k={k}", (x, y), xytext=(6, -10), textcoords="offset points",
                     fontsize=9, color="#555")

    # Oracle point
    op = points["oracle"]
    ax.errorbar([op["calls_mean"]], [op["f1_mean"]],
                 xerr=[[op["calls_mean"] - op["calls_ci95"][0]], [op["calls_ci95"][1] - op["calls_mean"]]],
                 yerr=[[op["f1_mean"] - op["f1_ci95"][0]], [op["f1_ci95"][1] - op["f1_mean"]]],
                 fmt="^", color="#2ca02c", ms=11, capsize=4, label="oracle (upper bound)", zorder=3)
    ax.annotate("oracle", (op["calls_mean"], op["f1_mean"]), xytext=(8, 4),
                 textcoords="offset points", fontsize=9, color="#2ca02c")

    # Ours
    ours = points["ours"]
    ax.errorbar([ours["calls_mean"]], [ours["f1_mean"]],
                 xerr=[[ours["calls_mean"] - ours["calls_ci95"][0]], [ours["calls_ci95"][1] - ours["calls_mean"]]],
                 yerr=[[ours["f1_mean"] - ours["f1_ci95"][0]], [ours["f1_ci95"][1] - ours["f1_mean"]]],
                 fmt="*", color="#d62728", ms=18, capsize=4, label=args.rule_name, zorder=4)

    # Visual: dashed line from ours up to "where fixed-k matches our calls"
    # and right to "where fixed-k matches our F1" -- only if helpful
    ax.axhline(ours["f1_mean"], color="#d62728", lw=0.8, ls=":", alpha=0.4)
    ax.axvline(ours["calls_mean"], color="#d62728", lw=0.8, ls=":", alpha=0.4)

    ax.set_xlabel("Average LLM calls per question", fontsize=11)
    ax.set_ylabel("F1 (×100) on HotpotQA distractor dev_eval", fontsize=11)
    ax.set_title("Budget-aware stopping — Pareto over F1 vs LLM calls", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_xlim(0.6, 5.4)

    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=160)
    print(f"\nWrote {args.out_fig}")

    out = {
        "rule": args.rule,
        "rule_name": args.rule_name,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "points": {k: {kk: vv for kk, vv in v.items()} for k, v in points.items()},
        "sensitivity": sensitivity,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
