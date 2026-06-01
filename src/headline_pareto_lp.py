"""Headline Pareto figure (LP parquet) for the paper.

Produces:
  - figures/pareto.pdf           (vector, drop-in for latex)
  - figures/pareto_headline.png  (raster, for slides)
  - results/pareto_data_lp.json  (point estimates + 95% bootstrap CIs)

The LP parquet is the logprob-enriched trace. We compute `answer_stable`
inline from the cached `current_answer` column (no fresh LLM calls), then
sweep fixed-k=1..5, oracle stop, and the locked rule. Paired bootstrap with
1000 resamples, seed=42.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def add_answer_stable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_norm"] = df["current_answer"].map(normalize)
    df["_prev"] = df.groupby("qid")["_norm"].shift(1)
    df["answer_stable"] = ((df["_prev"].notna()) & (df["_prev"] == df["_norm"])).astype(int)
    return df


def per_question_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = df.eval(rule)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def per_question_fixed_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    sub = df[df["round"] == k][["qid", "current_em", "current_f1", "round"]].copy()
    return sub.sort_values("qid").reset_index(drop=True)


def per_question_oracle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    idx = df.groupby("qid", sort=False)["current_f1"].idxmax()
    return df.loc[idx, ["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def bootstrap_point(pq: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pq)
    f1 = pq["current_f1"].values
    em = pq["current_em"].values
    r = pq["round"].values

    f1_b = np.zeros(n_boot); em_b = np.zeros(n_boot); r_b = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_b[i] = f1[idx].mean(); em_b[i] = em[idx].mean(); r_b[i] = r[idx].mean()

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
    ap.add_argument("--eval", default="results/signal_log_lp_enriched.parquet")
    ap.add_argument("--rule", default="answer_stable == 1")
    ap.add_argument("--rule-name", default="answer-stable (ours)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-pdf", default="figures/pareto.pdf")
    ap.add_argument("--out-png", default="figures/pareto_headline.png")
    ap.add_argument("--out-json", default="results/pareto_data_lp.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.eval)
    df = add_answer_stable(df)
    print(f"Loaded {len(df)} rows over {df['qid'].nunique()} qids from {args.eval}")

    points: dict[str, dict] = {}
    for k in [1, 2, 3, 4, 5]:
        points[f"fixed_k={k}"] = bootstrap_point(per_question_fixed_k(df, k), args.n_boot, args.seed)
    points["oracle"] = bootstrap_point(per_question_oracle(df), args.n_boot, args.seed)
    points["ours"] = bootstrap_point(per_question_rule(df, args.rule), args.n_boot, args.seed)

    print("\n" + "=" * 84)
    print(f"{'method':<22} {'F1':>7} {'F1 CI95':>18} {'calls':>7} {'calls CI95':>14}")
    print("-" * 84)
    for name in ["fixed_k=1", "fixed_k=2", "fixed_k=3", "fixed_k=4", "fixed_k=5", "ours", "oracle"]:
        p = points[name]
        print(f"{name:<22} {p['f1_mean']:7.2f} [{p['f1_ci95'][0]:6.2f},{p['f1_ci95'][1]:6.2f}]  "
              f"{p['calls_mean']:7.2f}  [{p['calls_ci95'][0]:.2f},{p['calls_ci95'][1]:.2f}]")

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    fk_x = [points[f"fixed_k={k}"]["calls_mean"] for k in [1, 2, 3, 4, 5]]
    fk_y = [points[f"fixed_k={k}"]["f1_mean"] for k in [1, 2, 3, 4, 5]]
    fk_yerr = np.array([[points[f"fixed_k={k}"]["f1_mean"] - points[f"fixed_k={k}"]["f1_ci95"][0],
                          points[f"fixed_k={k}"]["f1_ci95"][1] - points[f"fixed_k={k}"]["f1_mean"]]
                         for k in [1, 2, 3, 4, 5]]).T

    ax.errorbar(fk_x, fk_y, yerr=fk_yerr, fmt="o-", color="#7f7f7f",
                 lw=1.4, ms=5.5, capsize=3, label="fixed-$k$ (baseline)", zorder=2)
    for k, x, y in zip([1, 2, 3, 4, 5], fk_x, fk_y):
        ax.annotate(f"$k{{=}}{k}$", (x, y), xytext=(6, -10), textcoords="offset points",
                     fontsize=8, color="#555")

    op = points["oracle"]
    ax.errorbar([op["calls_mean"]], [op["f1_mean"]],
                 xerr=[[op["calls_mean"] - op["calls_ci95"][0]], [op["calls_ci95"][1] - op["calls_mean"]]],
                 yerr=[[op["f1_mean"] - op["f1_ci95"][0]], [op["f1_ci95"][1] - op["f1_mean"]]],
                 fmt="^", color="#2ca02c", ms=10, capsize=3, label="oracle (upper bound)", zorder=3)

    ours = points["ours"]
    ax.errorbar([ours["calls_mean"]], [ours["f1_mean"]],
                 xerr=[[ours["calls_mean"] - ours["calls_ci95"][0]], [ours["calls_ci95"][1] - ours["calls_mean"]]],
                 yerr=[[ours["f1_mean"] - ours["f1_ci95"][0]], [ours["f1_ci95"][1] - ours["f1_mean"]]],
                 fmt="*", color="#d62728", ms=16, capsize=3, label=args.rule_name, zorder=4)

    ax.axhline(ours["f1_mean"], color="#d62728", lw=0.6, ls=":", alpha=0.4)
    ax.axvline(ours["calls_mean"], color="#d62728", lw=0.6, ls=":", alpha=0.4)

    ax.set_xlabel("Average LLM calls per question", fontsize=10)
    ax.set_ylabel("F1 on HotpotQA distractor (300q)", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_xlim(0.6, 5.4)

    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_pdf)
    fig.savefig(args.out_png, dpi=180)
    print(f"\nWrote {args.out_pdf}")
    print(f"Wrote {args.out_png}")

    out = {
        "rule": args.rule, "rule_name": args.rule_name, "eval_parquet": args.eval,
        "n_boot": args.n_boot, "seed": args.seed,
        "points": points,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
