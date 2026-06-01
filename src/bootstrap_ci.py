"""Paired bootstrap CIs for the locked stopping rule vs fixed-k baselines.

Resamples qids (with replacement, 1000x) and reports:
- 95% CI on the locked rule's F1, EM, avg_calls
- 95% CI on the paired difference (ours - fixed-k) for k in {2, 3, 5}
- One-sided p-values for "ours worse than fixed-k"

Paired = same resampled qids for both methods, so the CI on the difference
is much tighter than independent CIs (same per-question noise cancels).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def per_question_or_rule(df: pd.DataFrame, t_conf: float, t_overlap: float) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = (df["calibrated_conf"] > t_conf) | (df["overlap_signal"] > t_overlap)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def per_question_fixed_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    sub = df[df["round"] == k][["qid", "current_em", "current_f1", "round"]].copy()
    return sub.sort_values("qid").reset_index(drop=True)


def paired_bootstrap(ours: pd.DataFrame, base: pd.DataFrame, n_boot: int, seed: int) -> dict:
    assert (ours["qid"].values == base["qid"].values).all(), "qid mismatch between methods"
    rng = np.random.default_rng(seed)
    n = len(ours)
    o_em = ours["current_em"].values
    o_f1 = ours["current_f1"].values
    o_r = ours["round"].values
    b_em = base["current_em"].values
    b_f1 = base["current_f1"].values
    b_r = base["round"].values

    diff_f1 = np.zeros(n_boot)
    diff_em = np.zeros(n_boot)
    diff_r = np.zeros(n_boot)
    o_f1_b = np.zeros(n_boot)
    b_f1_b = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diff_f1[i] = (o_f1[idx] - b_f1[idx]).mean()
        diff_em[i] = (o_em[idx] - b_em[idx]).mean()
        diff_r[i] = (o_r[idx] - b_r[idx]).mean()
        o_f1_b[i] = o_f1[idx].mean()
        b_f1_b[i] = b_f1[idx].mean()

    def ci(arr, scale=1.0):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return [round(float(lo * scale), 3), round(float(hi * scale), 3)]

    return {
        "ours_f1_mean": round(float(o_f1.mean() * 100), 3),
        "ours_f1_ci95": ci(o_f1_b, scale=100),
        "base_f1_mean": round(float(b_f1.mean() * 100), 3),
        "base_f1_ci95": ci(b_f1_b, scale=100),
        "diff_f1_mean": round(float((o_f1 - b_f1).mean() * 100), 3),
        "diff_f1_ci95": ci(diff_f1, scale=100),
        "diff_em_mean": round(float((o_em - b_em).mean() * 100), 3),
        "diff_em_ci95": ci(diff_em, scale=100),
        "diff_calls_mean": round(float((o_r - b_r).mean()), 3),
        "diff_calls_ci95": ci(diff_r, scale=1),
        "p_ours_worse_f1": round(float((diff_f1 < 0).mean()), 4),
        "p_ours_better_f1": round(float((diff_f1 > 0).mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/signal_log_enriched.parquet")
    ap.add_argument("--locked-rule", default="results/locked_rule.txt")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/bootstrap_ci.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.eval)
    locked = json.loads(Path(args.locked_rule).read_text())
    t_conf = locked["t_conf"]
    t_overlap = locked["t_overlap"]

    print("=" * 72)
    print(f"Paired bootstrap CI ({args.n_boot} resamples, seed={args.seed})")
    print(f"  locked rule: T_conf={t_conf}  T_overlap={t_overlap}")
    print(f"  eval: {args.eval}  n_questions={df['qid'].nunique()}")
    print("=" * 72)

    ours = per_question_or_rule(df, t_conf, t_overlap)

    out = {
        "rule": locked["rule"],
        "n_questions": int(df["qid"].nunique()),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "ours_em_mean": round(float(ours["current_em"].mean() * 100), 3),
        "ours_f1_mean": round(float(ours["current_f1"].mean() * 100), 3),
        "ours_avg_calls": round(float(ours["round"].mean()), 3),
        "comparisons": {},
    }

    for k in [1, 2, 3, 5]:
        base = per_question_fixed_k(df, k)
        res = paired_bootstrap(ours, base, args.n_boot, args.seed)
        out["comparisons"][f"fixed_k={k}"] = res
        print(f"\nvs fixed-k={k}:")
        print(f"  F1 (ours)         {res['ours_f1_mean']:6.2f}  CI95 {res['ours_f1_ci95']}")
        print(f"  F1 (k={k})          {res['base_f1_mean']:6.2f}  CI95 {res['base_f1_ci95']}")
        print(f"  diff_F1 (ours-k)  {res['diff_f1_mean']:+6.2f}  CI95 {res['diff_f1_ci95']}")
        print(f"  diff_EM (ours-k)  {res['diff_em_mean']:+6.2f}  CI95 {res['diff_em_ci95']}")
        print(f"  diff_calls        {res['diff_calls_mean']:+6.2f}  CI95 {res['diff_calls_ci95']}")
        # Verdict
        lo, hi = res["diff_f1_ci95"]
        if lo > 0:
            verdict = f"SIGNIFICANT WIN on F1 (CI excludes 0, lower bound +{lo:.2f})"
        elif hi < 0:
            verdict = f"SIGNIFICANT LOSS on F1 (CI excludes 0, upper bound {hi:.2f})"
        else:
            verdict = f"NOT SIGNIFICANT on F1 (CI spans 0)"
        print(f"  F1 verdict:       {verdict}")
        print(f"  P(ours worse on F1) = {res['p_ours_worse_f1']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
