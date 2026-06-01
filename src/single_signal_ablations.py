"""Single-signal ablation: for each available signal, find the best 1-threshold
rule on dev_tune, lock it, evaluate on dev_eval, paired-bootstrap vs fixed-k=3.

The point isn't to win — it's to show *why* `answer_stable` is the right choice.
For each signal, we report (tune-best threshold, eval F1, eval calls, CI vs k=3).

Binary signals (answer_stable, answer_in_evidence): rule is "signal == 1".
Continuous signals (calibrated_conf, overlap_signal, top_score_z_global,
top_score_z_intra, gap): sweep threshold on tune, pick best F1 within budget=3.0,
lock and eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def per_question_threshold(df: pd.DataFrame, col: str, thr: float, op: str = ">") -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    if op == ">":
        df["_stop"] = df[col] > thr
    elif op == "==1":
        df["_stop"] = df[col] == 1
    else:
        raise ValueError(op)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def per_question_fixed_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df[df["round"] == k][["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def paired_bootstrap(ours: pd.DataFrame, base: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    assert (ours["qid"].values == base["qid"].values).all()
    rng = np.random.default_rng(seed)
    n = len(ours)
    o_f1 = ours["current_f1"].values
    o_r = ours["round"].values
    b_f1 = base["current_f1"].values
    diff_f1 = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diff_f1[i] = (o_f1[idx] - b_f1[idx]).mean()
    lo, hi = np.percentile(diff_f1, [2.5, 97.5])
    return {
        "ours_f1": float(o_f1.mean() * 100),
        "ours_calls": float(o_r.mean()),
        "base_f1": float(b_f1.mean() * 100),
        "diff_f1": float((o_f1 - b_f1).mean() * 100),
        "diff_f1_ci95": [float(lo * 100), float(hi * 100)],
    }


def sweep_continuous(df_tune: pd.DataFrame, col: str, budget: float) -> tuple[float, dict]:
    lo, hi = float(df_tune[col].min()), float(df_tune[col].max())
    grid = np.linspace(lo, hi, 40)
    best = None
    best_thr = None
    for thr in grid:
        pq = per_question_threshold(df_tune, col, float(thr), op=">")
        f1 = float(pq["current_f1"].mean() * 100)
        calls = float(pq["round"].mean())
        if calls > budget:
            continue
        if best is None or f1 > best["f1"]:
            best = {"f1": f1, "em": float(pq["current_em"].mean() * 100), "calls": calls}
            best_thr = float(thr)
    return best_thr, best or {"f1": 0.0, "em": 0.0, "calls": 5.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_enriched.parquet")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/single_signal_ablations.json")
    args = ap.parse_args()

    df_t = pd.read_parquet(args.tune)
    df_e = pd.read_parquet(args.eval)

    binary_signals = ["answer_stable", "answer_in_evidence"]
    continuous_signals = ["calibrated_conf", "overlap_signal", "top_score_z_global", "top_score_z_intra", "gap"]

    base = per_question_fixed_k(df_e, 3)
    results = {}

    print("=" * 96)
    print(f"{'signal':<22} {'thr':>8} {'tune_F1':>8} {'tune_c':>7}    {'eval_F1':>8} {'eval_c':>7}  {'vs k=3 ΔF1':>12} {'CI95':>17}")
    print("-" * 96)

    for sig in binary_signals:
        pq_t = per_question_threshold(df_t, sig, 0, op="==1")
        pq_e = per_question_threshold(df_e, sig, 0, op="==1")
        boot = paired_bootstrap(pq_e, base, args.n_boot)
        results[sig] = {
            "type": "binary",
            "threshold": "==1",
            "tune_f1": float(pq_t["current_f1"].mean() * 100),
            "tune_calls": float(pq_t["round"].mean()),
            "eval_f1": boot["ours_f1"],
            "eval_calls": boot["ours_calls"],
            "diff_f1_vs_k3": boot["diff_f1"],
            "diff_f1_ci95": boot["diff_f1_ci95"],
        }
        r = results[sig]
        lo, hi = r["diff_f1_ci95"]
        verdict = "WIN" if lo > 0 else "LOSS" if hi < 0 else "tie"
        print(f"{sig:<22} {'==1':>8} {r['tune_f1']:8.2f} {r['tune_calls']:7.2f}    "
              f"{r['eval_f1']:8.2f} {r['eval_calls']:7.2f}  {r['diff_f1_vs_k3']:+12.2f} "
              f"[{lo:+6.2f},{hi:+6.2f}] {verdict}")

    for sig in continuous_signals:
        thr, tune_best = sweep_continuous(df_t, sig, args.budget)
        if thr is None:
            print(f"{sig:<22}  no feasible threshold")
            continue
        pq_e = per_question_threshold(df_e, sig, thr, op=">")
        boot = paired_bootstrap(pq_e, base, args.n_boot)
        results[sig] = {
            "type": "continuous",
            "threshold": thr,
            "tune_f1": tune_best["f1"],
            "tune_calls": tune_best["calls"],
            "eval_f1": boot["ours_f1"],
            "eval_calls": boot["ours_calls"],
            "diff_f1_vs_k3": boot["diff_f1"],
            "diff_f1_ci95": boot["diff_f1_ci95"],
        }
        r = results[sig]
        lo, hi = r["diff_f1_ci95"]
        verdict = "WIN" if lo > 0 else "LOSS" if hi < 0 else "tie"
        print(f"{sig:<22} {thr:8.3f} {r['tune_f1']:8.2f} {r['tune_calls']:7.2f}    "
              f"{r['eval_f1']:8.2f} {r['eval_calls']:7.2f}  {r['diff_f1_vs_k3']:+12.2f} "
              f"[{lo:+6.2f},{hi:+6.2f}] {verdict}")

    print("\nReference: fixed_k=3 eval F1 = 65.45")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
