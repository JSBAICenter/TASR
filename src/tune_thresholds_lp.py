"""Threshold sweep + lock for the logit-margin variant.

Two rule families tested against dev_tune:

  A. Drop-in margin (2 thresholds):
       stop if calibrated_logit_margin > T_margin OR overlap_signal > T_overlap

  B. 3-way OR (3 thresholds):
       stop if calibrated_conf > T_conf
            OR calibrated_logit_margin > T_margin
            OR overlap_signal > T_overlap

Both sweep against dev_tune (`signal_log_lp_enriched_tune.parquet`), pick the
best F1 within budget=3.0 avg_calls, then evaluate the locked rule on
`signal_log_lp_enriched.parquet`.

Outputs:
  results/threshold_sweep_lp_A_tune.csv
  results/threshold_sweep_lp_B_tune.csv
  results/locked_rule_lp.txt   (winner of A vs B by F1)
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_A(df: pd.DataFrame, t_margin: float, t_overlap: float) -> dict:
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = (df["calibrated_logit_margin"] > t_margin) | (df["overlap_signal"] > t_overlap)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return {
        "em": float(chosen["current_em"].mean() * 100),
        "f1": float(chosen["current_f1"].mean() * 100),
        "avg_calls": float(chosen["round"].mean()),
        "n_questions": int(len(chosen)),
        "n_stopped_early": int(len(stops)),
    }


def simulate_B(df: pd.DataFrame, t_conf: float, t_margin: float, t_overlap: float) -> dict:
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = (
        (df["calibrated_conf"] > t_conf)
        | (df["calibrated_logit_margin"] > t_margin)
        | (df["overlap_signal"] > t_overlap)
    )
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return {
        "em": float(chosen["current_em"].mean() * 100),
        "f1": float(chosen["current_f1"].mean() * 100),
        "avg_calls": float(chosen["round"].mean()),
        "n_questions": int(len(chosen)),
        "n_stopped_early": int(len(stops)),
    }


def sweep_A(df_tune: pd.DataFrame) -> pd.DataFrame:
    t_margin_grid = np.round(np.arange(0.20, 0.85, 0.025), 4)
    t_overlap_grid = np.round(np.arange(0.05, 0.41, 0.025), 4)
    rows = []
    for tm, to in product(t_margin_grid, t_overlap_grid):
        m = simulate_A(df_tune, float(tm), float(to))
        rows.append({"t_margin": float(tm), "t_overlap": float(to), **m})
    return pd.DataFrame(rows)


def sweep_B(df_tune: pd.DataFrame) -> pd.DataFrame:
    # Coarser 3D grid to keep size manageable: ~7 x 7 x 7 = 343 points.
    t_conf_grid = np.round(np.arange(0.30, 0.66, 0.05), 4)
    t_margin_grid = np.round(np.arange(0.30, 0.85, 0.075), 4)
    t_overlap_grid = np.round(np.arange(0.05, 0.41, 0.05), 4)
    rows = []
    for tc, tm, to in product(t_conf_grid, t_margin_grid, t_overlap_grid):
        m = simulate_B(df_tune, float(tc), float(tm), float(to))
        rows.append({"t_conf": float(tc), "t_margin": float(tm), "t_overlap": float(to), **m})
    return pd.DataFrame(rows)


def pick_best(grid: pd.DataFrame, budget: float) -> pd.Series:
    feasible = grid[grid["avg_calls"] <= budget].copy()
    if feasible.empty:
        feasible = grid.sort_values("avg_calls").head(20)
    return feasible.sort_values(["f1", "em", "avg_calls"], ascending=[False, False, True]).iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_lp_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_lp_enriched.parquet")
    ap.add_argument("--out-A", default="results/threshold_sweep_lp_A_tune.csv")
    ap.add_argument("--out-B", default="results/threshold_sweep_lp_B_tune.csv")
    ap.add_argument("--out-rule", default="results/locked_rule_lp.txt")
    ap.add_argument("--budget", type=float, default=3.0)
    args = ap.parse_args()

    df_tune = pd.read_parquet(args.tune)
    df_eval = pd.read_parquet(args.eval)

    print("=" * 72)
    print(f"Family A: drop-in margin   stop if margin > T_m OR overlap > T_o")
    print("=" * 72)
    gA = sweep_A(df_tune)
    Path(args.out_A).parent.mkdir(parents=True, exist_ok=True)
    gA.to_csv(args.out_A, index=False)
    print(f"  Grid: {len(gA)} pairs, wrote {args.out_A}")
    bestA = pick_best(gA, args.budget)
    print(f"  Best (tune): T_margin={bestA['t_margin']:.4f}  T_overlap={bestA['t_overlap']:.4f}  "
          f"EM={bestA['em']:.2f}  F1={bestA['f1']:.2f}  avg_calls={bestA['avg_calls']:.2f}")
    print(f"  Top 5 by F1 within budget:")
    print(gA[gA["avg_calls"] <= args.budget].sort_values("f1", ascending=False).head(5).to_string(index=False))

    print()
    print("=" * 72)
    print(f"Family B: 3-way OR   stop if conf > T_c OR margin > T_m OR overlap > T_o")
    print("=" * 72)
    gB = sweep_B(df_tune)
    gB.to_csv(args.out_B, index=False)
    print(f"  Grid: {len(gB)} triples, wrote {args.out_B}")
    bestB = pick_best(gB, args.budget)
    print(f"  Best (tune): T_conf={bestB['t_conf']:.4f}  T_margin={bestB['t_margin']:.4f}  "
          f"T_overlap={bestB['t_overlap']:.4f}  EM={bestB['em']:.2f}  F1={bestB['f1']:.2f}  avg_calls={bestB['avg_calls']:.2f}")
    print(f"  Top 5 by F1 within budget:")
    print(gB[gB["avg_calls"] <= args.budget].sort_values("f1", ascending=False).head(5).to_string(index=False))

    # Winner.
    if bestA["f1"] >= bestB["f1"]:
        winner = "A"
        rule_str = f"stop if calibrated_logit_margin > {bestA['t_margin']:.4f} OR overlap_signal > {bestA['t_overlap']:.4f}"
        winner_pick = bestA
        eval_m = simulate_A(df_eval, float(bestA["t_margin"]), float(bestA["t_overlap"]))
        locked = {
            "family": "A",
            "rule": rule_str,
            "t_margin": float(bestA["t_margin"]),
            "t_overlap": float(bestA["t_overlap"]),
        }
    else:
        winner = "B"
        rule_str = (
            f"stop if calibrated_conf > {bestB['t_conf']:.4f} "
            f"OR calibrated_logit_margin > {bestB['t_margin']:.4f} "
            f"OR overlap_signal > {bestB['t_overlap']:.4f}"
        )
        winner_pick = bestB
        eval_m = simulate_B(df_eval, float(bestB["t_conf"]), float(bestB["t_margin"]), float(bestB["t_overlap"]))
        locked = {
            "family": "B",
            "rule": rule_str,
            "t_conf": float(bestB["t_conf"]),
            "t_margin": float(bestB["t_margin"]),
            "t_overlap": float(bestB["t_overlap"]),
        }

    locked["tune"] = {
        "em": float(winner_pick["em"]),
        "f1": float(winner_pick["f1"]),
        "avg_calls": float(winner_pick["avg_calls"]),
        "n_stopped_early": int(winner_pick["n_stopped_early"]),
        "n_questions": int(winner_pick["n_questions"]),
    }
    locked["eval"] = eval_m
    locked["budget_max_avg_calls"] = float(args.budget)

    Path(args.out_rule).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_rule).write_text(json.dumps(locked, indent=2))

    print()
    print("=" * 72)
    print(f"WINNER: family {winner}")
    print("=" * 72)
    print(f"  Rule: {rule_str}")
    print(f"  Tune: EM {locked['tune']['em']:6.2f}  F1 {locked['tune']['f1']:6.2f}  avg_calls {locked['tune']['avg_calls']:.2f}")
    print(f"  Eval: EM {eval_m['em']:6.2f}  F1 {eval_m['f1']:6.2f}  avg_calls {eval_m['avg_calls']:.2f}")
    print()
    print("  Reference (dev_eval, Day 2/3):")
    print(f"    closed-book   EM 25.30  F1 33.60  calls 1.00")
    print(f"    fixed-k=1     EM 41.33  F1 49.86  calls 1.00")
    print(f"    fixed-k=3     EM 54.67  F1 65.45  calls 3.00   <-- Pareto target")
    print(f"    fixed-k=5     EM 57.67  F1 69.42  calls 5.00")
    print(f"    2-signal rule           EM 54.33  F1 64.74  calls 2.57")
    print()
    if eval_m["f1"] >= 65.45 and eval_m["avg_calls"] < 3.0:
        print("  PASS: F1 >= fixed-k=3 AND avg_calls < 3.0")
    elif eval_m["f1"] >= 65.45:
        print(f"  PARTIAL: F1 matched but calls not lower ({eval_m['avg_calls']:.2f} vs 3.0)")
    elif eval_m["avg_calls"] < 3.0:
        print(f"  PARTIAL: cheaper but F1 below ({eval_m['f1']:.2f} vs 65.45)")
    else:
        print(f"  FAIL: F1 {eval_m['f1']:.2f} vs 65.45, avg_calls {eval_m['avg_calls']:.2f} vs 3.0")


if __name__ == "__main__":
    main()
