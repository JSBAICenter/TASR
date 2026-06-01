"""Threshold sweep + lock for the budget-aware stopping rule.

Rule form: stop if `calibrated_conf > T_conf` OR `overlap_signal > T_overlap`.

Day 4 experiments 1.1 + 1.2:
- 1.1: grid sweep (T_conf, T_overlap) on dev_tune. Save grid CSV + heatmap.
- 1.2: pick the pair maximizing F1 subject to avg_rounds <= --budget.
       Write locked rule to results/locked_rule.txt. Evaluate on dev_eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simulate_or_rule(df: pd.DataFrame, t_conf: float, t_overlap: float) -> dict:
    """Vectorized replay of the OR rule.

    For each qid, stop at the first round where calibrated_conf > t_conf
    OR overlap_signal > t_overlap. Fall back to the last row if no stop fires.
    """
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = (df["calibrated_conf"] > t_conf) | (df["overlap_signal"] > t_overlap)

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


def sweep(df_tune: pd.DataFrame, t_conf_grid: np.ndarray, t_overlap_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for tc in t_conf_grid:
        for to in t_overlap_grid:
            m = simulate_or_rule(df_tune, float(tc), float(to))
            rows.append({"t_conf": float(tc), "t_overlap": float(to), **m})
    return pd.DataFrame(rows)


def pick_best(grid: pd.DataFrame, max_avg_calls: float) -> pd.Series:
    feasible = grid[grid["avg_calls"] <= max_avg_calls]
    if feasible.empty:
        print(f"  WARNING: no pair meets avg_calls <= {max_avg_calls}; falling back to lowest avg_calls")
        feasible = grid.sort_values("avg_calls").head(20)
    # Primary: max F1. Secondary: max EM. Tertiary: min avg_calls (ties broken in our favor).
    return feasible.sort_values(["f1", "em", "avg_calls"], ascending=[False, False, True]).iloc[0]


def plot_heatmap(grid: pd.DataFrame, out_path: str, metric: str = "f1") -> None:
    pivot = grid.pivot(index="t_overlap", columns="t_conf", values=metric)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        cmap="viridis",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.upper())
    ax.set_xlabel("T_conf (calibrated_conf threshold)")
    ax.set_ylabel("T_overlap (overlap_signal threshold)")
    ax.set_title(f"Threshold sweep on dev_tune: {metric.upper()} by (T_conf, T_overlap)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_enriched.parquet")
    ap.add_argument("--out-grid", default="results/threshold_sweep_tune.csv")
    ap.add_argument("--out-rule", default="results/locked_rule.txt")
    ap.add_argument("--out-fig", default="figures/threshold_heatmap.png")
    ap.add_argument("--budget", type=float, default=3.0)
    args = ap.parse_args()

    df_tune = pd.read_parquet(args.tune)
    df_eval = pd.read_parquet(args.eval)

    print("=" * 72)
    print(f"Experiment 1.1 - threshold sweep on {args.tune}  (n_rows={len(df_tune)})")
    print("=" * 72)

    # T_conf max set just above the highest calibrated_conf in tune to make sure
    # the grid covers "stop almost nothing".
    t_conf_grid = np.round(np.arange(0.30, 0.70, 0.02), 4)
    t_overlap_grid = np.round(np.arange(0.05, 0.41, 0.025), 4)
    print(f"  Grid: {len(t_conf_grid)} T_conf x {len(t_overlap_grid)} T_overlap = "
          f"{len(t_conf_grid) * len(t_overlap_grid)} pairs")
    print(f"  Budget: avg_calls <= {args.budget}")

    grid = sweep(df_tune, t_conf_grid, t_overlap_grid)
    Path(args.out_grid).parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out_grid, index=False)
    print(f"  Wrote grid to {args.out_grid}  ({len(grid)} rows)")

    plot_heatmap(grid, args.out_fig, metric="f1")

    feasible = grid[grid["avg_calls"] <= args.budget].copy()
    print(f"\n  Feasible pairs (avg_calls <= {args.budget}): {len(feasible)}/{len(grid)}")
    print("\n  Top 8 by F1 within budget:")
    print(feasible.sort_values("f1", ascending=False).head(8).to_string(index=False))

    print("\n  Top 5 by F1/call across the whole grid:")
    g2 = grid.assign(f1_per_call=lambda d: d["f1"] / d["avg_calls"])
    print(g2.sort_values("f1_per_call", ascending=False).head(5).to_string(index=False))

    # ---- Experiment 1.2: lock + apply to eval ----
    print("\n" + "=" * 72)
    print("Experiment 1.2 - lock rule and evaluate on dev_eval")
    print("=" * 72)

    best = pick_best(grid, max_avg_calls=args.budget)
    locked = {
        "t_conf": float(best["t_conf"]),
        "t_overlap": float(best["t_overlap"]),
        "rule": f"stop if calibrated_conf > {best['t_conf']:.4f} OR overlap_signal > {best['t_overlap']:.4f}",
        "tune": {
            "em": float(best["em"]),
            "f1": float(best["f1"]),
            "avg_calls": float(best["avg_calls"]),
            "n_stopped_early": int(best["n_stopped_early"]),
            "n_questions": int(best["n_questions"]),
        },
        "budget_max_avg_calls": float(args.budget),
    }

    eval_m = simulate_or_rule(df_eval, locked["t_conf"], locked["t_overlap"])
    locked["eval"] = eval_m

    Path(args.out_rule).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_rule).write_text(json.dumps(locked, indent=2))
    print(f"  Locked rule -> {args.out_rule}")
    print(f"    {locked['rule']}")
    print(f"    Tune:  EM {locked['tune']['em']:6.2f}  F1 {locked['tune']['f1']:6.2f}  "
          f"avg_calls {locked['tune']['avg_calls']:.2f}  "
          f"stopped early {locked['tune']['n_stopped_early']}/{locked['tune']['n_questions']}")
    print(f"    Eval:  EM {eval_m['em']:6.2f}  F1 {eval_m['f1']:6.2f}  "
          f"avg_calls {eval_m['avg_calls']:.2f}  "
          f"stopped early {eval_m['n_stopped_early']}/{eval_m['n_questions']}")

    print("\n  Reference (dev_eval, Day 2/3):")
    print(f"    closed-book          EM 25.30  F1 33.60  calls 1.00  (no retrieval)")
    print(f"    fixed-k=1            EM 41.33  F1 49.86  calls 1.00")
    print(f"    fixed-k=3            EM 54.67  F1 65.45  calls 3.00   <-- Pareto target")
    print(f"    fixed-k=5            EM 57.67  F1 69.42  calls 5.00")
    print(f"    oracle               EM 60.33  F1 70.92  calls 2.90   (upper bound)")

    # Headline verdict.
    target_f1 = 65.45
    target_calls = 3.0
    print("\n  Verdict vs fixed-k=3:")
    if eval_m["f1"] >= target_f1 and eval_m["avg_calls"] < target_calls:
        print(f"    PASS: F1 {eval_m['f1']:.2f} >= {target_f1} AND avg_calls {eval_m['avg_calls']:.2f} < {target_calls}")
    elif eval_m["f1"] >= target_f1:
        print(f"    PARTIAL: F1 matched but calls not lower ({eval_m['avg_calls']:.2f} vs {target_calls})")
    elif eval_m["avg_calls"] < target_calls:
        print(f"    PARTIAL: cheaper ({eval_m['avg_calls']:.2f} vs {target_calls}) but F1 below target ({eval_m['f1']:.2f} vs {target_f1})")
    else:
        print(f"    FAIL: F1 {eval_m['f1']:.2f} vs {target_f1}, avg_calls {eval_m['avg_calls']:.2f} vs {target_calls}")


if __name__ == "__main__":
    main()
