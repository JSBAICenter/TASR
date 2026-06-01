"""Offline policy simulator.

Replays each question's per-round trace under a stopping rule and reports
EM, F1, and average rounds used. No LLM calls — pure dataframe walk.

Built-in rules:
- fixed_k_rule(k): always stop at round k. Used to validate the simulator
  matches Day 2 fixed-k baselines.
- oracle_rule:     stop at the first round where current_em == 1.
                   Upper bound an early-stopping rule can hit.

Validation: fixed-k=1/3/5 EM produced by the simulator must equal the
EM reported by running [src/eval_wrapper.py](src/eval_wrapper.py) on the
fixed_k{k}_dev_eval.json baselines to within float-rounding (0.1).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_wrapper import evaluate  # noqa: E402


Rule = Callable[[pd.Series], bool]


def simulate(df: pd.DataFrame, rule_fn: Rule) -> dict:
    df = df.sort_values(["qid", "round"])
    results: dict[str, dict] = {}
    for qid, group in df.groupby("qid", sort=False):
        chosen = None
        for _, row in group.iterrows():
            if rule_fn(row):
                chosen = row
                break
        if chosen is None:
            chosen = group.iloc[-1]
        results[qid] = {
            "answer": chosen["current_answer"],
            "rounds_used": int(chosen["round"]),
            "em": float(chosen["current_em"]),
            "f1": float(chosen["current_f1"]),
        }
    ems = np.array([r["em"] for r in results.values()])
    f1s = np.array([r["f1"] for r in results.values()])
    calls = np.array([r["rounds_used"] for r in results.values()])
    metrics = {
        "em": float(ems.mean() * 100),
        "f1": float(f1s.mean() * 100),
        "avg_calls": float(calls.mean()),
        "f1_per_call": float((f1s.mean() * 100) / calls.mean()),
        "n_questions": len(results),
    }
    return {"results": results, "metrics": metrics}


def fixed_k_rule(k: int) -> Rule:
    return lambda row: row["round"] >= k


def oracle_rule() -> Rule:
    return lambda row: row["current_em"] == 1


def validate_against_baselines(
    metrics_by_k: dict[int, dict],
    baseline_dir: str,
    gold_path: str,
) -> None:
    print("\nValidation vs Day 2 fixed-k baselines:")
    for k, m in metrics_by_k.items():
        path = Path(baseline_dir) / f"fixed_k{k}_dev_eval.json"
        if not path.exists():
            print(f"  k={k}: baseline JSON missing at {path} (skip)")
            continue
        with open(path) as f:
            baseline = json.load(f)
        preds = baseline["predictions"]
        baseline_eval = evaluate(preds, gold_path)
        diff_em = abs(m["em"] - baseline_eval["em"])
        diff_f1 = abs(m["f1"] - baseline_eval["f1"])
        flag = "OK" if max(diff_em, diff_f1) < 0.11 else "MISMATCH"
        print(
            f"  k={k}: simulator EM {m['em']:5.2f} / F1 {m['f1']:5.2f}  "
            f"vs baseline EM {baseline_eval['em']:5.2f} / F1 {baseline_eval['f1']:5.2f}  "
            f"diff_em={diff_em:.2f}  diff_f1={diff_f1:.2f}  [{flag}]"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", default="results/signal_log_enriched.parquet")
    ap.add_argument("--baseline-dir", default="results")
    ap.add_argument("--gold", default="data/dev_eval.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.enriched)
    print("=" * 60)
    print(f"SIMULATOR on {args.enriched}  (n_rows={len(df)})")
    print("=" * 60)

    metrics_by_k: dict[int, dict] = {}
    for k in [1, 2, 3, 4, 5]:
        out = simulate(df, fixed_k_rule(k))
        m = out["metrics"]
        metrics_by_k[k] = m
        print(
            f"Fixed k={k}: EM {m['em']:5.2f}  F1 {m['f1']:5.2f}  "
            f"avg_calls {m['avg_calls']:.2f}  F1/call {m['f1_per_call']:5.2f}"
        )

    out = simulate(df, oracle_rule())
    m = out["metrics"]
    print(
        f"\nOracle (first correct): EM {m['em']:5.2f}  F1 {m['f1']:5.2f}  "
        f"avg_calls {m['avg_calls']:.2f}  F1/call {m['f1_per_call']:5.2f}"
    )

    validate_against_baselines(metrics_by_k, args.baseline_dir, args.gold)


if __name__ == "__main__":
    main()
