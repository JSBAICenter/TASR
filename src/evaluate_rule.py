"""Lock + evaluate any rule string. Wraps the threshold sweep + bootstrap.

Usage:
  python src/evaluate_rule.py --rule "answer_stable == 1" \
      --name stable_only --eval results/signal_log_enriched.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def per_question_from_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
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


def paired_bootstrap(ours: pd.DataFrame, base: pd.DataFrame, n_boot: int, seed: int) -> dict:
    assert (ours["qid"].values == base["qid"].values).all()
    rng = np.random.default_rng(seed)
    n = len(ours)
    o_em, o_f1, o_r = ours["current_em"].values, ours["current_f1"].values, ours["round"].values
    b_em, b_f1, b_r = base["current_em"].values, base["current_f1"].values, base["round"].values

    diff_f1 = np.zeros(n_boot)
    diff_em = np.zeros(n_boot)
    diff_r = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diff_f1[i] = (o_f1[idx] - b_f1[idx]).mean()
        diff_em[i] = (o_em[idx] - b_em[idx]).mean()
        diff_r[i] = (o_r[idx] - b_r[idx]).mean()

    def ci(arr, scale=1.0):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return [round(float(lo * scale), 3), round(float(hi * scale), 3)]

    return {
        "ours_f1": round(float(o_f1.mean() * 100), 3),
        "base_f1": round(float(b_f1.mean() * 100), 3),
        "diff_f1": round(float((o_f1 - b_f1).mean() * 100), 3),
        "diff_f1_ci95": ci(diff_f1, scale=100),
        "diff_em": round(float((o_em - b_em).mean() * 100), 3),
        "diff_em_ci95": ci(diff_em, scale=100),
        "diff_calls": round(float((o_r - b_r).mean()), 3),
        "diff_calls_ci95": ci(diff_r, scale=1),
        "p_ours_worse_f1": round(float((diff_f1 < 0).mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True, help="pandas eval() rule string")
    ap.add_argument("--name", required=True, help="short label for outputs")
    ap.add_argument("--tune", default="results/signal_log_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_enriched.parquet")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df_t = pd.read_parquet(args.tune)
    df_e = pd.read_parquet(args.eval)

    ours_t = per_question_from_rule(df_t, args.rule)
    ours_e = per_question_from_rule(df_e, args.rule)

    summary = {
        "name": args.name,
        "rule": args.rule,
        "tune": {
            "f1": round(float(ours_t["current_f1"].mean() * 100), 3),
            "em": round(float(ours_t["current_em"].mean() * 100), 3),
            "avg_calls": round(float(ours_t["round"].mean()), 3),
            "n": int(len(ours_t)),
        },
        "eval": {
            "f1": round(float(ours_e["current_f1"].mean() * 100), 3),
            "em": round(float(ours_e["current_em"].mean() * 100), 3),
            "avg_calls": round(float(ours_e["round"].mean()), 3),
            "n": int(len(ours_e)),
        },
        "bootstrap_vs_fixed_k": {},
    }

    print(f"\n{'=' * 72}\nRule: {args.name}\n  {args.rule}\n{'=' * 72}")
    print(f"  Tune:  F1 {summary['tune']['f1']:6.2f}  EM {summary['tune']['em']:6.2f}  calls {summary['tune']['avg_calls']:.2f}")
    print(f"  Eval:  F1 {summary['eval']['f1']:6.2f}  EM {summary['eval']['em']:6.2f}  calls {summary['eval']['avg_calls']:.2f}")
    print(f"  tune-eval F1 gap: {summary['tune']['f1'] - summary['eval']['f1']:+.2f}")

    print(f"\n  Paired bootstrap ({args.n_boot} resamples) vs fixed-k baselines on dev_eval:")
    for k in [2, 3, 5]:
        base = per_question_fixed_k(df_e, k)
        res = paired_bootstrap(ours_e, base, args.n_boot, args.seed)
        summary["bootstrap_vs_fixed_k"][f"k={k}"] = res
        lo, hi = res["diff_f1_ci95"]
        verdict = (
            "SIG WIN" if lo > 0 else "SIG LOSS" if hi < 0 else "tie (CI spans 0)"
        )
        print(
            f"    k={k}: diff_F1 {res['diff_f1']:+6.2f} CI{res['diff_f1_ci95']}  "
            f"diff_calls {res['diff_calls']:+5.2f} CI{res['diff_calls_ci95']}  [{verdict}]"
        )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"rule_{args.name}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
