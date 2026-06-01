"""Hand-picked threshold sweep for the round-aware r1 rule.

For each candidate r1 signal, scan a grid of thresholds and report eval metrics
for the composite rule:
    stop  ==  (round == 1 AND X > T) OR (answer_stable == 1)

Unlike round_aware_r1.py, this script does NOT tune T on tune. It reports the
full T-vs-eval-F1 curve so we can read off the actual precision-coverage
behaviour and avoid tune-cherry-picking.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def add_answer_stable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy().reset_index(drop=True)
    norm = df["current_answer"].apply(normalize)
    same_qid = (df["qid"] == df["qid"].shift(1)).values
    prev_norm = norm.shift(1).fillna("").values
    df["answer_stable"] = ((norm.values == prev_norm) & same_qid).astype(int)
    return df


def per_question_with_mask(df: pd.DataFrame, stop_mask: np.ndarray) -> pd.DataFrame:
    d = df.copy()
    d["_stop"] = stop_mask
    d = d.sort_values(["qid", "round"])
    stops = d[d["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = d[~d["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def round_aware_mask(df: pd.DataFrame, sig: str, t: float) -> np.ndarray:
    r1_trigger = (df["round"].values == 1) & (df[sig].values > t)
    as_trigger = (df["answer_stable"].values == 1)
    return r1_trigger | as_trigger


def paired_bootstrap(ours: pd.DataFrame, base: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    assert (ours["qid"].values == base["qid"].values).all()
    rng = np.random.default_rng(seed)
    n = len(ours)
    o_f1, o_r = ours["current_f1"].values, ours["round"].values
    b_f1, b_r = base["current_f1"].values, base["round"].values
    diff_f1 = np.zeros(n_boot)
    diff_r = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diff_f1[i] = (o_f1[idx] - b_f1[idx]).mean()
        diff_r[i] = (o_r[idx] - b_r[idx]).mean()
    lo, hi = np.percentile(diff_f1, [2.5, 97.5])
    return {
        "diff_f1": float((o_f1 - b_f1).mean() * 100),
        "diff_f1_ci95": [float(lo * 100), float(hi * 100)],
        "diff_calls": float((o_r - b_r).mean()),
    }


def r1_precision_on_tune(df_t: pd.DataFrame, sig: str, t: float) -> dict:
    r1 = df_t[df_t["round"] == 1]
    mask = r1[sig].values > t
    n = int(mask.sum())
    if n == 0:
        return {"n_stops_tune": 0, "tune_prec_em": float("nan"), "tune_prec_f1": float("nan")}
    return {
        "n_stops_tune": n,
        "tune_prec_em": float(r1["current_em"].values[mask].mean() * 100),
        "tune_prec_f1": float(r1["current_f1"].values[mask].mean() * 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_lp_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_lp_enriched.parquet")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-csv", default="results/round_aware_r1_grid.csv")
    args = ap.parse_args()

    df_t = add_answer_stable(pd.read_parquet(args.tune))
    df_e = add_answer_stable(pd.read_parquet(args.eval))
    n_qid_e = df_e["qid"].nunique()

    base_as_mask_e = (df_e["answer_stable"].values == 1)
    base_as = per_question_with_mask(df_e, base_as_mask_e)
    base_as_f1 = float(base_as["current_f1"].mean() * 100)
    base_as_calls = float(base_as["round"].mean())
    base_as_em = float(base_as["current_em"].mean() * 100)

    print(f"Eval qids: {n_qid_e}")
    print(f"Baseline answer_stable: F1={base_as_f1:.2f}  EM={base_as_em:.2f}  calls={base_as_calls:.2f}\n")

    # Hand-picked thresholds from the r1 precision-coverage analysis on tune.
    grids = {
        # margin: covers the operating points called out earlier (T=0.6471/0.7333) plus a finer grid.
        "calibrated_logit_margin": [0.3846, 0.4500, 0.5000, 0.5500, 0.6000, 0.6471, 0.7000, 0.7333, 0.8000],
        # conf at r1 is binary (only values are 0.0 and 0.4421). Test both edges.
        "calibrated_conf": [-0.01, 0.4400],
        # retrieval-only signals — try a few quantile-derived thresholds for completeness.
        "top_score_z_intra": [2.00, 2.50, 2.80, 3.00],
        "gap": [2.0, 4.0, 6.0, 10.0],
        "top_bm25_score": [10.0, 15.0, 18.0, 25.0],
        "gap_1_2": [2.0, 4.0, 6.0, 10.0],
    }

    rows = []
    for sig, ts in grids.items():
        if sig not in df_e.columns:
            continue
        print(f"=== {sig} ===")
        print(f"{'T':>10} {'tune_n':>7} {'tune_prec_EM':>13} {'tune_prec_F1':>13} | {'eval_F1':>8} {'eval_EM':>8} {'eval_calls':>11} {'r1_stops':>9} {'ΔvsAS':>7} {'CI vs AS':>16} {'v':>4}")
        for t in ts:
            tinfo = r1_precision_on_tune(df_t, sig, t)
            e_mask = round_aware_mask(df_e, sig, t)
            pq_e = per_question_with_mask(df_e, e_mask)
            r1_trigger_e = (df_e["round"].values == 1) & (df_e[sig].values > t)
            r1_stops_e = int(df_e.loc[r1_trigger_e, "qid"].nunique())
            bs = paired_bootstrap(pq_e, base_as, args.n_boot)
            eval_f1 = float(pq_e["current_f1"].mean() * 100)
            eval_em = float(pq_e["current_em"].mean() * 100)
            eval_calls = float(pq_e["round"].mean())
            v = "WIN" if bs["diff_f1_ci95"][0] > 0 else ("LOSS" if bs["diff_f1_ci95"][1] < 0 else "tie")
            ci = f"[{bs['diff_f1_ci95'][0]:+5.2f},{bs['diff_f1_ci95'][1]:+5.2f}]"
            tprec_em = f"{tinfo['tune_prec_em']:>12.2f}%" if not np.isnan(tinfo['tune_prec_em']) else "        n/a"
            tprec_f1 = f"{tinfo['tune_prec_f1']:>12.2f}%" if not np.isnan(tinfo['tune_prec_f1']) else "        n/a"
            print(f"{t:>10.4f} {tinfo['n_stops_tune']:>7d} {tprec_em} {tprec_f1} | "
                  f"{eval_f1:>8.2f} {eval_em:>8.2f} {eval_calls:>11.2f} {r1_stops_e:>9d} "
                  f"{bs['diff_f1']:>+7.2f} {ci:>16} {v:>4}")
            rows.append({
                "signal": sig,
                "threshold": t,
                "tune_r1_stops": tinfo["n_stops_tune"],
                "tune_prec_em": tinfo["tune_prec_em"],
                "tune_prec_f1": tinfo["tune_prec_f1"],
                "eval_f1": eval_f1,
                "eval_em": eval_em,
                "eval_calls": eval_calls,
                "eval_r1_stops": r1_stops_e,
                "diff_f1_vs_as": bs["diff_f1"],
                "ci_as_lo": bs["diff_f1_ci95"][0],
                "ci_as_hi": bs["diff_f1_ci95"][1],
                "diff_calls_vs_as": bs["diff_calls"],
                "verdict": v,
            })
        print()

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
