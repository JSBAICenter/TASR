"""Quick A/B test of §2 (overlap_signal_idf) and §3 (top_score_z_robust).

For each of the two new signals, compare directly against its incumbent:
  - overlap_signal_idf  vs  overlap_signal (raw Jaccard)
  - top_score_z_robust  vs  top_score_z_intra (z-norm)

Tests:
  A. Single-signal stopping rule, threshold tuned on tune within budget.
  B. Round-aware r1 rule: (round==1 AND X>T) OR answer_stable, hand-picked T grid.
  C. Drop-in inside the answer_stable + X AND-clause (composition mode that
     produced ties in the DNF ablation).

All comparisons paired-bootstrap vs answer_stable (the headline) on eval.
"""

from __future__ import annotations

import argparse
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


def paired_bootstrap(ours: pd.DataFrame, base: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    assert (ours["qid"].values == base["qid"].values).all()
    rng = np.random.default_rng(seed)
    n = len(ours)
    o_f1, o_r = ours["current_f1"].values, ours["round"].values
    b_f1, b_r = base["current_f1"].values, base["round"].values
    diff_f1 = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diff_f1[i] = (o_f1[idx] - b_f1[idx]).mean()
    lo, hi = np.percentile(diff_f1, [2.5, 97.5])
    return {
        "diff_f1": float((o_f1 - b_f1).mean() * 100),
        "diff_f1_ci95": [float(lo * 100), float(hi * 100)],
        "diff_calls": float((o_r - b_r).mean()),
    }


def fmt_ci(lo, hi):
    return f"[{lo:+5.2f},{hi:+5.2f}]"


def verdict(lo, hi):
    if lo > 0:
        return "WIN"
    if hi < 0:
        return "LOSS"
    return "tie"


def sweep_single_signal(df_t: pd.DataFrame, sig: str, budget: float) -> tuple[float, dict]:
    vals = df_t[sig].values
    vals = vals[~np.isnan(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    cand = np.unique(np.concatenate([
        np.linspace(lo, hi, 40),
        np.quantile(vals, np.linspace(0, 1, 30)),
    ]))
    best, best_t = None, None
    for thr in cand:
        mask = (df_t[sig].values > thr)
        pq = per_question_with_mask(df_t, mask)
        f1 = float(pq["current_f1"].mean() * 100)
        calls = float(pq["round"].mean())
        if calls > budget:
            continue
        if best is None or f1 > best["f1"]:
            best = {"f1": f1, "calls": calls}
            best_t = float(thr)
    return (best_t if best_t is not None else float("inf")), (best or {"f1": 0.0, "calls": 5.0})


def round_aware_mask(df: pd.DataFrame, sig: str, t: float) -> np.ndarray:
    r1 = (df["round"].values == 1) & (df[sig].values > t)
    as_ = (df["answer_stable"].values == 1)
    return r1 | as_


def and_with_as_mask(df: pd.DataFrame, sig: str, t: float) -> np.ndarray:
    return (df["answer_stable"].values == 1) & (df[sig].values > t)


def report(df_e, base_as, label, mask, n_boot=1000):
    pq = per_question_with_mask(df_e, mask)
    bs = paired_bootstrap(pq, base_as, n_boot)
    f1 = float(pq["current_f1"].mean() * 100)
    em = float(pq["current_em"].mean() * 100)
    calls = float(pq["round"].mean())
    v = verdict(bs["diff_f1_ci95"][0], bs["diff_f1_ci95"][1])
    ci = fmt_ci(bs["diff_f1_ci95"][0], bs["diff_f1_ci95"][1])
    print(f"  {label:<55} F1={f1:6.2f} EM={em:6.2f} C={calls:5.2f}  ΔvsAS={bs['diff_f1']:+6.2f} {ci} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_lp_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_lp_enriched.parquet")
    ap.add_argument("--budget", type=float, default=3.0)
    args = ap.parse_args()

    df_t = add_answer_stable(pd.read_parquet(args.tune))
    df_e = add_answer_stable(pd.read_parquet(args.eval))

    base_as_mask = (df_e["answer_stable"].values == 1)
    base_as = per_question_with_mask(df_e, base_as_mask)
    base_as_f1 = float(base_as["current_f1"].mean() * 100)
    base_as_calls = float(base_as["round"].mean())
    print(f"Baseline answer_stable on eval: F1={base_as_f1:.2f}  calls={base_as_calls:.2f}\n")

    pairs = [
        ("overlap_signal", "overlap_signal_idf", "§2 IDF-Jaccard"),
        ("top_score_z_intra", "top_score_z_robust", "§3 robust median/IQR"),
    ]

    for incumbent, new, label in pairs:
        print(f"\n{'='*88}\n{label}: {new}  vs  incumbent {incumbent}\n{'='*88}")

        # ---- A. Single-signal stop rule, tune-optimal threshold within budget ----
        for sig in (incumbent, new):
            t, info = sweep_single_signal(df_t, sig, args.budget)
            if t == float("inf"):
                print(f"  [single-signal {sig}] no usable threshold")
                continue
            mask = (df_e[sig].values > t)
            print(f"\n[A] single-signal: stop if {sig} > {t:.4f}   (tune F1={info['f1']:.2f} calls={info['calls']:.2f})")
            report(df_e, base_as, f"     eval", mask)

        # ---- B. Round-aware r1 rule with hand-picked threshold grid ----
        # Use quantile-based hand picks to avoid tune-overfit (same protocol as round_aware_r1_grid.py).
        vals_t = df_t.loc[df_t["round"] == 1, new].values
        vals_t = vals_t[~np.isnan(vals_t)]
        if len(vals_t) > 1:
            quantiles = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]
            grid_new = sorted(set(float(np.quantile(vals_t, q)) for q in quantiles))
            print(f"\n[B] Round-aware (r==1 AND {new} > T) OR answer_stable   T-grid from tune r1 quantiles")
            for t in grid_new:
                m = round_aware_mask(df_e, new, t)
                report(df_e, base_as, f"     T={t:.4f}", m)

        vals_t_old = df_t.loc[df_t["round"] == 1, incumbent].values
        vals_t_old = vals_t_old[~np.isnan(vals_t_old)]
        if len(vals_t_old) > 1:
            grid_old = sorted(set(float(np.quantile(vals_t_old, q)) for q in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]))
            print(f"\n[B'] Round-aware with INCUMBENT {incumbent} (sanity baseline)")
            for t in grid_old:
                m = round_aware_mask(df_e, incumbent, t)
                report(df_e, base_as, f"     T={t:.4f}", m)

        # ---- C. answer_stable AND new>T (composition that produced DNF ties) ----
        # Threshold = median of r2+ values (the rounds where AS can fire).
        rmask = df_t["round"].values >= 2
        vals_t2 = df_t.loc[rmask, new].values
        vals_t2 = vals_t2[~np.isnan(vals_t2)]
        if len(vals_t2) > 1:
            grid = sorted(set(float(np.quantile(vals_t2, q)) for q in [0.25, 0.50, 0.75]))
            print(f"\n[C] AND-clause: answer_stable AND {new} > T")
            for t in grid:
                m = and_with_as_mask(df_e, new, t)
                report(df_e, base_as, f"     T={t:.4f}", m)


if __name__ == "__main__":
    main()
