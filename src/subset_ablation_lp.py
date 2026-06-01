"""Subset ablation on the LP parquet to test logit-margin + others vs answer_stable.

Adds answer_stable to the lp trace (computed in-memory, parquet untouched), then
OR-enumerates 2^6 - 1 = 63 subsets over:
        calibrated_logit_margin,
        answer_stable,
    calibrated_conf, overlap_signal, top_score_z_intra, gap.

Per-signal thresholds tuned on tune within budget. Each subset evaluated on eval
and paired-bootstrapped (1000x, seed=42) against:
    - answer_stable alone (the primary comparator),
    - the locked Family A rule (margin>0.725 OR overlap>0.15),
  - fixed-k=3.

Specifically flags the answer_stable+calibrated_logit_margin subset and supersets.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


BINARY_SIGNALS = {"answer_stable"}
LP_SIGNALS = [
    "calibrated_logit_margin",
    "answer_stable",
    "calibrated_conf",
    "overlap_signal",
    "top_score_z_intra",
    "gap",
]


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


def per_question_fixed_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df[df["round"] == k][["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


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
    lor, hir = np.percentile(diff_r, [2.5, 97.5])
    return {
        "diff_f1": float((o_f1 - b_f1).mean() * 100),
        "diff_f1_ci95": [float(lo * 100), float(hi * 100)],
        "diff_calls": float((o_r - b_r).mean()),
        "diff_calls_ci95": [float(lor), float(hir)],
    }


def sweep_signal_threshold(df_tune: pd.DataFrame, sig: str, budget: float) -> tuple[float, dict]:
    if sig in BINARY_SIGNALS:
        mask = (df_tune[sig].values == 1)
        pq = per_question_with_mask(df_tune, mask)
        return 1.0, {"f1": float(pq["current_f1"].mean() * 100), "calls": float(pq["round"].mean())}
    lo, hi = float(df_tune[sig].min()), float(df_tune[sig].max())
    grid = np.linspace(lo, hi, 40)
    best, best_thr = None, None
    for thr in grid:
        mask = (df_tune[sig].values > thr)
        pq = per_question_with_mask(df_tune, mask)
        f1 = float(pq["current_f1"].mean() * 100)
        calls = float(pq["round"].mean())
        if calls > budget:
            continue
        if best is None or f1 > best["f1"]:
            best = {"f1": f1, "calls": calls}
            best_thr = float(thr)
    return (best_thr if best_thr is not None else float(hi) + 1.0), (best or {"f1": 0.0, "calls": 5.0})


def signal_mask(df: pd.DataFrame, sig: str, thr: float) -> np.ndarray:
    if sig in BINARY_SIGNALS:
        return df[sig].values == 1
    return df[sig].values > thr


def fmt_ci(lo, hi):
    return f"[{lo:+5.2f},{hi:+5.2f}]"


def verdict(lo, hi):
    if lo > 0:
        return "WIN"
    if hi < 0:
        return "LOSS"
    return "tie"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_lp_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_lp_enriched.parquet")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-csv", default="results/subset_ablation_lp.csv")
    ap.add_argument("--out-json", default="results/subset_ablation_lp.json")
    ap.add_argument("--combiner", choices=["or", "and"], default="or",
                    help="OR=stop when any signal fires (default); AND=stop when all fire on the same round")
    args = ap.parse_args()

    df_t = pd.read_parquet(args.tune)
    df_e = pd.read_parquet(args.eval)
    df_t = add_answer_stable(df_t)
    df_e = add_answer_stable(df_e)

    signals = [s for s in LP_SIGNALS if s in df_t.columns and s in df_e.columns]
    print(f"Signals available: {signals}")

    print(f"\nPer-signal tuning on {args.tune} (budget calls ≤ {args.budget}):")
    locked: dict[str, tuple[float, dict]] = {}
    for sig in signals:
        thr, info = sweep_signal_threshold(df_t, sig, args.budget)
        locked[sig] = (thr, info)
        op_str = "==1" if sig in BINARY_SIGNALS else f">{thr:.4f}"
        print(f"  {sig:<26}  rule={op_str:<14}  tune_F1={info['f1']:6.2f}  tune_calls={info['calls']:4.2f}")

    # Baselines on eval.
    base_k3 = per_question_fixed_k(df_e, 3)
    base_k3_f1 = float(base_k3["current_f1"].mean() * 100)
    base_k3_calls = float(base_k3["round"].mean())

    base_as = per_question_with_mask(df_e, signal_mask(df_e, "answer_stable", 1.0))
    base_as_f1 = float(base_as["current_f1"].mean() * 100)
    base_as_calls = float(base_as["round"].mean())
    base_as_triggered = int(signal_mask(df_e, "answer_stable", 1.0).any() and
                            df_e.assign(_m=signal_mask(df_e, "answer_stable", 1.0))
                                .groupby("qid")["_m"].any().sum())

    # Locked Family A: margin>0.725 OR overlap>0.15.
    fam_a_mask_e = (df_e["calibrated_logit_margin"].values > 0.725) | (df_e["overlap_signal"].values > 0.15)
    base_famA = per_question_with_mask(df_e, fam_a_mask_e)
    base_famA_f1 = float(base_famA["current_f1"].mean() * 100)
    base_famA_calls = float(base_famA["round"].mean())

    print(f"\nEval baselines:")
    print(f"  fixed_k=3                          F1={base_k3_f1:6.2f}  calls={base_k3_calls:4.2f}")
    print(f"  answer_stable==1                  F1={base_as_f1:6.2f}  calls={base_as_calls:4.2f}  triggered={base_as_triggered}/{df_e['qid'].nunique()}")
    print(f"  Family A: margin>.725 OR ovl>.15   F1={base_famA_f1:6.2f}  calls={base_famA_calls:4.2f}")

    eval_masks = {sig: signal_mask(df_e, sig, thr) for sig, (thr, _) in locked.items()}
    tune_masks = {sig: signal_mask(df_t, sig, thr) for sig, (thr, _) in locked.items()}

    rows = []
    for r in range(1, len(signals) + 1):
        for combo in itertools.combinations(signals, r):
            if args.combiner == "or":
                e_mask = np.zeros(len(df_e), dtype=bool)
                t_mask = np.zeros(len(df_t), dtype=bool)
                for sig in combo:
                    e_mask |= eval_masks[sig]
                    t_mask |= tune_masks[sig]
            else:  # and
                e_mask = np.ones(len(df_e), dtype=bool)
                t_mask = np.ones(len(df_t), dtype=bool)
                for sig in combo:
                    e_mask &= eval_masks[sig]
                    t_mask &= tune_masks[sig]
            pq_e = per_question_with_mask(df_e, e_mask)
            pq_t = per_question_with_mask(df_t, t_mask)
            triggered = int(pd.DataFrame({"qid": df_e["qid"], "_m": e_mask}).groupby("qid")["_m"].any().sum())
            bs_vs_as = paired_bootstrap(pq_e, base_as, args.n_boot)
            bs_vs_famA = paired_bootstrap(pq_e, base_famA, args.n_boot)
            bs_vs_k3 = paired_bootstrap(pq_e, base_k3, args.n_boot)
            rows.append({
                "size": r,
                "subset": "+".join(combo),
                "has_margin": int("calibrated_logit_margin" in combo),
                "has_answer_stable": int("answer_stable" in combo),
                "tune_f1": float(pq_t["current_f1"].mean() * 100),
                "tune_calls": float(pq_t["round"].mean()),
                "eval_f1": float(pq_e["current_f1"].mean() * 100),
                "eval_em": float(pq_e["current_em"].mean() * 100),
                "eval_calls": float(pq_e["round"].mean()),
                "eval_triggered": triggered,
                "diff_f1_vs_as": bs_vs_as["diff_f1"],
                "ci_as_lo": bs_vs_as["diff_f1_ci95"][0],
                "ci_as_hi": bs_vs_as["diff_f1_ci95"][1],
                "diff_calls_vs_as": bs_vs_as["diff_calls"],
                "diff_f1_vs_famA": bs_vs_famA["diff_f1"],
                "ci_famA_lo": bs_vs_famA["diff_f1_ci95"][0],
                "ci_famA_hi": bs_vs_famA["diff_f1_ci95"][1],
                "diff_f1_vs_k3": bs_vs_k3["diff_f1"],
                "ci_k3_lo": bs_vs_k3["diff_f1_ci95"][0],
                "ci_k3_hi": bs_vs_k3["diff_f1_ci95"][1],
            })

    out_df = pd.DataFrame(rows).sort_values(["eval_f1", "eval_calls"], ascending=[False, True]).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}  ({len(out_df)} subsets)")

    def print_table(rows_df: pd.DataFrame, title: str, max_rows: int = 30) -> None:
        print(f"\n=== {title} ===")
        hdr = f"{'sz':>2} {'subset':<70} {'eF1':>6} {'eC':>5} {'trg':>4} {'ΔvsAS':>7} {'CI vs AS':>16} {'vd':>4} {'ΔvsFA':>7} {'CI vs FA':>16}"
        print(hdr)
        for _, r in rows_df.head(max_rows).iterrows():
            v_as = verdict(r["ci_as_lo"], r["ci_as_hi"])
            ci_as = fmt_ci(r["ci_as_lo"], r["ci_as_hi"])
            ci_fa = fmt_ci(r["ci_famA_lo"], r["ci_famA_hi"])
            print(f"{int(r['size']):>2} {r['subset']:<70} {r['eval_f1']:6.2f} {r['eval_calls']:5.2f} {int(r['eval_triggered']):>4} "
                  f"{r['diff_f1_vs_as']:+7.2f} {ci_as:>16} {v_as:>4} {r['diff_f1_vs_famA']:+7.2f} {ci_fa:>16}")

    print_table(out_df, "TOP 30 BY EVAL F1")

    margin_rows = out_df[out_df["has_margin"] == 1].copy()
    print_table(margin_rows, f"ALL {len(margin_rows)} SUBSETS CONTAINING calibrated_logit_margin (sorted by F1)")

    combined = out_df[(out_df["has_margin"] == 1) & (out_df["has_answer_stable"] == 1)].copy()
    print_table(combined, f"ALL {len(combined)} SUBSETS WITH BOTH answer_stable AND calibrated_logit_margin")

    wins_vs_as = out_df[out_df["ci_as_lo"] > 0]
    print(f"\nSubsets that significantly beat answer_stable (CI vs AS excludes 0):  {len(wins_vs_as)}/{len(out_df)}")
    if len(wins_vs_as):
        print_table(wins_vs_as, "WINNERS vs answer_stable", max_rows=len(wins_vs_as))

    out = {
        "signals": signals,
        "budget_max_avg_calls": args.budget,
        "baselines": {
            "fixed_k3": {"f1": base_k3_f1, "calls": base_k3_calls},
            "answer_stable": {"f1": base_as_f1, "calls": base_as_calls, "triggered": base_as_triggered},
            "family_a": {"f1": base_famA_f1, "calls": base_famA_calls, "rule": "margin>0.725 OR overlap>0.15"},
        },
        "locked_thresholds": {s: {"threshold": thr, "tune_f1": info["f1"], "tune_calls": info["calls"]}
                              for s, (thr, info) in locked.items()},
        "n_subsets": len(out_df),
        "n_significant_wins_vs_answer_stable": int(len(wins_vs_as)),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
