"""Round-aware ablation: round-1 trigger + answer_stable for r2+.

For each candidate round-1 signal X:
  1. Sweep threshold T on tune (round-1 rows only).
  2. For each T, build the composite rule
       stop  ==  (round == 1 AND X > T) OR (answer_stable == 1)
     and evaluate on tune. Pick T that maximizes tune F1 within budget.
  3. Evaluate locked rule on eval; paired bootstrap (1000x, seed=42) vs
     answer_stable alone.

answer_stable is the AS=0-at-r1 floor — this experiment buys back the round-1
call when X is so high we trust the round-1 answer.
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


R1_CANDIDATES = [
    "calibrated_logit_margin",
    "calibrated_conf",
    "top_score_z_intra",
    "gap",
    "top_bm25_score",
    "gap_1_2",
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


def round_aware_mask(df: pd.DataFrame, sig: str, t: float) -> np.ndarray:
    """(round==1 AND sig > t) OR (answer_stable == 1)."""
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
    lor, hir = np.percentile(diff_r, [2.5, 97.5])
    return {
        "diff_f1": float((o_f1 - b_f1).mean() * 100),
        "diff_f1_ci95": [float(lo * 100), float(hi * 100)],
        "diff_calls": float((o_r - b_r).mean()),
        "diff_calls_ci95": [float(lor), float(hir)],
    }


def sweep_t_on_tune(df_t: pd.DataFrame, sig: str, budget: float) -> tuple[float, dict]:
    """Sweep T over r1 quantiles of sig on tune; pick best tune F1 within budget."""
    r1_vals = df_t[df_t["round"] == 1][sig].values
    r1_vals = r1_vals[~np.isnan(r1_vals)]
    if len(np.unique(r1_vals)) < 2:
        return float("inf"), {"f1": 0.0, "calls": 5.0, "r1_stops": 0}
    # Use both quantile grid and unique values; capped at 60 candidates.
    cand = np.unique(np.concatenate([
        np.quantile(r1_vals, np.linspace(0, 1, 30)),
        np.unique(r1_vals),
    ]))
    cand = cand[cand >= r1_vals.min()]
    cand = np.sort(np.unique(cand))[-60:]
    best, best_t = None, None
    for t in cand:
        mask = round_aware_mask(df_t, sig, float(t))
        pq = per_question_with_mask(df_t, mask)
        f1 = float(pq["current_f1"].mean() * 100)
        calls = float(pq["round"].mean())
        if calls > budget:
            continue
        # r1 stops = number of qids where r1 triggered and they stopped at r1
        r1_trigger = (df_t["round"].values == 1) & (df_t[sig].values > float(t))
        r1_stops = int(df_t.loc[r1_trigger, "qid"].nunique())
        if best is None or f1 > best["f1"] or (f1 == best["f1"] and calls < best["calls"]):
            best = {"f1": f1, "calls": calls, "r1_stops": r1_stops}
            best_t = float(t)
    return (best_t if best_t is not None else float("inf")), (best or {"f1": 0.0, "calls": 5.0, "r1_stops": 0})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_lp_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_lp_enriched.parquet")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-csv", default="results/round_aware_r1.csv")
    ap.add_argument("--out-json", default="results/round_aware_r1.json")
    args = ap.parse_args()

    df_t = add_answer_stable(pd.read_parquet(args.tune))
    df_e = add_answer_stable(pd.read_parquet(args.eval))
    signals = [s for s in R1_CANDIDATES if s in df_t.columns and s in df_e.columns]
    n_qid_t = df_t["qid"].nunique()
    n_qid_e = df_e["qid"].nunique()

    # Baselines.
    base_as_mask_e = (df_e["answer_stable"].values == 1)
    base_as = per_question_with_mask(df_e, base_as_mask_e)
    base_as_f1 = float(base_as["current_f1"].mean() * 100)
    base_as_calls = float(base_as["round"].mean())
    base_as_em = float(base_as["current_em"].mean() * 100)

    base_k3 = df_e[df_e["round"] == 3][["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)
    base_k3_f1 = float(base_k3["current_f1"].mean() * 100)

    print(f"Tune: {n_qid_t} qids   Eval: {n_qid_e} qids   Budget: ≤{args.budget} calls\n")
    print(f"Eval baselines:")
    print(f"  answer_stable          F1={base_as_f1:6.2f}  EM={base_as_em:.2f}  calls={base_as_calls:4.2f}")
    print(f"  fixed_k=3              F1={base_k3_f1:6.2f}")
    print()

    rows = []
    print(f"{'signal':<26} {'lock T':>10} {'tF1':>6} {'tC':>5} {'tR1':>4} | {'eF1':>6} {'eEM':>6} {'eC':>5} {'eR1':>4} {'ΔvsAS':>7} {'CI vs AS':>16} {'ΔCalls':>7}")
    print("-" * 130)
    for sig in signals:
        t, info_t = sweep_t_on_tune(df_t, sig, args.budget)
        if t == float("inf"):
            print(f"{sig:<26}  no usable threshold (signal binary at r1)")
            continue
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
        print(f"{sig:<26} {t:>10.4f} {info_t['f1']:>6.2f} {info_t['calls']:>5.2f} {info_t['r1_stops']:>4d} | "
              f"{eval_f1:>6.2f} {eval_em:>6.2f} {eval_calls:>5.2f} {r1_stops_e:>4d} "
              f"{bs['diff_f1']:>+7.2f} {ci:>16} {bs['diff_calls']:>+7.2f}  {v}")
        rows.append({
            "signal": sig,
            "rule": f"(r==1 AND {sig} > {t:.4f}) OR answer_stable",
            "threshold": t,
            "tune_f1": info_t["f1"],
            "tune_calls": info_t["calls"],
            "tune_r1_stops": info_t["r1_stops"],
            "eval_f1": eval_f1,
            "eval_em": eval_em,
            "eval_calls": eval_calls,
            "eval_r1_stops": r1_stops_e,
            "diff_f1_vs_as": bs["diff_f1"],
            "ci_as_lo": bs["diff_f1_ci95"][0],
            "ci_as_hi": bs["diff_f1_ci95"][1],
            "diff_calls_vs_as": bs["diff_calls"],
            "verdict_vs_as": v,
        })

    if not rows:
        print("\nNo usable signals.")
        return

    out_df = pd.DataFrame(rows).sort_values("eval_f1", ascending=False).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    out = {
        "baseline_answer_stable_eval": {
            "f1": base_as_f1, "em": base_as_em, "calls": base_as_calls,
        },
        "baseline_fixed_k3_eval_f1": base_k3_f1,
        "n_qid_eval": n_qid_e,
        "budget_max_avg_calls": args.budget,
        "rules": rows,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
