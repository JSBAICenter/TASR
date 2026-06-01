"""Full DNF ablation: every AND/OR mixed combination of the 6 signals.

For each non-empty subset S of LP_SIGNALS (63 of them) and each set partition P
of S (Bell number B(|S|) of them), build a DNF stopping rule:
    stop if  OR over groups in P  of  ( AND over members of group )
This collapses to pure OR when every group is a singleton, pure AND when there's
one big group, and a mix in between (e.g. (stable & gap) | top_z).

Total formulas: sum_{k=1..6} C(6,k) * B(k) = 6 + 30 + 100 + 225 + 312 + 203 = 876.

Per-signal thresholds tuned on tune within budget. Each formula evaluated on
eval and paired-bootstrapped (1000x, seed=42) against:
    - answer_stable alone (primary comparator),
    - the locked Family A rule (margin>0.725 OR overlap>0.15),
  - fixed-k=3.

Separately reports winners that contain answer_stable and winners that don't.
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

# Short aliases for compact formula strings.
SHORT = {
    "calibrated_logit_margin": "mrg",
    "answer_stable": "as",
    "calibrated_conf": "cnf",
    "overlap_signal": "ovl",
    "top_score_z_intra": "tzi",
    "gap": "gap",
}


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
    return {
        "diff_f1": float((o_f1 - b_f1).mean() * 100),
        "diff_f1_ci95": [float(lo * 100), float(hi * 100)],
        "diff_calls": float((o_r - b_r).mean()),
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


def set_partitions(items: list) -> list[list[list]]:
    """All set partitions of items (Bell number)."""
    if len(items) == 0:
        return [[]]
    if len(items) == 1:
        return [[list(items)]]
    first = items[0]
    rest = items[1:]
    out = []
    for p in set_partitions(rest):
        out.append([[first]] + [list(g) for g in p])
        for i in range(len(p)):
            new_p = [list(g) for g in p]
            new_p[i] = [first] + new_p[i]
            out.append(new_p)
    return out


def canonical_partition(p: list[list[str]]) -> tuple:
    """Canonical form for dedup: each group sorted, groups sorted by first elem."""
    groups = tuple(sorted(tuple(sorted(g)) for g in p))
    return groups


def formula_str(p: list[list[str]]) -> str:
    """Pretty-print: groups joined by '|', members within a group by '&'."""
    parts = []
    for g in sorted(p, key=lambda gg: tuple(sorted(SHORT[s] for s in gg))):
        names = sorted(SHORT[s] for s in g)
        if len(names) == 1:
            parts.append(names[0])
        else:
            parts.append("&".join(names))
    return "|".join(parts)


def dnf_mask(df_masks: dict[str, np.ndarray], partition: list[list[str]], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for group in partition:
        g_mask = np.ones(n, dtype=bool)
        for sig in group:
            g_mask &= df_masks[sig]
        out |= g_mask
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_lp_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_lp_enriched.parquet")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-csv", default="results/subset_ablation_lp_dnf.csv")
    ap.add_argument("--out-json", default="results/subset_ablation_lp_dnf.json")
    args = ap.parse_args()

    df_t = pd.read_parquet(args.tune)
    df_e = pd.read_parquet(args.eval)
    df_t = add_answer_stable(df_t)
    df_e = add_answer_stable(df_e)

    signals = [s for s in LP_SIGNALS if s in df_t.columns and s in df_e.columns]
    print(f"Signals: {signals}")

    print(f"\nPer-signal tuning on {args.tune} (budget calls ≤ {args.budget}):")
    locked = {}
    for sig in signals:
        thr, info = sweep_signal_threshold(df_t, sig, args.budget)
        locked[sig] = (thr, info)
        op_str = "==1" if sig in BINARY_SIGNALS else f">{thr:.4f}"
        print(f"  {sig:<26}  rule={op_str:<14}  tune_F1={info['f1']:6.2f}  tune_calls={info['calls']:4.2f}")

    eval_masks = {sig: signal_mask(df_e, sig, thr) for sig, (thr, _) in locked.items()}
    tune_masks = {sig: signal_mask(df_t, sig, thr) for sig, (thr, _) in locked.items()}

    base_k3 = per_question_fixed_k(df_e, 3)
    base_k3_f1 = float(base_k3["current_f1"].mean() * 100)
    base_as = per_question_with_mask(df_e, signal_mask(df_e, "answer_stable", 1.0))
    base_as_f1 = float(base_as["current_f1"].mean() * 100)
    base_as_calls = float(base_as["round"].mean())
    fam_a_mask_e = (df_e["calibrated_logit_margin"].values > 0.725) | (df_e["overlap_signal"].values > 0.15)
    base_famA = per_question_with_mask(df_e, fam_a_mask_e)
    base_famA_f1 = float(base_famA["current_f1"].mean() * 100)

    print(f"\nEval baselines:")
    print(f"  fixed_k=3              F1={base_k3_f1:6.2f}")
    print(f"  answer_stable          F1={base_as_f1:6.2f}  calls={base_as_calls:4.2f}")
    print(f"  Family A               F1={base_famA_f1:6.2f}")

    # Enumerate.
    n_e = len(df_e)
    n_t = len(df_t)
    rows = []
    seen = set()
    for k in range(1, len(signals) + 1):
        for subset in itertools.combinations(signals, k):
            for partition in set_partitions(list(subset)):
                key = canonical_partition(partition)
                if key in seen:
                    continue
                seen.add(key)
                e_mask = dnf_mask(eval_masks, partition, n_e)
                t_mask = dnf_mask(tune_masks, partition, n_t)
                pq_e = per_question_with_mask(df_e, e_mask)
                pq_t = per_question_with_mask(df_t, t_mask)
                triggered = int(pd.DataFrame({"qid": df_e["qid"], "_m": e_mask}).groupby("qid")["_m"].any().sum())
                bs_as = paired_bootstrap(pq_e, base_as, args.n_boot)
                bs_fa = paired_bootstrap(pq_e, base_famA, args.n_boot)
                bs_k3 = paired_bootstrap(pq_e, base_k3, args.n_boot)
                rows.append({
                    "size": k,
                    "n_groups": len(partition),
                    "max_group_size": max(len(g) for g in partition),
                    "formula": formula_str(partition),
                    "has_margin": int("calibrated_logit_margin" in subset),
                    "has_answer_stable": int("answer_stable" in subset),
                    "tune_f1": float(pq_t["current_f1"].mean() * 100),
                    "tune_calls": float(pq_t["round"].mean()),
                    "eval_f1": float(pq_e["current_f1"].mean() * 100),
                    "eval_em": float(pq_e["current_em"].mean() * 100),
                    "eval_calls": float(pq_e["round"].mean()),
                    "eval_triggered": triggered,
                    "diff_f1_vs_as": bs_as["diff_f1"],
                    "ci_as_lo": bs_as["diff_f1_ci95"][0],
                    "ci_as_hi": bs_as["diff_f1_ci95"][1],
                    "diff_calls_vs_as": bs_as["diff_calls"],
                    "diff_f1_vs_famA": bs_fa["diff_f1"],
                    "ci_famA_lo": bs_fa["diff_f1_ci95"][0],
                    "ci_famA_hi": bs_fa["diff_f1_ci95"][1],
                    "diff_f1_vs_k3": bs_k3["diff_f1"],
                    "ci_k3_lo": bs_k3["diff_f1_ci95"][0],
                    "ci_k3_hi": bs_k3["diff_f1_ci95"][1],
                })
    print(f"\nEnumerated {len(rows)} distinct DNF formulas")

    out_df = pd.DataFrame(rows).sort_values(["eval_f1", "eval_calls"], ascending=[False, True]).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    def verdict(lo, hi):
        return "WIN" if lo > 0 else "LOSS" if hi < 0 else "tie"

    def print_table(rows_df, title, max_rows=40):
        print(f"\n=== {title} (showing up to {max_rows}) ===")
        print(f"{'sz':>2} {'g':>2} {'formula':<55} {'eF1':>6} {'eC':>5} {'trg':>4} {'ΔvsAS':>7} {'CI vs AS':>16} {'v':>4} {'ΔvsFA':>7}")
        for _, r in rows_df.head(max_rows).iterrows():
            v = verdict(r["ci_as_lo"], r["ci_as_hi"])
            ci = f"[{r['ci_as_lo']:+5.2f},{r['ci_as_hi']:+5.2f}]"
            print(f"{int(r['size']):>2} {int(r['n_groups']):>2} {r['formula']:<55} {r['eval_f1']:6.2f} {r['eval_calls']:5.2f} {int(r['eval_triggered']):>4} "
                  f"{r['diff_f1_vs_as']:+7.2f} {ci:>16} {v:>4} {r['diff_f1_vs_famA']:+7.2f}")

    print_table(out_df, "TOP 40 BY EVAL F1")

    wins_vs_as = out_df[out_df["ci_as_lo"] > 0].copy()
    print(f"\n>>> Formulas significantly beating answer_stable: {len(wins_vs_as)}/{len(out_df)}")
    if len(wins_vs_as):
        print_table(wins_vs_as, f"ALL {len(wins_vs_as)} WINNERS vs answer_stable", max_rows=len(wins_vs_as))

    wins_no_as = wins_vs_as[wins_vs_as["has_answer_stable"] == 0]
    print(f"\n>>> Winners that DON'T use answer_stable: {len(wins_no_as)}")
    if len(wins_no_as):
        print_table(wins_no_as, "WINNERS WITHOUT answer_stable", max_rows=len(wins_no_as))

    wins_with_margin = wins_vs_as[wins_vs_as["has_margin"] == 1]
    print(f"\n>>> Winners using calibrated_logit_margin: {len(wins_with_margin)}")
    if len(wins_with_margin):
        print_table(wins_with_margin, "WINNERS USING margin", max_rows=len(wins_with_margin))

    # Top "others-only" formulas (no answer_stable), ranked by F1, even if not significant.
    no_as = out_df[out_df["has_answer_stable"] == 0]
    print(f"\n>>> Best 'others-only' formulas (no answer_stable), ranked by F1:")
    print_table(no_as.head(20), "TOP 20 NO-answer_stable FORMULAS", max_rows=20)

    out = {
        "signals": signals,
        "budget_max_avg_calls": args.budget,
        "locked_thresholds": {s: {"threshold": thr, "tune_f1": info["f1"], "tune_calls": info["calls"]}
                              for s, (thr, info) in locked.items()},
        "baselines": {
            "fixed_k3_f1": base_k3_f1,
            "answer_stable_f1": base_as_f1,
            "answer_stable_calls": base_as_calls,
            "family_a_f1": base_famA_f1,
        },
        "n_formulas": len(out_df),
        "n_winners_vs_answer_stable": int(len(wins_vs_as)),
        "n_winners_no_answer_stable": int(len(wins_no_as)),
        "n_winners_with_margin": int(len(wins_with_margin)),
        "top_10_by_f1": out_df.head(10)[["size", "n_groups", "formula", "eval_f1", "eval_calls",
                                          "eval_triggered", "diff_f1_vs_as", "ci_as_lo", "ci_as_hi"]].to_dict("records"),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
