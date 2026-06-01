"""Comprehensive subset ablation: every OR-combination of stopping signals.

For each signal, tune its best individual threshold on tune (within budget).
Then enumerate every non-empty subset of signals (2^N - 1 of them), OR-combine
using those locked per-signal thresholds, evaluate on eval, and (for top-K)
run a paired bootstrap vs fixed-k=3.

This answers: "do any signal combinations win where singletons don't?"

Singletons fall out as size-1 subsets. Confirst parquet has fewer signals
(no judge, no CE) so its sweep is smaller. Both are run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


BINARY_SIGNALS = {"answer_stable", "answer_in_evidence"}
ALL_SIGNALS = [
    "answer_stable",
    "answer_stable_f1",
    "answer_stable_seq",
    "answer_in_evidence",
    "calibrated_conf",
    "overlap_signal",
    "top_score_z_intra",
    "ce_top_score",
    "judge_entailment",
    "gap",
]


def per_question_with_mask(df: pd.DataFrame, stop_mask: np.ndarray) -> pd.DataFrame:
    """Given a per-row boolean stop_mask, take first stop per qid; fallback to last round."""
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
        "diff_f1": float((o_f1 - b_f1).mean() * 100),
        "diff_f1_ci95": [float(lo * 100), float(hi * 100)],
    }


def sweep_signal_threshold(df_tune: pd.DataFrame, sig: str, budget: float) -> tuple[float, dict]:
    """Find the best threshold for a single signal on tune within budget."""
    if sig in BINARY_SIGNALS:
        mask = (df_tune[sig].values == 1)
        pq = per_question_with_mask(df_tune, mask)
        return 1.0, {
            "f1": float(pq["current_f1"].mean() * 100),
            "calls": float(pq["round"].mean()),
        }
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_enriched.parquet")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-size", type=int, default=None, help="cap subset size (default: all)")
    ap.add_argument("--top-k", type=int, default=30, help="how many top subsets get full bootstrap")
    ap.add_argument("--out-csv", default="results/subset_ablation_canonical.csv")
    ap.add_argument("--out-top", default="results/subset_ablation_canonical_topK.json")
    ap.add_argument("--label", default="canonical")
    args = ap.parse_args()

    df_t = pd.read_parquet(args.tune)
    df_e = pd.read_parquet(args.eval)

    signals = [s for s in ALL_SIGNALS if s in df_t.columns and s in df_e.columns]
    print(f"[{args.label}] {len(signals)} signals available: {signals}")

    print(f"\n[{args.label}] Per-signal tuning on {args.tune} (budget calls ≤ {args.budget}):")
    locked: dict[str, tuple[float, dict]] = {}
    for sig in signals:
        thr, info = sweep_signal_threshold(df_t, sig, args.budget)
        locked[sig] = (thr, info)
        op_str = "==1" if sig in BINARY_SIGNALS else f">{thr:.3f}"
        print(f"  {sig:<24}  rule={op_str:<14}  tune_F1={info['f1']:6.2f}  tune_calls={info['calls']:4.2f}")

    base = per_question_fixed_k(df_e, 3)
    base_f1 = float(base["current_f1"].mean() * 100)
    base_calls = float(base["round"].mean())
    print(f"\n[{args.label}] Reference fixed_k=3: F1={base_f1:.2f}, calls={base_calls:.2f}")

    print(f"\n[{args.label}] Pre-computing eval per-row masks for each signal...")
    eval_masks = {sig: signal_mask(df_e, sig, thr) for sig, (thr, _) in locked.items()}
    tune_masks = {sig: signal_mask(df_t, sig, thr) for sig, (thr, _) in locked.items()}

    max_size = args.max_size or len(signals)
    rows = []
    print(f"\n[{args.label}] Enumerating subsets of size 1..{max_size}...")
    total = 0
    for r in range(1, max_size + 1):
        total += sum(1 for _ in itertools.combinations(signals, r))
    print(f"  total subsets: {total}")

    counter = 0
    for r in range(1, max_size + 1):
        for combo in itertools.combinations(signals, r):
            counter += 1
            e_mask = np.zeros(len(df_e), dtype=bool)
            t_mask = np.zeros(len(df_t), dtype=bool)
            for sig in combo:
                e_mask |= eval_masks[sig]
                t_mask |= tune_masks[sig]
            pq_e = per_question_with_mask(df_e, e_mask)
            pq_t = per_question_with_mask(df_t, t_mask)
            rows.append({
                "size": r,
                "subset": "+".join(combo),
                "tune_f1": float(pq_t["current_f1"].mean() * 100),
                "tune_calls": float(pq_t["round"].mean()),
                "eval_f1": float(pq_e["current_f1"].mean() * 100),
                "eval_calls": float(pq_e["round"].mean()),
                "eval_em": float(pq_e["current_em"].mean() * 100),
            })
            if counter % 50 == 0:
                print(f"  {counter}/{total} subsets evaluated...", flush=True)

    out_df = pd.DataFrame(rows)
    out_df["diff_f1_vs_k3"] = out_df["eval_f1"] - base_f1
    out_df["diff_calls_vs_k3"] = out_df["eval_calls"] - base_calls
    out_df["f1_per_call"] = out_df["eval_f1"] / out_df["eval_calls"]
    out_df["f1_minus_05calls"] = out_df["eval_f1"] - 0.5 * out_df["eval_calls"]

    out_df = out_df.sort_values(["eval_f1", "eval_calls"], ascending=[False, True]).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\n[{args.label}] Wrote {args.out_csv}  ({len(out_df)} rows)")

    # Full bootstrap on top-K by eval F1 (and separately by F1−0.5*calls)
    top_by_f1 = out_df.head(args.top_k)
    top_by_obj = out_df.sort_values("f1_minus_05calls", ascending=False).head(args.top_k)
    top_subsets = set(top_by_f1["subset"]) | set(top_by_obj["subset"])

    boots = {}
    for sub_str in top_subsets:
        sigs = sub_str.split("+")
        e_mask = np.zeros(len(df_e), dtype=bool)
        for sig in sigs:
            e_mask |= eval_masks[sig]
        pq_e = per_question_with_mask(df_e, e_mask)
        b = paired_bootstrap(pq_e, base, args.n_boot)
        boots[sub_str] = b

    def fmt(row):
        b = boots.get(row["subset"], {})
        lo, hi = b.get("diff_f1_ci95", [float("nan"), float("nan")])
        verdict = "WIN" if lo > 0 else "LOSS" if hi < 0 else "tie"
        return (row["size"], row["subset"], row["eval_f1"], row["eval_calls"],
                row["diff_f1_vs_k3"], lo, hi, verdict, row["f1_minus_05calls"])

    print(f"\n[{args.label}] === TOP {args.top_k} BY EVAL F1 ===")
    print(f"{'sz':>2} {'subset':<60} {'eF1':>6} {'eC':>5} {'ΔF1':>7} {'CI95':>16} {'verdict':>8} {'F1-.5C':>7}")
    for _, row in top_by_f1.iterrows():
        sz, name, f1, c, d, lo, hi, v, obj = fmt(row)
        ci = f"[{lo:+5.2f},{hi:+5.2f}]" if not np.isnan(lo) else " "*16
        print(f"{sz:>2} {name:<60} {f1:6.2f} {c:5.2f} {d:+7.2f} {ci:>16} {v:>8} {obj:7.2f}")

    print(f"\n[{args.label}] === TOP {args.top_k} BY F1 − 0.5·calls (paper sensitivity metric) ===")
    print(f"{'sz':>2} {'subset':<60} {'eF1':>6} {'eC':>5} {'ΔF1':>7} {'CI95':>16} {'verdict':>8} {'F1-.5C':>7}")
    for _, row in top_by_obj.iterrows():
        sz, name, f1, c, d, lo, hi, v, obj = fmt(row)
        ci = f"[{lo:+5.2f},{hi:+5.2f}]" if not np.isnan(lo) else " "*16
        print(f"{sz:>2} {name:<60} {f1:6.2f} {c:5.2f} {d:+7.2f} {ci:>16} {v:>8} {obj:7.2f}")

    out = {
        "label": args.label,
        "signals": signals,
        "locked_thresholds": {s: {"threshold": thr, "tune_f1": info["f1"], "tune_calls": info["calls"]}
                              for s, (thr, info) in locked.items()},
        "fixed_k3_eval": {"f1": base_f1, "calls": base_calls},
        "top_k_bootstrap": boots,
        "n_subsets": len(out_df),
    }
    Path(args.out_top).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out_top}")


if __name__ == "__main__":
    main()
