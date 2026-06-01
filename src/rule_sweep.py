"""Try several stopping-rule variants over the now-six-signal log.

Each rule is a pandas query string. We simulate stop-at-first-True-row,
fall back to the last round. Reports tune + eval metrics side-by-side.

Signals available: calibrated_conf, overlap_signal, answer_stable,
answer_in_evidence, top_score_z_global/intra, gap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def simulate(df: pd.DataFrame, rule: str) -> dict:
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = df.eval(rule)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return {
        "em": float(chosen["current_em"].mean() * 100),
        "f1": float(chosen["current_f1"].mean() * 100),
        "avg_calls": float(chosen["round"].mean()),
        "n_stopped_early": int(len(stops)),
        "n_questions": int(len(chosen)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="results/signal_log_enriched_tune.parquet")
    ap.add_argument("--eval", default="results/signal_log_enriched.parquet")
    args = ap.parse_args()

    df_t = pd.read_parquet(args.tune)
    df_e = pd.read_parquet(args.eval)

    rules = [
        ("(A) status quo locked",       "calibrated_conf > 0.58 or overlap_signal > 0.125"),
        ("(B) stable only",             "answer_stable == 1"),
        ("(C) grounded only",           "answer_in_evidence == 1"),
        ("(D) stable OR grounded",      "answer_stable == 1 or answer_in_evidence == 1"),
        ("(E) stable AND grounded",     "answer_stable == 1 and answer_in_evidence == 1"),
        ("(F) stable OR conf>0.55",     "answer_stable == 1 or calibrated_conf > 0.55"),
        ("(G) (st AND gr) OR conf>0.58","(answer_stable == 1 and answer_in_evidence == 1) or calibrated_conf > 0.58"),
        ("(H) all 4 OR (loose)",        "answer_stable == 1 or answer_in_evidence == 1 or calibrated_conf > 0.58 or overlap_signal > 0.125"),
        ("(I) stable OR (conf AND gr)", "answer_stable == 1 or (calibrated_conf > 0.55 and answer_in_evidence == 1)"),
    ]

    header = (
        f"{'rule':<32}  "
        f"{'tune_F1':>7} {'tune_EM':>7} {'tune_calls':>10}    "
        f"{'eval_F1':>7} {'eval_EM':>7} {'eval_calls':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, rule in rules:
        t = simulate(df_t, rule)
        e = simulate(df_e, rule)
        print(
            f"{name:<32}  "
            f"{t['f1']:7.2f} {t['em']:7.2f} {t['avg_calls']:10.2f}    "
            f"{e['f1']:7.2f} {e['em']:7.2f} {e['avg_calls']:10.2f}"
        )

    print("\nReference (dev_eval):")
    print(f"  fixed-k=2            F1 63.02  EM 53.00  calls 2.00")
    print(f"  fixed-k=3            F1 65.45  EM 54.67  calls 3.00   <-- target")
    print(f"  fixed-k=5            F1 69.42  EM 57.67  calls 5.00")
    print(f"  oracle               F1 70.92  EM 60.33  calls 2.90   (upper bound)")


if __name__ == "__main__":
    main()
