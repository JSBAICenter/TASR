"""Build the enriched parquet for the logprobs variant.

Reads `signal_log_lp{,_tune}.parquet` (produced by run_instrumented_with_logprobs.py),
applies the existing signal computations, AND adds a new signal:

  calibrated_logit_margin: per-round isotonic regression on `answer_token_margin`
                           -> P(current_em). Same per-round pattern as
                           calibrated_conf, but the raw input is continuous and
                           genuinely informative (corr ~ +0.40 vs ~+0.15 for raw
                           confidence on the smoke test).

Output: `signal_log_lp_enriched{,_tune}.parquet` with all of the base columns
plus `calibrated_logit_margin`. Drop-in replacement for tune_thresholds.py if
we just want to swap the confidence signal for the margin signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from signals import (  # noqa: E402
    calibration_report,
    fit_isotonic_per_round,
    gap_1_2,
    overlap_signal,
    predict_per_round,
    top_score_z_global,
    top_score_z_intra,
)


def fit_margin_per_round(df_tune: pd.DataFrame, label_col: str = "current_em") -> dict:
    """Per-round isotonic fit on `answer_token_margin` -> P(em).

    Mirrors fit_isotonic_per_round but operates on continuous input. Degenerate
    rounds (all-NaN margins, or std=0) fall back to the round's empirical mean.
    """
    models: dict[int, IsotonicRegression | float] = {}
    for r, sub in df_tune.groupby("round"):
        x_all = sub["answer_token_margin"].values
        y_all = sub[label_col].values.astype(float)
        valid = ~np.isnan(x_all)
        x = x_all[valid].astype(float)
        y = y_all[valid]
        if len(x) < 5 or np.std(x) == 0:
            models[int(r)] = float(y_all.mean()) if len(y_all) else 0.5
        else:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
            iso.fit(x, y)
            models[int(r)] = iso
    return models


def predict_margin_per_round(models: dict, df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), 0.5, dtype=float)  # default for NaN margin
    rounds = df["round"].values
    x = df["answer_token_margin"].values
    for r, m in models.items():
        mask = rounds == r
        if not mask.any():
            continue
        if isinstance(m, float):
            out[mask] = m
        else:
            x_slice = x[mask]
            valid = ~np.isnan(x_slice)
            preds = np.full(mask.sum(), 0.5, dtype=float)
            if valid.any():
                preds[valid] = m.predict(x_slice[valid])
            out[mask] = preds
    return out


def margin_calibration_report(df_tune: pd.DataFrame, models: dict, label_col: str = "current_em") -> pd.DataFrame:
    """Per-round, bucket the continuous margin into deciles and report mean EM vs calibrated."""
    rows = []
    for r in sorted(df_tune["round"].unique()):
        sub = df_tune[df_tune["round"] == r].dropna(subset=["answer_token_margin"]).copy()
        if len(sub) == 0:
            continue
        m = models[int(r)]
        sub["margin_bucket"] = pd.qcut(sub["answer_token_margin"], q=min(5, sub["answer_token_margin"].nunique()), labels=False, duplicates="drop")
        for b, bsub in sub.groupby("margin_bucket"):
            cal = m if isinstance(m, float) else float(m.predict([bsub["answer_token_margin"].mean()])[0])
            rows.append({
                "round": int(r),
                "margin_bucket": int(b),
                "n_samples": len(bsub),
                "margin_mean": round(bsub["answer_token_margin"].mean(), 3),
                "raw_em_rate": round(bsub[label_col].mean(), 3),
                "calibrated_at_bucket_mean": round(cal, 3),
            })
    return pd.DataFrame(rows)


def enrich_lp(eval_path: str, tune_path: str, data_eval: str, data_tune: str, out_prefix: str) -> None:
    df_eval = pd.read_parquet(eval_path)
    df_tune = pd.read_parquet(tune_path)

    # Existing signals (recomputed on the LP data so everything is consistent).
    df_eval["top_score_z_global"] = top_score_z_global(df_eval)
    df_tune["top_score_z_global"] = top_score_z_global(df_tune)
    intra_eval = top_score_z_intra(data_eval)
    intra_tune = top_score_z_intra(data_tune)
    df_eval["top_score_z_intra"] = df_eval["qid"].map(intra_eval).astype(float)
    df_tune["top_score_z_intra"] = df_tune["qid"].map(intra_tune).astype(float)
    df_eval["gap"] = gap_1_2(df_eval)
    df_tune["gap"] = gap_1_2(df_tune)
    df_eval["overlap_signal"] = overlap_signal(df_eval)
    df_tune["overlap_signal"] = overlap_signal(df_tune)

    # calibrated_conf (per-round isotonic on raw 1-5 confidence).
    conf_models = fit_isotonic_per_round(df_tune, label_col="current_em")
    df_eval["calibrated_conf"] = predict_per_round(conf_models, df_eval)
    df_tune["calibrated_conf"] = predict_per_round(conf_models, df_tune)

    # NEW: calibrated_logit_margin (per-round isotonic on answer_token_margin).
    margin_models = fit_margin_per_round(df_tune, label_col="current_em")
    df_eval["calibrated_logit_margin"] = predict_margin_per_round(margin_models, df_eval)
    df_tune["calibrated_logit_margin"] = predict_margin_per_round(margin_models, df_tune)

    # Diagnostics.
    print("=== confidence calibration (raw conf -> calibrated, on tune) ===")
    conf_rep, conf_brier = calibration_report(df_tune, conf_models)
    print(conf_rep.to_string(index=False))
    print(f"Brier (conf): {conf_brier:.4f}")
    print()
    print("=== margin calibration (5-bucket per round, on tune) ===")
    margin_rep = margin_calibration_report(df_tune, margin_models)
    print(margin_rep.to_string(index=False))
    preds = predict_margin_per_round(margin_models, df_tune)
    brier = float(np.mean((preds - df_tune["current_em"].values) ** 2))
    print(f"Brier (margin): {brier:.4f}")
    print()
    print("=== signal comparison: corr with current_em (eval) ===")
    for col in ["calibrated_conf", "calibrated_logit_margin", "overlap_signal", "top_score_z_intra"]:
        c = df_eval[col].corr(df_eval["current_em"])
        print(f"  {col:30s} corr={c:+.3f}")

    eval_out = Path(f"{out_prefix}.parquet")
    tune_out = Path(f"{out_prefix}_tune.parquet")
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_parquet(eval_out, index=False)
    df_tune.to_parquet(tune_out, index=False)
    print(f"\nWrote {eval_out} ({df_eval.shape}) and {tune_out} ({df_tune.shape})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/signal_log_lp.parquet")
    ap.add_argument("--tune", default="results/signal_log_lp_tune.parquet")
    ap.add_argument("--data-eval", default="data/dev_eval.json")
    ap.add_argument("--data-tune", default="data/dev_tune.json")
    ap.add_argument("--out-prefix", default="results/signal_log_lp_enriched")
    args = ap.parse_args()
    enrich_lp(args.eval, args.tune, args.data_eval, args.data_tune, args.out_prefix)


if __name__ == "__main__":
    main()
