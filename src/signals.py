"""Signal extraction for the budget-aware stopping rule.

Reads an instrumented parquet (canonical or confidence-first), computes the
candidate signals, fits the isotonic confidence calibrator on the tune split,
and writes enriched parquets.

Signals:
- top_score_z_global: z-score of top_bm25_score across all rows in the split.
                      Tells us if THIS question's retrieval is unusually
                      confident vs the population of other questions.
- top_score_z_intra:  z-score of rank-1 within THIS question's 10-paragraph
                      BM25 pool. Stronger retrieval-confidence signal.
- gap:                rank1 - rank2 BM25 score (validated pass-through).
- overlap_signal:     Jaccard of new vs prior evidence tokens (pass-through).
                      Always 0 at round 1 by construction.
- calibrated_conf:    per-round isotonic mapping llm_confidence -> P(current_em),
                      fit on tune split. Per-round because the same raw
                      confidence carries different reliability across rounds
                      (e.g. canonical: round-1 conf=5 -> 0.45 EM, round-5
                      conf=5 -> 0.64 EM). Rounds with <2 distinct confidence
                      values fall back to the empirical mean for that round.

Usage:
  python src/signals.py                      # canonical: signal_log{,_tune}.parquet
  python src/signals.py --eval results/signal_log_confirst.parquet \
                       --tune results/signal_log_confirst_tune.parquet \
                       --out-prefix results/signal_log_confirst_enriched
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

from retrieval import build_paragraphs, retrieve  # noqa: E402


def top_score_z_global(df: pd.DataFrame) -> pd.Series:
    col = df["top_bm25_score"]
    mu, sigma = col.mean(), col.std()
    if sigma == 0:
        return pd.Series(0.0, index=df.index)
    return (col - mu) / sigma


def top_score_z_intra(data_path: str) -> dict[str, float]:
    """Per-qid: z-score of rank-1 BM25 within this question's 10 paragraphs.

    Returns a {qid: z} mapping. Caller broadcasts across rounds.
    """
    with open(data_path) as f:
        items = json.load(f)
    out: dict[str, float] = {}
    for item in items:
        paragraphs = build_paragraphs(item)
        ranked = retrieve(item["question"], paragraphs, k=len(paragraphs))
        scores = np.array([r["score"] for r in ranked])
        mu, sigma = scores.mean(), scores.std()
        out[item["id"]] = 0.0 if sigma == 0 else float((scores[0] - mu) / sigma)
    return out


def gap_1_2(df: pd.DataFrame) -> pd.Series:
    gap = df["top_bm25_score"] - df["rank2_bm25_score"]
    if not (gap >= -1e-9).all():
        raise ValueError("gap_1_2 has negative values - check scoring")
    return gap.clip(lower=0.0)


def overlap_signal(df: pd.DataFrame) -> pd.Series:
    ov = df["jaccard_overlap"].copy()
    if not (ov[df["round"] == 1] == 0).all():
        raise ValueError("Round-1 overlap should be 0 by construction")
    if not ov.between(0.0, 1.0).all():
        raise ValueError("overlap values outside [0,1]")
    return ov


PerRoundCalibrators = dict[int, IsotonicRegression | float]


def fit_isotonic_per_round(df_tune: pd.DataFrame, label_col: str = "current_em") -> PerRoundCalibrators:
    """One isotonic per round. Degenerate rounds (single conf value) -> constant mean."""
    models: PerRoundCalibrators = {}
    for r, sub in df_tune.groupby("round"):
        x = sub["llm_confidence"].values.astype(float)
        y = sub[label_col].values.astype(float)
        if len(np.unique(x)) < 2:
            models[int(r)] = float(y.mean()) if len(y) else 0.0
        else:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
            iso.fit(x, y)
            models[int(r)] = iso
    return models


def predict_per_round(models: PerRoundCalibrators, df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    rounds = df["round"].values
    conf = df["llm_confidence"].values.astype(float)
    for r, m in models.items():
        mask = rounds == r
        if not mask.any():
            continue
        out[mask] = m if isinstance(m, float) else m.predict(conf[mask])
    return out


def calibration_report(df_tune: pd.DataFrame, models: PerRoundCalibrators, label_col: str = "current_em") -> tuple[pd.DataFrame, float]:
    rows = []
    for r in sorted(df_tune["round"].unique()):
        sub = df_tune[df_tune["round"] == r]
        m = models[int(r)]
        for c in sorted(sub["llm_confidence"].unique()):
            cmask = sub["llm_confidence"] == c
            cal = m if isinstance(m, float) else float(m.predict([float(c)])[0])
            rows.append({
                "round": int(r),
                "raw_conf": int(c),
                "n_samples": int(cmask.sum()),
                "raw_em_rate": round(float(sub.loc[cmask, label_col].mean()), 4),
                "calibrated": round(cal, 4),
                "degenerate": isinstance(m, float),
            })
    preds = predict_per_round(models, df_tune)
    brier = float(np.mean((preds - df_tune[label_col].values.astype(float)) ** 2))
    return pd.DataFrame(rows), brier


def enrich(eval_path: str, tune_path: str, data_eval: str, data_tune: str, out_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_eval = pd.read_parquet(eval_path)
    df_tune = pd.read_parquet(tune_path)

    # Global z (computed within each split independently).
    df_eval["top_score_z_global"] = top_score_z_global(df_eval)
    df_tune["top_score_z_global"] = top_score_z_global(df_tune)

    # Intra-pool z (one value per qid, broadcast).
    intra_eval = top_score_z_intra(data_eval)
    intra_tune = top_score_z_intra(data_tune)
    df_eval["top_score_z_intra"] = df_eval["qid"].map(intra_eval).astype(float)
    df_tune["top_score_z_intra"] = df_tune["qid"].map(intra_tune).astype(float)

    df_eval["gap"] = gap_1_2(df_eval)
    df_tune["gap"] = gap_1_2(df_tune)
    df_eval["overlap_signal"] = overlap_signal(df_eval)
    df_tune["overlap_signal"] = overlap_signal(df_tune)

    models = fit_isotonic_per_round(df_tune, label_col="current_em")
    df_eval["calibrated_conf"] = predict_per_round(models, df_eval)
    df_tune["calibrated_conf"] = predict_per_round(models, df_tune)

    rep, brier = calibration_report(df_tune, models)
    print("Per-round calibration mapping (raw conf -> P(correct) on tune):")
    print(rep.to_string(index=False))
    degen = [r for r, m in models.items() if isinstance(m, float)]
    if degen:
        print(f"  Degenerate rounds (single conf value, used empirical mean): {degen}")
    print(f"Brier score on tune: {brier:.4f} (lower is better; baseline = p*(1-p))")

    eval_out = Path(f"{out_prefix}.parquet")
    tune_out = Path(f"{out_prefix}_tune.parquet")
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_parquet(eval_out, index=False)
    df_tune.to_parquet(tune_out, index=False)
    print(f"\nWrote {eval_out} ({df_eval.shape}) and {tune_out} ({df_tune.shape})")

    cols = ["top_score_z_global", "top_score_z_intra", "gap", "overlap_signal", "calibrated_conf"]
    print("\nNaN per signal (eval):")
    print(df_eval[cols].isna().sum().to_string())
    print("\nSignal ranges on eval:")
    for c in cols:
        print(f"  {c:22s} [{df_eval[c].min():+.3f}, {df_eval[c].max():+.3f}]")

    return df_eval, df_tune


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/signal_log.parquet")
    ap.add_argument("--tune", default="results/signal_log_tune.parquet")
    ap.add_argument("--data-eval", default="data/dev_eval.json")
    ap.add_argument("--data-tune", default="data/dev_tune.json")
    ap.add_argument("--out-prefix", default="results/signal_log_enriched")
    args = ap.parse_args()
    enrich(args.eval, args.tune, args.data_eval, args.data_tune, args.out_prefix)


if __name__ == "__main__":
    main()
