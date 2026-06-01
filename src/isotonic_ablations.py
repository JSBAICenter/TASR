"""Two isotonic calibration ablations on the canonical signal log.

A. F1-target: fit isotonic to map llm_confidence -> current_f1 instead of
   current_em. Tells us whether confidence tracks partial correctness more
   reliably than exact-match.

B. Per-round: fit one isotonic per round on tune, score the matching round
   on eval. Tells us whether confidence reliability drifts across rounds
   (e.g. round-1 calibration vs round-5).

Reports Brier (lower better) and Cohen's d of the calibrated score for
correct vs incorrect rows on eval. Compares each ablation to the
EM-target global baseline that ships in signal_log_enriched.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def fit_iso(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(x, y)
    return iso


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (a.std() + b.std()) / 2 + 1e-9
    return abs(a.mean() - b.mean()) / pooled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/signal_log_enriched.parquet")
    ap.add_argument("--tune", default="results/signal_log_enriched_tune.parquet")
    args = ap.parse_args()

    df_eval = pd.read_parquet(args.eval)
    df_tune = pd.read_parquet(args.tune)

    x_tune = df_tune["llm_confidence"].values.astype(float)
    x_eval = df_eval["llm_confidence"].values.astype(float)
    y_em_tune = df_tune["current_em"].values.astype(float)
    y_em_eval = df_eval["current_em"].values.astype(float)
    y_f1_tune = df_tune["current_f1"].values.astype(float)
    y_f1_eval = df_eval["current_f1"].values.astype(float)

    print("=" * 72)
    print("Isotonic ablations on canonical signal log")
    print(f"  tune n={len(df_tune)}   eval n={len(df_eval)}")
    print("=" * 72)

    # --- Baseline already in parquet: global EM-target ---
    base_eval = df_eval["calibrated_conf"].values
    print("\n[Baseline] global isotonic, target=current_em (shipped)")
    print(f"  Brier(eval, EM): {brier(base_eval, y_em_eval):.4f}")
    print(f"  Brier(eval, F1): {brier(base_eval, y_f1_eval):.4f}")
    print(f"  Cohen's d (correct vs incorrect on EM): "
          f"{cohens_d(base_eval[y_em_eval == 1], base_eval[y_em_eval == 0]):.3f}")

    # --- Ablation A: target=F1 ---
    iso_f1 = fit_iso(x_tune, y_f1_tune)
    p_f1_tune = iso_f1.predict(x_tune)
    p_f1_eval = iso_f1.predict(x_eval)
    print("\n[Ablation A] global isotonic, target=current_f1")
    print(f"  raw_conf -> calibrated mapping:")
    for c in sorted(np.unique(x_tune)):
        em_rate = float(y_em_tune[x_tune == c].mean())
        f1_rate = float(y_f1_tune[x_tune == c].mean())
        print(f"    conf={int(c)}  n={int((x_tune == c).sum()):3d}  "
              f"EM_rate={em_rate:.3f}  F1_rate={f1_rate:.3f}  "
              f"cal_f1={float(iso_f1.predict([c])[0]):.3f}")
    print(f"  Brier(tune, F1): {brier(p_f1_tune, y_f1_tune):.4f}")
    print(f"  Brier(eval, F1): {brier(p_f1_eval, y_f1_eval):.4f}")
    print(f"  Brier(eval, EM): {brier(p_f1_eval, y_em_eval):.4f}")
    print(f"  Cohen's d (correct vs incorrect on EM): "
          f"{cohens_d(p_f1_eval[y_em_eval == 1], p_f1_eval[y_em_eval == 0]):.3f}")

    # --- Ablation B: per-round isotonic, target=EM ---
    print("\n[Ablation B] per-round isotonic, target=current_em")
    rounds = sorted(df_tune["round"].unique())
    per_round_pred = np.zeros_like(x_eval)
    for r in rounds:
        m_t = df_tune["round"].values == r
        m_e = df_eval["round"].values == r
        if m_t.sum() < 2 or len(np.unique(x_tune[m_t])) < 2:
            # singleton -> fall back to mean
            p = float(y_em_tune[m_t].mean()) if m_t.any() else 0.0
            per_round_pred[m_e] = p
            print(f"  round={r}: n_tune={int(m_t.sum())} - "
                  f"insufficient variation, used constant p={p:.3f}")
            continue
        iso_r = fit_iso(x_tune[m_t], y_em_tune[m_t])
        per_round_pred[m_e] = iso_r.predict(x_eval[m_e])
        cs = sorted(np.unique(x_tune[m_t]))
        mapping = ", ".join(f"{int(c)}->{float(iso_r.predict([c])[0]):.2f}" for c in cs)
        print(f"  round={r}: n_tune={int(m_t.sum())}  n_eval={int(m_e.sum())}  "
              f"mapping {{{mapping}}}")
    print(f"  Brier(eval, EM): {brier(per_round_pred, y_em_eval):.4f}")
    print(f"  Brier(eval, F1): {brier(per_round_pred, y_f1_eval):.4f}")
    print(f"  Cohen's d (correct vs incorrect on EM): "
          f"{cohens_d(per_round_pred[y_em_eval == 1], per_round_pred[y_em_eval == 0]):.3f}")

    # --- Combo C: per-round, target=F1 ---
    print("\n[Ablation C] per-round isotonic, target=current_f1")
    per_round_f1 = np.zeros_like(x_eval)
    for r in rounds:
        m_t = df_tune["round"].values == r
        m_e = df_eval["round"].values == r
        if m_t.sum() < 2 or len(np.unique(x_tune[m_t])) < 2:
            p = float(y_f1_tune[m_t].mean()) if m_t.any() else 0.0
            per_round_f1[m_e] = p
            continue
        iso_r = fit_iso(x_tune[m_t], y_f1_tune[m_t])
        per_round_f1[m_e] = iso_r.predict(x_eval[m_e])
    print(f"  Brier(eval, F1): {brier(per_round_f1, y_f1_eval):.4f}")
    print(f"  Brier(eval, EM): {brier(per_round_f1, y_em_eval):.4f}")
    print(f"  Cohen's d (correct vs incorrect on EM): "
          f"{cohens_d(per_round_f1[y_em_eval == 1], per_round_f1[y_em_eval == 0]):.3f}")

    print("\n" + "=" * 72)
    print("Summary (eval Brier, lower=better; d on EM split, higher=better):")
    print("=" * 72)
    rows = [
        ("global, target=EM (baseline)", brier(base_eval, y_em_eval),
         brier(base_eval, y_f1_eval),
         cohens_d(base_eval[y_em_eval == 1], base_eval[y_em_eval == 0])),
        ("global, target=F1            ", brier(p_f1_eval, y_em_eval),
         brier(p_f1_eval, y_f1_eval),
         cohens_d(p_f1_eval[y_em_eval == 1], p_f1_eval[y_em_eval == 0])),
        ("per-round, target=EM         ", brier(per_round_pred, y_em_eval),
         brier(per_round_pred, y_f1_eval),
         cohens_d(per_round_pred[y_em_eval == 1], per_round_pred[y_em_eval == 0])),
        ("per-round, target=F1         ", brier(per_round_f1, y_em_eval),
         brier(per_round_f1, y_f1_eval),
         cohens_d(per_round_f1[y_em_eval == 1], per_round_f1[y_em_eval == 0])),
    ]
    print(f"  {'variant':32s}  {'Brier_EM':>9s}  {'Brier_F1':>9s}  {'d_EM':>6s}")
    for name, be, bf, d in rows:
        print(f"  {name:32s}  {be:9.4f}  {bf:9.4f}  {d:6.3f}")


if __name__ == "__main__":
    main()
