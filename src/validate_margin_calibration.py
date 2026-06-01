"""Quick diagnostic on dev_tune only.

Runs after run_instrumented_with_logprobs.py --split tune finishes. Validates
that the margin signal is worth the dev_eval re-run (which costs another ~2 hr).

Pass-fail criteria:
- Brier score on margin < Brier score on raw confidence (margin is a better
  predictor of correctness)
- Per-round mean(em=1) - mean(em=0) > 1.0 nat (clean separation in raw units)
- corr(margin, em) > 0.25 on at least 3 of 5 rounds
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))


def per_round_isotonic(df: pd.DataFrame, x_col: str) -> tuple[dict, float]:
    """Fit per-round isotonic, return models + overall Brier."""
    models: dict = {}
    preds = np.zeros(len(df))
    rounds = df["round"].values
    for r, sub in df.groupby("round"):
        x = sub[x_col].dropna().values
        y = sub.loc[sub[x_col].notna(), "current_em"].values.astype(float)
        if len(x) < 5 or np.std(x) == 0:
            mean = float(sub["current_em"].mean()) if len(sub) else 0.5
            models[int(r)] = mean
            preds[rounds == r] = mean
        else:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip", increasing=True)
            iso.fit(x.astype(float), y)
            models[int(r)] = iso
            mask = rounds == r
            x_slice = df.loc[mask, x_col].values
            valid = ~np.isnan(x_slice)
            out_slice = np.full(mask.sum(), float(sub["current_em"].mean()), dtype=float)
            if valid.any():
                out_slice[valid] = iso.predict(x_slice[valid].astype(float))
            preds[mask] = out_slice
    y_true = df["current_em"].values.astype(float)
    brier = float(np.mean((preds - y_true) ** 2))
    return models, brier


def main() -> None:
    path = "results/signal_log_lp_tune.parquet"
    if not Path(path).exists():
        print(f"ERROR: {path} not found - is the dev_tune run still in progress?")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"Loaded {path}: {len(df)} rows ({df['qid'].nunique()} questions)")
    print()
    print("=" * 72)
    print("Per-round raw signal stats (margin vs confidence)")
    print("=" * 72)
    rows = []
    for r in sorted(df["round"].unique()):
        sub = df[df["round"] == r]
        sub_margin = sub.dropna(subset=["answer_token_margin"])
        corr_m = sub_margin["answer_token_margin"].corr(sub_margin["current_em"]) if len(sub_margin) > 1 else float("nan")
        corr_c = sub["llm_confidence"].corr(sub["current_em"]) if sub["llm_confidence"].std() > 0 else float("nan")
        sep_m = (sub_margin[sub_margin["current_em"] == 1]["answer_token_margin"].mean()
                 - sub_margin[sub_margin["current_em"] == 0]["answer_token_margin"].mean())
        sep_c = (sub[sub["current_em"] == 1]["llm_confidence"].mean()
                 - sub[sub["current_em"] == 0]["llm_confidence"].mean())
        rows.append({
            "round": int(r),
            "n": len(sub),
            "corr_margin": round(corr_m, 3),
            "corr_conf": round(corr_c, 3),
            "sep_margin (em=1 minus em=0)": round(sep_m, 2),
            "sep_conf (em=1 minus em=0)": round(sep_c, 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("=" * 72)
    print("Calibration quality (Brier score, lower is better)")
    print("=" * 72)
    _, brier_m = per_round_isotonic(df, "answer_token_margin")
    _, brier_c = per_round_isotonic(df, "llm_confidence")
    p_em = df["current_em"].mean()
    brier_baseline = p_em * (1 - p_em)
    print(f"  Brier on margin:     {brier_m:.4f}")
    print(f"  Brier on confidence: {brier_c:.4f}")
    print(f"  Brier baseline (p*(1-p)): {brier_baseline:.4f}")
    print()
    if brier_m < brier_c:
        print(f"  Margin BEATS confidence by {(brier_c - brier_m):.4f} Brier units")
    else:
        print(f"  Margin LOSES to confidence by {(brier_m - brier_c):.4f} Brier units")
    if brier_m < brier_baseline:
        print(f"  Margin BEATS uninformative baseline by {(brier_baseline - brier_m):.4f}")
    else:
        print(f"  Margin LOSES to uninformative baseline (signal probably useless)")

    print()
    print("=" * 72)
    print("Verdict")
    print("=" * 72)
    corr_count = sum(1 for r in rows if r["corr_margin"] > 0.25)
    sep_count = sum(1 for r in rows if r["sep_margin (em=1 minus em=0)"] > 1.0)
    margin_better = brier_m < brier_c
    print(f"  Corr > 0.25 on {corr_count}/5 rounds (need >=3)")
    print(f"  Separation > 1.0 nat on {sep_count}/5 rounds (need >=3)")
    print(f"  Margin Brier < Confidence Brier: {margin_better}")
    if corr_count >= 3 and sep_count >= 3 and margin_better:
        print("\n  GO: kick off dev_eval run. Margin is a real signal.")
    else:
        print("\n  STOP: signal too weak. Investigate before committing dev_eval time.")


if __name__ == "__main__":
    main()
