"""Paired bootstrap CIs for AS_m25 vs fixed-k baselines on open-domain cells.

Reads per-question parquets, enriches with answer_stable and
calibrated_logit_margin (using distractor tune-split calibration), applies
AS_m25 rule, and computes paired bootstrap CIs vs fixed-k={1,3,5}.

Usage:
  python src/bootstrap_ci_opendomain.py
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


SHARED = str(Path(__file__).resolve().parent.parent / "results")

MODEL_TUNE = {
    "qwen": [
        "results/signal_log_lp_tune.parquet",
    ],
    "devstral": [
        f"{SHARED}/devstral/signal_log_lp_tune.parquet",
        "results/devstral/signal_log_lp_tune.parquet",
    ],
    "gemma": [
        f"{SHARED}/gemma/signal_log_lp_tune.parquet",
        "results/gemma/signal_log_lp_tune.parquet",
    ],
}

EVAL_PATHS = {
    ("qwen", "fullwiki"):     [f"{SHARED}/qwen_fullwiki/signal_log_lp.parquet",
                               "results/qwen_fullwiki/signal_log_lp.parquet"],
    ("qwen", "nq"):           [f"{SHARED}/qwen_nq/signal_log_lp.parquet",
                               "results/qwen_nq/signal_log_lp.parquet"],
    ("qwen", "trivia"):       [f"{SHARED}/qwen_trivia/signal_log_lp.parquet",
                               "results/qwen_trivia/signal_log_lp.parquet"],
    ("devstral", "fullwiki"): ["results/devstral_fullwiki/signal_log_lp.parquet",
                               f"{SHARED}/devstral_fullwiki/signal_log_lp.parquet"],
    ("devstral", "nq"):       ["results/devstral_nq/signal_log_lp.parquet",
                               f"{SHARED}/devstral_nq/signal_log_lp.parquet"],
    ("gemma", "fullwiki"):    ["results/gemma_fullwiki/signal_log_lp.parquet",
                               f"{SHARED}/gemma_fullwiki/signal_log_lp.parquet"],
    ("gemma", "nq"):          ["results/gemma_nq/signal_log_lp.parquet",
                               f"{SHARED}/gemma_nq/signal_log_lp.parquet"],
    ("devstral", "trivia"):   ["results/devstral_trivia/signal_log_lp.parquet"],
    ("gemma", "trivia"):      ["results/gemma_trivia/signal_log_lp.parquet"],
}

MARGIN_COL = "answer_token_margin"


def _ensure_margin_col(df):
    if MARGIN_COL not in df.columns and "margin_first" in df.columns:
        df = df.rename(columns={"margin_first": MARGIN_COL})
    return df


def fit_margin_per_round(df_tune, label_col="current_em"):
    models = {}
    for r, sub in df_tune.groupby("round"):
        x_all = sub[MARGIN_COL].values
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


def predict_margin_per_round(models, df):
    out = np.full(len(df), 0.5, dtype=float)
    rounds = df["round"].values
    x = df[MARGIN_COL].values
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


def _normalize_answer(s):
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def add_answer_stable(df):
    df = df.sort_values(["qid", "round"]).copy()
    df["_norm"] = df["current_answer"].map(_normalize_answer)
    df["_prev"] = df.groupby("qid")["_norm"].shift(1)
    df["answer_stable"] = ((df["_prev"].notna()) & (df["_prev"] == df["_norm"])).astype(int)
    return df.drop(columns=["_norm", "_prev"])


def per_question_rule(df, rule):
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = df.eval(rule)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def per_question_fixed_k(df, k):
    return df[df["round"] == k][["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def paired_bootstrap(ours, base, n_boot=1000, seed=42):
    assert (ours["qid"].values == base["qid"].values).all(), "qid mismatch"
    rng = np.random.default_rng(seed)
    n = len(ours)
    o_f1 = ours["current_f1"].values
    o_r = ours["round"].values
    b_f1 = base["current_f1"].values
    b_r = base["round"].values

    diff_f1 = np.zeros(n_boot)
    diff_r = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diff_f1[i] = (o_f1[idx] - b_f1[idx]).mean()
        diff_r[i] = (o_r[idx] - b_r[idx]).mean()

    def ci(arr, scale=1.0):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return [round(float(lo * scale), 3), round(float(hi * scale), 3)]

    lo, hi = ci(diff_f1, scale=100)
    if lo > 0:
        verdict = "SIGNIFICANT WIN"
    elif hi < 0:
        verdict = "SIGNIFICANT LOSS"
    else:
        verdict = "NOT SIGNIFICANT"

    return {
        "ours_f1": round(float(o_f1.mean() * 100), 2),
        "base_f1": round(float(b_f1.mean() * 100), 2),
        "diff_f1_mean": round(float((o_f1 - b_f1).mean() * 100), 2),
        "diff_f1_ci95": [lo, hi],
        "diff_calls_mean": round(float((o_r - b_r).mean()), 3),
        "diff_calls_ci95": ci(diff_r),
        "p_ours_worse_f1": round(float((diff_f1 < 0).mean()), 4),
        "verdict": verdict,
    }


def main():
    out_path = Path("results/bootstrap_ci_opendomain.json")

    model_calibrators = {}
    for m, paths in MODEL_TUNE.items():
        for p in paths:
            if Path(p).exists():
                df_t = _ensure_margin_col(pd.read_parquet(p))
                if MARGIN_COL not in df_t.columns:
                    print(f"  [skip] {m}: {p} has no margin column")
                    continue
                model_calibrators[m] = fit_margin_per_round(df_t, "current_em")
                print(f"  {m}: fit calibrator on {df_t.qid.nunique()} tune Qs from {p}")
                break
        else:
            print(f"[skip tune] {m}: no tune parquet found")

    results = {}
    for (m, c), paths in sorted(EVAL_PATHS.items()):
        if m not in model_calibrators:
            print(f"[skip] {m}/{c}: no calibrator")
            continue
        path = None
        for p in paths:
            if Path(p).exists():
                path = p
                break
        if path is None:
            print(f"[skip] {m}/{c}: no eval parquet found")
            continue

        df = _ensure_margin_col(pd.read_parquet(path))
        print(f"  loaded {path}")
        df["calibrated_logit_margin"] = predict_margin_per_round(model_calibrators[m], df)
        df = add_answer_stable(df)

        rule_pq = per_question_rule(df, "answer_stable == 1 and calibrated_logit_margin > 0.25")
        cell_key = f"{m}_{c}"

        cell_out = {
            "n_questions": int(df.qid.nunique()),
            "AS_m25_f1": round(float(rule_pq["current_f1"].mean() * 100), 2),
            "AS_m25_calls": round(float(rule_pq["round"].mean()), 3),
            "vs_baselines": {},
        }

        print(f"\n{'='*60}")
        print(f"{cell_key}  (n={df.qid.nunique()})")
        print(f"  AS_m25: F1={cell_out['AS_m25_f1']:.2f}  calls={cell_out['AS_m25_calls']:.2f}")

        for k in [1, 3, 5]:
            base_pq = per_question_fixed_k(df, k)
            if len(base_pq) != len(rule_pq):
                print(f"  [warn] fixed_k={k} has {len(base_pq)} rows vs {len(rule_pq)} for AS_m25")
                continue
            res = paired_bootstrap(rule_pq, base_pq)
            cell_out["vs_baselines"][f"fixed_k={k}"] = res
            print(f"  vs k={k}: diff_F1={res['diff_f1_mean']:+.2f}  CI95={res['diff_f1_ci95']}  {res['verdict']}")

        results[cell_key] = cell_out

    if results:
        all_f1 = [v["AS_m25_f1"] for v in results.values()]
        all_calls = [v["AS_m25_calls"] for v in results.values()]
        print(f"\n{'='*60}")
        print(f"Macro ({len(results)} cells): F1={np.mean(all_f1):.2f}  calls={np.mean(all_calls):.2f}")

        # Macro bootstrap: paired difference across all cells vs k=3
        print("\n=== Macro paired bootstrap vs k=3 ===")
        all_ours_f1 = []
        all_base_f1 = []
        for v in results.values():
            if "fixed_k=3" in v["vs_baselines"]:
                all_ours_f1.append(v["vs_baselines"]["fixed_k=3"]["ours_f1"])
                all_base_f1.append(v["vs_baselines"]["fixed_k=3"]["base_f1"])
        if all_ours_f1:
            macro_diff = np.mean(all_ours_f1) - np.mean(all_base_f1)
            print(f"  Macro diff_F1 vs k=3: {macro_diff:+.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
