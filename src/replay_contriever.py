"""Replay AS_m25 (and baselines) on Contriever experiment parquets.

Same logic as replay_wiki.py: fits per-round isotonic on the BM25-distractor
tune split (100 Qs), then applies the locked calibration unchanged to the
Contriever eval data. Then runs paired bootstrap CIs vs fixed-k=3.

Usage:
  python src/replay_contriever.py
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
    "qwen":     "results/signal_log_lp_tune.parquet",
    "devstral": f"{SHARED}/devstral/signal_log_lp_tune.parquet",
    "gemma":    f"{SHARED}/gemma/signal_log_lp_tune.parquet",
}

CONTRIEVER_CELLS = {
    ("qwen", "fullwiki"):     "results/qwen_contriever_fullwiki/signal_log_lp.parquet",
    ("devstral", "fullwiki"): "results/devstral_contriever_fullwiki/signal_log_lp.parquet",
    ("gemma", "fullwiki"):    "results/gemma_contriever_fullwiki/signal_log_lp.parquet",
    ("qwen", "nq"):           "results/qwen_contriever_nq/signal_log_lp.parquet",
    ("devstral", "nq"):       "results/devstral_contriever_nq/signal_log_lp.parquet",
    ("gemma", "nq"):          "results/gemma_contriever_nq/signal_log_lp.parquet",
    ("qwen", "trivia"):       "results/qwen_dkrr_trivia/signal_log_lp.parquet",
    ("devstral", "trivia"):   "results/devstral_dkrr_trivia/signal_log_lp.parquet",
    ("gemma", "trivia"):      "results/gemma_dkrr_trivia/signal_log_lp.parquet",
}

RULES = {
    "AS_m25": "answer_stable == 1 and calibrated_logit_margin > 0.25",
    "AS":     "answer_stable == 1",
}


def fit_margin_per_round(df_tune, label_col="current_em"):
    models = {}
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


def predict_margin_per_round(models, df):
    out = np.full(len(df), 0.5, dtype=float)
    for r, m in models.items():
        mask = df["round"].values == r
        if not mask.any():
            continue
        if isinstance(m, float):
            out[mask] = m
        else:
            x = df.loc[mask, "answer_token_margin"].values
            valid = ~np.isnan(x)
            preds = np.full(mask.sum(), 0.5)
            if valid.any():
                preds[valid] = m.predict(x[valid])
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
    chosen = pd.concat([stops, fallback])
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def per_question_fixed_k(df, k):
    return df[df["round"] == k][["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def paired_bootstrap(ours, base, n_boot=1000, seed=42):
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
    lo, hi = round(float(lo * 100), 3), round(float(hi * 100), 3)

    return {
        "ours_f1": round(float(o_f1.mean() * 100), 2),
        "base_f1": round(float(b_f1.mean() * 100), 2),
        "diff_f1_mean": round(float((o_f1 - b_f1).mean() * 100), 2),
        "diff_f1_ci95": [lo, hi],
        "ours_calls": round(float(o_r.mean()), 3),
        "verdict": "SIGNIFICANT WIN" if lo > 0 else ("SIGNIFICANT LOSS" if hi < 0 else "NOT SIGNIFICANT"),
    }


def main():
    # Fit calibrators
    calibrators = {}
    for m, p in MODEL_TUNE.items():
        if not Path(p).exists():
            print(f"[skip] {m}: {p} missing")
            continue
        df_t = pd.read_parquet(p)
        calibrators[m] = fit_margin_per_round(df_t)
        print(f"  {m}: fit calibrator on {df_t.qid.nunique()} Qs from {p}")

    results = {}
    for (m, c), path in sorted(CONTRIEVER_CELLS.items()):
        if m not in calibrators:
            continue
        if not Path(path).exists():
            print(f"[skip] {m}/{c}: {path} missing")
            continue

        df = pd.read_parquet(path)
        df["calibrated_logit_margin"] = predict_margin_per_round(calibrators[m], df)
        df = add_answer_stable(df)

        cell_key = f"{m}_{c}_contriever"
        print(f"\n{'='*60}")
        print(f"{cell_key}  (n={df.qid.nunique()})")

        cell_out = {"n_questions": int(df.qid.nunique()), "rules": {}, "vs_k3": {}}

        for k in [1, 3, 5]:
            pq = per_question_fixed_k(df, k)
            f1 = round(float(pq["current_f1"].mean() * 100), 2)
            cell_out["rules"][f"fixed_k={k}"] = {"f1": f1, "calls": k}
            print(f"  fixed_k={k}: F1={f1:.2f}")

        for name, rule in RULES.items():
            pq = per_question_rule(df, rule)
            f1 = round(float(pq["current_f1"].mean() * 100), 2)
            calls = round(float(pq["round"].mean()), 3)
            cell_out["rules"][name] = {"f1": f1, "calls": calls}
            print(f"  {name}: F1={f1:.2f}  calls={calls:.2f}")

            base_k3 = per_question_fixed_k(df, 3)
            if len(base_k3) == len(pq):
                bs = paired_bootstrap(pq, base_k3)
                cell_out["vs_k3"][name] = bs
                print(f"    vs k=3: diff_F1={bs['diff_f1_mean']:+.2f}  CI95={bs['diff_f1_ci95']}  {bs['verdict']}")

        results[cell_key] = cell_out

    out_path = Path("results/contriever_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Compare BM25 vs Contriever side by side
    bm25_path = Path("results/wiki_replay_table.json")
    if bm25_path.exists() and results:
        bm25 = json.loads(bm25_path.read_text())
        print(f"\n{'='*60}")
        print("BM25 vs Contriever comparison (AS_m25 on fullwiki):")
        print(f"{'Model':<12} {'BM25 F1':>10} {'Contriever F1':>15} {'BM25 calls':>12} {'Contriever calls':>18}")
        for m in ["qwen", "devstral", "gemma"]:
            ck = f"{m}_fullwiki_contriever"
            if ck in results and m in bm25 and "fullwiki" in bm25[m]:
                bm25_f1 = bm25[m]["fullwiki"]["rules"]["AS_m25"]["f1"]
                bm25_c = bm25[m]["fullwiki"]["rules"]["AS_m25"]["calls"]
                cont_f1 = results[ck]["rules"]["AS_m25"]["f1"]
                cont_c = results[ck]["rules"]["AS_m25"]["calls"]
                print(f"{m:<12} {bm25_f1:>10.2f} {cont_f1:>15.2f} {bm25_c:>12.3f} {cont_c:>18.3f}")


if __name__ == "__main__":
    main()
