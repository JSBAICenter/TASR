"""Replay AS_m25 (and baselines) on the broadened-eval parquets.

Per-model: fit the per-round isotonic on `answer_token_margin -> current_em` on
the model's own headline tune split (HotpotQA-distractor, 100 Qs), then apply
the locked calibration unchanged to each new corpus for that model. That is
the cleanest external-validity claim: same per-model calibration, same rule
thresholds, three new corpora.

Rules replayed:
  - AS         : answer_stable == 1
  - AS_m25     : answer_stable == 1 AND calibrated_logit_margin > 0.25  (headline)
  - AS_m20     : answer_stable == 1 AND calibrated_logit_margin > 0.20
  - fixed_k=1,2,3,5
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enrich_lp import fit_margin_per_round, predict_margin_per_round  # noqa: E402


MODEL_TUNE = {
    "qwen":     "results/signal_log_lp_tune.parquet",
    "devstral": "results/devstral/signal_log_lp_tune.parquet",
    "gemma":    "results/gemma/signal_log_lp_tune.parquet",
}

# (model, corpus) -> eval parquet. Trivia only for Qwen for now.
EVAL_PATHS = {
    ("qwen", "fullwiki"):     "results/qwen_fullwiki/signal_log_lp.parquet",
    ("qwen", "nq"):           "results/qwen_nq/signal_log_lp.parquet",
    ("qwen", "trivia"):       "results/qwen_trivia/signal_log_lp.parquet",
    ("devstral", "fullwiki"): "results/devstral_fullwiki/signal_log_lp.parquet",
    ("devstral", "nq"):       "results/devstral_nq/signal_log_lp.parquet",
    ("gemma", "fullwiki"):    "results/gemma_fullwiki/signal_log_lp.parquet",
    ("gemma", "nq"):          "results/gemma_nq/signal_log_lp.parquet",
}

RULES = {
    "AS":              "answer_stable == 1",
    "AS_m25":          "answer_stable == 1 and calibrated_logit_margin > 0.25",
    "AS_m20":          "answer_stable == 1 and calibrated_logit_margin > 0.20",
    # Guardrail family: stable + supporting-sentence overlap
    # and/or evidence entailment. `jaccard_overlap` here = passage-overlap with
    # prior rounds (proxy for "no new evidence"); `judge_entailment` = 1-5 LLM
    # judge scoring whether evidence supports the answer.
    "AS_ov":           "answer_stable == 1 and jaccard_overlap >= 0.15",
    "AS_en":           "answer_stable == 1 and judge_entailment >= 4",
    "AS_ov_or_en":     "answer_stable == 1 and (jaccard_overlap >= 0.15 or judge_entailment >= 4)",
    "AS_m25_ov_or_en": "answer_stable == 1 and calibrated_logit_margin > 0.25 and (jaccard_overlap >= 0.15 or judge_entailment >= 4)",
}


def _normalize_answer(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def add_answer_stable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_norm"] = df["current_answer"].map(_normalize_answer)
    df["_prev"] = df.groupby("qid")["_norm"].shift(1)
    df["answer_stable"] = ((df["_prev"].notna()) & (df["_prev"] == df["_norm"])).astype(int)
    return df.drop(columns=["_norm", "_prev"])


def per_question_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = df.eval(rule)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def per_question_fixed_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df[df["round"] == k][["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def summarize(pq: pd.DataFrame) -> dict:
    return {
        "f1":    round(float(pq["current_f1"].mean() * 100), 2),
        "em":    round(float(pq["current_em"].mean() * 100), 2),
        "calls": round(float(pq["round"].mean()), 3),
        "n":     int(len(pq)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/wiki_replay_table.json")
    args = ap.parse_args()

    # Fit per-model calibration once.
    model_models = {}
    for m, p in MODEL_TUNE.items():
        if not Path(p).exists():
            print(f"[skip] {m}: tune parquet {p} missing")
            continue
        df_t = pd.read_parquet(p)
        model_models[m] = fit_margin_per_round(df_t, "current_em")
        print(f"  {m}: fit on {len(df_t)} rows ({df_t.qid.nunique()} Qs) from {p}")
    print()

    table: dict[str, dict] = {}
    for (m, c), path in EVAL_PATHS.items():
        if m not in model_models:
            continue
        if not Path(path).exists():
            print(f"[skip] {m}/{c}: {path} missing")
            continue
        df = pd.read_parquet(path)
        df["calibrated_logit_margin"] = predict_margin_per_round(model_models[m], df)
        df = add_answer_stable(df)

        per = {
            "n_questions": int(df.qid.nunique()),
            "answer_token_margin_mean": round(float(df["answer_token_margin"].mean()), 3),
            "calibrated_logit_margin_mean": round(float(df["calibrated_logit_margin"].mean()), 3),
            "rules": {},
        }
        print(f"=== {m}/{c}  (n={df.qid.nunique()})  margin mean={df.answer_token_margin.mean():.2f}  calib mean={df.calibrated_logit_margin.mean():.3f}")
        for k in [1, 2, 3, 5]:
            pq = per_question_fixed_k(df, k)
            if len(pq) == 0:
                continue
            per["rules"][f"fixed_k={k}"] = summarize(pq)
            s = per["rules"][f"fixed_k={k}"]
            print(f"  fixed_k={k}:  F1 {s['f1']:6.2f}  EM {s['em']:6.2f}  calls {s['calls']:.2f}")
        for name, rule in RULES.items():
            pq = per_question_rule(df, rule)
            per["rules"][name] = summarize(pq)
            s = per["rules"][name]
            print(f"  {name:8s}:  F1 {s['f1']:6.2f}  EM {s['em']:6.2f}  calls {s['calls']:.2f}")
        print()
        table.setdefault(m, {})[c] = per

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
