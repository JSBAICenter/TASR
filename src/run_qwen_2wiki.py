"""Run the Qwen instrumented-lp trace on 2WikiMultiHopQA (within-model cross-dataset).

Mirror of src/run_devstral_2wiki.py without the env override: relies on the
default `.env` Qwen config so the existing Qwen disk cache is reused.
Outputs are namespaced under results/qwen_2wiki/. After the trace finishes,
computes the headline `answer_stable == 1` rule with paired-bootstrap CIs vs
fixed-k baselines, plus a per-question-type breakdown.

Run in background so it survives a disconnect:
    nohup python src/run_qwen_2wiki.py > logs/qwen_2wiki.log 2>&1 &
    disown
"""
from __future__ import annotations

import json
import os
import re
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from run_instrumented_with_logprobs import run_instrumented_lp  # noqa: E402

NS = Path("results/qwen_2wiki")
NS.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def simulate_rule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_norm"] = df["current_answer"].map(normalize)
    df["_prev"] = df.groupby("qid")["_norm"].shift(1)
    df["answer_stable"] = ((df["_prev"].notna()) & (df["_prev"] == df["_norm"])).astype(int)
    stops = df[df["answer_stable"] == 1].groupby("qid", sort=False).head(1)
    stopped = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped)].groupby("qid", sort=False).tail(1)
    return pd.concat([stops, fallback]).sort_values("qid").reset_index(drop=True)


def fixed_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df[df["round"] == k].sort_values("qid").reset_index(drop=True)


def paired_boot(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(a)
    out = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[i] = (a[idx] - b[idx]).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(out.mean()), [float(lo), float(hi)]


def headline_report(df: pd.DataFrame, label: str) -> dict:
    pq_ours = simulate_rule(df)
    fixed = {k: fixed_k(df, k) for k in [1, 2, 3, 5]}
    qids = pq_ours["qid"].values
    for k, pq_b in fixed.items():
        assert (pq_b["qid"].values == qids).all(), f"qid mismatch k={k}"

    f1_o = pq_ours["current_f1"].values * 100
    em_o = pq_ours["current_em"].values * 100
    calls_o = pq_ours["round"].values.astype(float)

    log(f"  [{label}] n={len(qids)}  rule F1={f1_o.mean():.2f}  EM={em_o.mean():.2f}  calls={calls_o.mean():.2f}")

    comparisons = {}
    for k, pq_b in fixed.items():
        f1_b = pq_b["current_f1"].values * 100
        em_b = pq_b["current_em"].values * 100
        d_f1, ci_f1 = paired_boot(f1_o, f1_b)
        d_em, ci_em = paired_boot(em_o, em_b)
        comparisons[f"fixed_k={k}"] = {
            "base_f1_mean": float(f1_b.mean()),
            "base_em_mean": float(em_b.mean()),
            "diff_f1_mean": d_f1,
            "diff_f1_ci95": ci_f1,
            "diff_em_mean": d_em,
            "diff_em_ci95": ci_em,
            "diff_calls_mean": float(calls_o.mean() - k),
        }
        log(f"    vs fixed_k={k}:  ΔF1={d_f1:+.2f} [{ci_f1[0]:+.2f}, {ci_f1[1]:+.2f}]  "
            f"ΔEM={d_em:+.2f}  Δcalls={float(calls_o.mean() - k):+.2f}")
    return {
        "label": label,
        "n_questions": int(len(qids)),
        "ours_f1_mean": float(f1_o.mean()),
        "ours_em_mean": float(em_o.mean()),
        "ours_avg_calls": float(calls_o.mean()),
        "comparisons": comparisons,
    }


def main() -> None:
    log("=" * 60)
    log("Qwen on 2WikiMultiHopQA (within-model cross-dataset transfer)")
    log(f"  endpoint: {os.environ.get('LLM_BASE_URL', '<from .env>')}")
    log(f"  model:    {os.environ.get('LLM_MODEL', '<from .env>')}")
    log(f"  output:   {NS}/")
    log("=" * 60)

    pairs = [
        ("data/2wiki_dev_tune.json", NS / "signal_log_lp_tune.parquet"),
        ("data/2wiki_dev_eval.json", NS / "signal_log_lp.parquet"),
    ]
    for data, out in pairs:
        if out.exists():
            df = pd.read_parquet(out)
            log(f"[skip] {out} exists ({len(df)} rows, {df['qid'].nunique()} qids)")
            continue
        log(f"starting {data} -> {out}")
        df = run_instrumented_lp(data, str(out))
        log(f"wrote {out}: {len(df)} rows")

    log("=" * 60)
    log("Headline on 2Wiki dev_eval (Qwen)")
    log("=" * 60)
    df = pd.read_parquet(NS / "signal_log_lp.parquet")
    headline = headline_report(df, "2Wiki dev_eval (overall)")

    log("\nPer-question-type breakdown:")
    by_type = {}
    for qtype, sub in df.groupby("question_type"):
        if sub["qid"].nunique() < 5:
            continue
        by_type[qtype] = headline_report(sub, f"2Wiki {qtype}")

    out_json = NS / "bootstrap_ci_answer_stable.json"
    out_json.write_text(json.dumps({"overall": headline, "by_type": by_type}, indent=2))
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
