"""End-to-end Devstral pipeline on 2WikiMultiHopQA (cross-dataset, full ablation).

Mirror of src/run_devstral_pipeline.py but with 2Wiki data paths and
results/devstral_2wiki/ + figures/devstral_2wiki/ namespaces. Stage 3 (the
expensive LP parquet) will skip because src/run_devstral_2wiki.py already
produced it; same for the existing per-type bootstrap JSON.

Run in background so it survives a disconnect:
    nohup venv/bin/python src/run_devstral_2wiki_full.py \\
        > logs/devstral_2wiki_full.log 2>&1 &
    disown

Resumable: every stage checks whether its output artifact exists and skips
if so.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEVSTRAL_ENV = {
    "LLM_BASE_URL": os.getenv("DEVSTRAL_BASE_URL", "https://your-devstral-endpoint.example/v1"),
    "LLM_MODEL": os.getenv("DEVSTRAL_MODEL", "mistralai/Devstral-Small-2-24B-Instruct-2512"),
    "LLM_API_KEY": os.getenv("DEVSTRAL_API_KEY", "REPLACE_ME"),
    "LLM_NO_THINKING_KWARG": "true",
}
for k, v in DEVSTRAL_ENV.items():
    os.environ[k] = v

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from eval_wrapper import evaluate  # noqa: E402
from run_baselines import run_closed_book  # noqa: E402
from run_instrumented import run_instrumented, summary as instrumented_summary  # noqa: E402


NS = Path("results/devstral_2wiki")
FIG_NS = Path("figures/devstral_2wiki")
NS.mkdir(parents=True, exist_ok=True)
FIG_NS.mkdir(parents=True, exist_ok=True)

DATA_TUNE = "data/2wiki_dev_tune.json"
DATA_EVAL = "data/2wiki_dev_eval.json"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_subprocess(label: str, cmd: list[str]) -> None:
    log(f"  $ {' '.join(cmd)}")
    env = {**os.environ}
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (rc={proc.returncode})")


def _normalize_answer(s: str) -> str:
    import re
    import string

    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _per_question_answer_stable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_norm"] = df["current_answer"].map(_normalize_answer)
    df["_prev"] = df.groupby("qid")["_norm"].shift(1)
    df["answer_stable"] = ((df["_prev"].notna()) & (df["_prev"] == df["_norm"])).astype(int)
    stops = df[df["answer_stable"] == 1].groupby("qid", sort=False).head(1)
    stopped = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def _per_question_fixed_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    sub = df[df["round"] == k][["qid", "current_em", "current_f1", "round"]].copy()
    return sub.sort_values("qid").reset_index(drop=True)


def stage_closed_book() -> None:
    log("STAGE 1: closed-book on both splits (2Wiki)")
    for split, data in [("dev_tune", DATA_TUNE), ("dev_eval", DATA_EVAL)]:
        out = NS / f"closed_book_{split}.json"
        if out.exists():
            log(f"  [skip] {out} exists")
            preds = json.load(open(out))
        else:
            preds = run_closed_book(data, str(out))
        result = evaluate(preds, data)
        log(f"  closed-book {split}: EM={result['em']:.1f}%  F1={result['f1']:.1f}%")


def stage_instrumented_plain() -> None:
    log("STAGE 2: instrumented k=5 plain on both splits (2Wiki)")
    pairs = [
        (DATA_TUNE, NS / "signal_log_tune.parquet"),
        (DATA_EVAL, NS / "signal_log.parquet"),
    ]
    for data, out in pairs:
        if out.exists():
            df = pd.read_parquet(out)
            log(f"  [skip] {out} exists ({len(df)} rows)")
        else:
            log(f"  starting {data} -> {out}")
            df = run_instrumented(data, str(out))
            log(f"  wrote {out}: {len(df)} rows")
        instrumented_summary(df, str(out))


def stage_instrumented_logprobs() -> None:
    log("STAGE 3: instrumented k=5 with logprobs on both splits (2Wiki)")
    from run_instrumented_with_logprobs import run_instrumented_lp  # noqa: E402
    pairs = [
        (DATA_TUNE, NS / "signal_log_lp_tune.parquet"),
        (DATA_EVAL, NS / "signal_log_lp.parquet"),
    ]
    for data, out in pairs:
        if out.exists():
            df = pd.read_parquet(out)
            log(f"  [skip] {out} exists ({len(df)} rows over {df['qid'].nunique()} qids)")
            continue
        log(f"  starting {data} -> {out}")
        df = run_instrumented_lp(data, str(out))
        log(f"  wrote {out}: {len(df)} rows")


def stage_enrich() -> None:
    log("STAGE 4: signal enrichment on plain parquet (2Wiki)")
    out_eval = NS / "signal_log_enriched.parquet"
    out_tune = NS / "signal_log_enriched_tune.parquet"
    if out_eval.exists() and out_tune.exists():
        log("  [skip] enriched plain parquets exist")
    else:
        run_subprocess("enrich plain", [
            sys.executable, "src/signals.py",
            "--eval", str(NS / "signal_log.parquet"),
            "--tune", str(NS / "signal_log_tune.parquet"),
            "--data-eval", DATA_EVAL,
            "--data-tune", DATA_TUNE,
            "--out-prefix", str(NS / "signal_log_enriched"),
        ])

    log("STAGE 5: signal enrichment on logprobs parquet (2Wiki)")
    out_lp_eval = NS / "signal_log_lp_enriched.parquet"
    out_lp_tune = NS / "signal_log_lp_enriched_tune.parquet"
    if out_lp_eval.exists() and out_lp_tune.exists():
        log("  [skip] enriched lp parquets exist")
    else:
        run_subprocess("enrich lp", [
            sys.executable, "src/enrich_lp.py",
            "--eval", str(NS / "signal_log_lp.parquet"),
            "--tune", str(NS / "signal_log_lp_tune.parquet"),
            "--data-eval", DATA_EVAL,
            "--data-tune", DATA_TUNE,
            "--out-prefix", str(NS / "signal_log_lp_enriched"),
        ])

    log("STAGE 6: v2 signals on lp parquet (2Wiki)")
    df_check = pd.read_parquet(out_lp_eval)
    if "overlap_signal_idf" in df_check.columns and "top_score_z_robust" in df_check.columns:
        log("  [skip] v2 signals already present")
    else:
        run_subprocess("enrich v2", [
            sys.executable, "src/enrich_v2_signals.py",
            "--eval", str(out_lp_eval),
            "--tune", str(out_lp_tune),
            "--data-eval", DATA_EVAL,
            "--data-tune", DATA_TUNE,
        ])


def stage_offline_eval() -> None:
    log("STAGE 7: headline `answer_stable == 1` rule + paired bootstrap CIs vs fixed-k")
    import numpy as np

    out_bootstrap = NS / "bootstrap_ci_answer_stable.json"
    if out_bootstrap.exists():
        log(f"  [skip] {out_bootstrap} exists (preserves earlier per-type breakdown)")
        return

    df = pd.read_parquet(NS / "signal_log_lp_enriched.parquet")
    pq_ours = _per_question_answer_stable(df)
    fixed = {k: _per_question_fixed_k(df, k) for k in [1, 2, 3, 5]}
    qids = pq_ours["qid"].values
    for k in fixed:
        assert (fixed[k]["qid"].values == qids).all()

    n_boot, seed = 1000, 42
    rng = np.random.default_rng(seed)
    n = len(qids)
    f1_o = pq_ours["current_f1"].values * 100.0
    em_o = pq_ours["current_em"].values * 100.0
    calls_o = pq_ours["round"].values.astype(float)

    out: dict = {
        "rule": "stop if answer_stable == 1",
        "data_source": str(NS / "signal_log_lp_enriched.parquet"),
        "n_questions": int(n),
        "ours_em_mean": float(em_o.mean()),
        "ours_f1_mean": float(f1_o.mean()),
        "ours_avg_calls": float(calls_o.mean()),
        "comparisons": {},
    }
    for k, pq_b in fixed.items():
        f1_b = pq_b["current_f1"].values * 100.0
        em_b = pq_b["current_em"].values * 100.0
        diff_f1 = np.zeros(n_boot); diff_em = np.zeros(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            diff_f1[i] = (f1_o[idx] - f1_b[idx]).mean()
            diff_em[i] = (em_o[idx] - em_b[idx]).mean()
        def ci(a):
            lo, hi = np.percentile(a, [2.5, 97.5])
            return [float(lo), float(hi)]
        out["comparisons"][f"fixed_k={k}"] = {
            "base_f1_mean": float(f1_b.mean()),
            "base_em_mean": float(em_b.mean()),
            "diff_f1_mean": float(diff_f1.mean()),
            "diff_f1_ci95": ci(diff_f1),
            "diff_em_mean": float(diff_em.mean()),
            "diff_em_ci95": ci(diff_em),
            "diff_calls_mean": float(calls_o.mean() - k),
        }
    out_bootstrap.write_text(json.dumps(out, indent=2))


def stage_offline_ablations() -> None:
    log("STAGE 8: 876-DNF formula ablation (2Wiki)")
    out_dnf_csv = NS / "subset_ablation_lp_dnf.csv"
    if not out_dnf_csv.exists():
        run_subprocess("DNF ablation", [
            sys.executable, "src/subset_ablation_lp_dnf.py",
            "--eval", str(NS / "signal_log_lp_enriched.parquet"),
            "--tune", str(NS / "signal_log_lp_enriched_tune.parquet"),
            "--out-csv", str(out_dnf_csv),
            "--out-json", str(NS / "subset_ablation_lp_dnf.json"),
        ])
    else:
        log(f"  [skip] {out_dnf_csv} exists")

    log("STAGE 9: round-aware r1 trigger grid (2Wiki)")
    out_r1_csv = NS / "round_aware_r1_grid.csv"
    if not out_r1_csv.exists():
        run_subprocess("round-aware r1", [
            sys.executable, "src/round_aware_r1_grid.py",
            "--eval", str(NS / "signal_log_lp_enriched.parquet"),
            "--tune", str(NS / "signal_log_lp_enriched_tune.parquet"),
            "--out-csv", str(out_r1_csv),
        ])
    else:
        log(f"  [skip] {out_r1_csv} exists")


def stage_figures_and_qualitative() -> None:
    log("STAGE 10: Pareto figure on 2Wiki + Devstral lp parquet")
    out_pdf = FIG_NS / "pareto.pdf"
    if not out_pdf.exists():
        run_subprocess("Pareto figure", [
            sys.executable, "src/headline_pareto_lp.py",
            "--eval", str(NS / "signal_log_lp_enriched.parquet"),
            "--out-pdf", str(out_pdf),
            "--out-png", str(FIG_NS / "pareto_headline.png"),
            "--out-json", str(NS / "pareto_data_lp.json"),
        ])
    else:
        log(f"  [skip] {out_pdf} exists")

    log("STAGE 11: qualitative dilution analysis on 2Wiki + Devstral lp parquet")
    out_summary = NS / "qualitative" / "dilution_summary.json"
    if not out_summary.exists():
        run_subprocess("qualitative dilution", [
            sys.executable, "src/qualitative_dilution.py",
            "--parquet", str(NS / "signal_log_lp_enriched.parquet"),
            "--eval-json", DATA_EVAL,
            "--out-dir", str(NS / "qualitative"),
        ])
    else:
        log(f"  [skip] {out_summary} exists")


def main() -> None:
    log("=" * 60)
    log("Devstral on 2WikiMultiHopQA (full pipeline)")
    log(f"  endpoint: {os.environ['LLM_BASE_URL']}")
    log(f"  model:    {os.environ['LLM_MODEL']}")
    log(f"  output:   {NS}/  and  {FIG_NS}/")
    log("=" * 60)

    stage_closed_book()
    stage_instrumented_plain()
    stage_instrumented_logprobs()
    stage_enrich()
    stage_offline_eval()
    stage_offline_ablations()
    stage_figures_and_qualitative()

    log("=" * 60)
    log("Pipeline complete.")
    log("=" * 60)


if __name__ == "__main__":
    main()
