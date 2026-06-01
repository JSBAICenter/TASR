"""End-to-end Devstral cross-model replication pipeline.

Re-runs every experiment we did on Qwen against a configured Devstral
endpoint. All output is namespaced under results/devstral/ and figures/devstral/
so Qwen artifacts are never overwritten. The LLM cache key already includes
the model name, so adding Devstral is purely additive on the cache.

Run in background so it survives a disconnect:
    nohup python src/run_devstral_pipeline.py \\
        > logs/devstral_pipeline.log 2>&1 &
    disown

The pipeline is resumable. Every stage checks whether its output artifact
exists and skips if so; the logprob runner has internal per-25-question
parquet checkpoints. To force a re-run of a stage, delete its output file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


# Force Devstral env BEFORE importing llm_client so background runs do not
# depend on the parent shell's state.
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


NS = Path("results/devstral")
FIG_NS = Path("figures/devstral")
NS.mkdir(parents=True, exist_ok=True)
FIG_NS.mkdir(parents=True, exist_ok=True)


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
    """Simulate `stop if answer_stable == 1` per question; fall back to last round."""
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


# ------------------------------------------------------------------- stages
def stage_closed_book() -> None:
    log("STAGE 1: closed-book on both splits")
    for split in ["dev_tune", "dev_eval"]:
        data = f"data/{split}.json"
        out = NS / f"closed_book_{split}.json"
        if out.exists():
            log(f"  [skip] {out} exists")
            preds = json.load(open(out))
        else:
            preds = run_closed_book(data, str(out))
        result = evaluate(preds, data)
        log(f"  closed-book {split}: EM={result['em']:.1f}%  F1={result['f1']:.1f}%")


def stage_instrumented_plain() -> None:
    log("STAGE 2: instrumented k=5 on both splits (no logprobs)")
    pairs = [
        ("data/dev_tune.json", NS / "signal_log_tune.parquet"),
        ("data/dev_eval.json", NS / "signal_log.parquet"),
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
    log("STAGE 3: instrumented k=5 with logprobs on both splits")
    # Lazy import: avoids loading sklearn at module import time.
    from run_instrumented_with_logprobs import run_instrumented_lp  # noqa: E402
    pairs = [
        ("data/dev_tune.json", NS / "signal_log_lp_tune.parquet"),
        ("data/dev_eval.json", NS / "signal_log_lp.parquet"),
    ]
    for data, out in pairs:
        if out.exists():
            df = pd.read_parquet(out)
            log(f"  [skip] {out} exists ({len(df)} rows over {df['qid'].nunique()} qids)")
            continue
        log(f"  starting {data} -> {out} (this is the heavy stage; expect hours)")
        df = run_instrumented_lp(data, str(out))
        log(f"  wrote {out}: {len(df)} rows")


def stage_enrich() -> None:
    log("STAGE 4: signal enrichment on plain parquet")
    out_eval = NS / "signal_log_enriched.parquet"
    out_tune = NS / "signal_log_enriched_tune.parquet"
    if out_eval.exists() and out_tune.exists():
        log(f"  [skip] enriched plain parquets exist")
    else:
        run_subprocess("enrich plain", [
            sys.executable, "src/signals.py",
            "--eval", str(NS / "signal_log.parquet"),
            "--tune", str(NS / "signal_log_tune.parquet"),
            "--data-eval", "data/dev_eval.json",
            "--data-tune", "data/dev_tune.json",
            "--out-prefix", str(NS / "signal_log_enriched"),
        ])

    log("STAGE 5: signal enrichment on logprobs parquet (adds calibrated_logit_margin)")
    out_lp_eval = NS / "signal_log_lp_enriched.parquet"
    out_lp_tune = NS / "signal_log_lp_enriched_tune.parquet"
    if out_lp_eval.exists() and out_lp_tune.exists():
        log(f"  [skip] enriched lp parquets exist")
    else:
        run_subprocess("enrich lp", [
            sys.executable, "src/enrich_lp.py",
            "--eval", str(NS / "signal_log_lp.parquet"),
            "--tune", str(NS / "signal_log_lp_tune.parquet"),
            "--data-eval", "data/dev_eval.json",
            "--data-tune", "data/dev_tune.json",
            "--out-prefix", str(NS / "signal_log_lp_enriched"),
        ])

    log("STAGE 6: v2 signals (IDF-Jaccard, robust z) on lp parquet")
    # enrich_v2_signals.py writes IN-PLACE — but only if columns are missing.
    df_check = pd.read_parquet(out_lp_eval)
    if "overlap_signal_idf" in df_check.columns and "top_score_z_robust" in df_check.columns:
        log("  [skip] v2 signals already present on devstral lp parquet")
    else:
        run_subprocess("enrich v2", [
            sys.executable, "src/enrich_v2_signals.py",
            "--eval", str(out_lp_eval),
            "--tune", str(out_lp_tune),
            "--data-eval", "data/dev_eval.json",
            "--data-tune", "data/dev_tune.json",
        ])


def stage_offline_eval() -> None:
    log("STAGE 7: headline `answer_stable == 1` rule + paired bootstrap CIs vs fixed-k")
    import numpy as np

    out_bootstrap = NS / "bootstrap_ci_answer_stable.json"
    if out_bootstrap.exists():
        log(f"  [skip] {out_bootstrap} exists")
        return

    df = pd.read_parquet(NS / "signal_log_lp_enriched.parquet")
    pq_ours = _per_question_answer_stable(df)
    fixed = {k: _per_question_fixed_k(df, k) for k in [1, 2, 3, 5]}

    qids = pq_ours["qid"].values
    for k in fixed:
        assert (fixed[k]["qid"].values == qids).all(), f"qid mismatch fixed_k={k}"

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
        "n_boot": n_boot,
        "seed": seed,
        "ours_em_mean": float(em_o.mean()),
        "ours_f1_mean": float(f1_o.mean()),
        "ours_avg_calls": float(calls_o.mean()),
        "comparisons": {},
    }
    log(f"  ours: F1={f1_o.mean():.2f}  EM={em_o.mean():.2f}  calls={calls_o.mean():.2f}  (n={n})")

    for k, pq_b in fixed.items():
        f1_b = pq_b["current_f1"].values * 100.0
        em_b = pq_b["current_em"].values * 100.0
        calls_b = np.full(n, float(k))

        diff_f1 = np.zeros(n_boot)
        diff_em = np.zeros(n_boot)
        diff_calls = np.zeros(n_boot)
        ours_f1_b = np.zeros(n_boot)
        base_f1_b = np.zeros(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            ours_f1_b[i] = f1_o[idx].mean()
            base_f1_b[i] = f1_b[idx].mean()
            diff_f1[i] = (f1_o[idx] - f1_b[idx]).mean()
            diff_em[i] = (em_o[idx] - em_b[idx]).mean()
            diff_calls[i] = (calls_o[idx] - calls_b[idx]).mean()

        def ci(a):
            lo, hi = np.percentile(a, [2.5, 97.5])
            return [float(lo), float(hi)]

        cmp = {
            "ours_f1_mean": float(f1_o.mean()),
            "ours_f1_ci95": ci(ours_f1_b),
            "base_f1_mean": float(f1_b.mean()),
            "base_f1_ci95": ci(base_f1_b),
            "diff_f1_mean": float(diff_f1.mean()),
            "diff_f1_ci95": ci(diff_f1),
            "diff_em_mean": float(diff_em.mean()),
            "diff_em_ci95": ci(diff_em),
            "diff_calls_mean": float(diff_calls.mean()),
            "diff_calls_ci95": ci(diff_calls),
            "p_ours_better_f1": float((diff_f1 > 0).mean()),
            "p_ours_worse_f1": float((diff_f1 < 0).mean()),
        }
        out["comparisons"][f"fixed_k={k}"] = cmp
        log(
            f"  vs fixed_k={k}:  ΔF1={cmp['diff_f1_mean']:+.2f} "
            f"[{cmp['diff_f1_ci95'][0]:+.2f}, {cmp['diff_f1_ci95'][1]:+.2f}]  "
            f"Δcalls={cmp['diff_calls_mean']:+.2f}"
        )

    out_bootstrap.write_text(json.dumps(out, indent=2))
    log(f"  wrote {out_bootstrap}")


def stage_offline_ablations() -> None:
    log("STAGE 8: OR / AND / mixed-DNF (876 formulas) ablation on lp parquet")
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

    log("STAGE 9: round-aware r1 trigger grid")
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
    log("STAGE 10: Pareto figure on Devstral lp parquet")
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

    log("STAGE 11: qualitative dilution analysis on Devstral lp parquet")
    out_summary = NS / "qualitative" / "dilution_summary.json"
    if not out_summary.exists():
        run_subprocess("qualitative dilution", [
            sys.executable, "src/qualitative_dilution.py",
            "--parquet", str(NS / "signal_log_lp_enriched.parquet"),
            "--eval-json", "data/dev_eval.json",
            "--out-dir", str(NS / "qualitative"),
        ])
    else:
        log(f"  [skip] {out_summary} exists")


def main() -> None:
    log("=" * 60)
    log("Devstral cross-model replication pipeline")
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
