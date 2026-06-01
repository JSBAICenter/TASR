"""Multi-cell LLM-judge entailment runner.

Wraps `judge_signal.judge_split` over every (model, corpus, parquet) cell that
matters for the guardrail experiments: 6 distractor eval cells (HotpotQA +
2Wiki, each x 3 models) and 7 broadened eval cells (qwen x {fullwiki, NQ,
trivia}, devstral/gemma x {fullwiki, NQ}).

Same judge prompt and model (DEFAULT_MODEL = qwen) across all cells so
comparisons are fair. The judge sees only (question, evidence, candidate
answer); it does not know which agent produced the answer.

LLM calls are SHA-keyed cached in `data/llm_cache/` so reruns are free and
interruption is safe.

Writes `judge_entailment` (1-5) back into each parquet in place.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_signal import judge_split, report  # noqa: E402
from retrieval import build_paragraphs, retrieve  # noqa: E402


# -----------------------------------------------------------------------------
# Cell registry. (model, corpus) -> parquet to score + question-data source.
# Retriever flavor is implicit: "distractor" uses build_paragraphs+retrieve,
# "pyserini" uses PyseriniBM25 against the prebuilt index.
# -----------------------------------------------------------------------------

DISTRACTOR_CELLS = [
    # (model, corpus, parquet_path, data_json_path)
    ("qwen",     "hp",     "results/signal_log_lp_enriched.parquet",         "data/dev_eval.json"),
    ("qwen",     "2wiki",  "results/qwen_2wiki/signal_log_lp_enriched.parquet",     "data/2wiki_dev_eval.json"),
    ("devstral", "hp",     "results/devstral/signal_log_lp_enriched.parquet",       "data/dev_eval.json"),
    ("devstral", "2wiki",  "results/devstral_2wiki/signal_log_lp_enriched.parquet", "data/2wiki_dev_eval.json"),
    ("gemma",    "hp",     "results/gemma/signal_log_lp_enriched.parquet",          "data/dev_eval.json"),
    ("gemma",    "2wiki",  "results/gemma_2wiki/signal_log_lp_enriched.parquet",    "data/2wiki_dev_eval.json"),
]

BROADENED_CELLS = [
    # (model, corpus, parquet_path, loader_corpus_label)
    ("qwen",     "fullwiki", "results/qwen_fullwiki/signal_log_lp.parquet",     "fullwiki"),
    ("qwen",     "nq",       "results/qwen_nq/signal_log_lp.parquet",           "nq"),
    ("qwen",     "trivia",   "results/qwen_trivia/signal_log_lp.parquet",       "trivia"),
    ("devstral", "fullwiki", "results/devstral_fullwiki/signal_log_lp.parquet", "fullwiki"),
    ("devstral", "nq",       "results/devstral_nq/signal_log_lp.parquet",       "nq"),
    ("gemma",    "fullwiki", "results/gemma_fullwiki/signal_log_lp.parquet",    "fullwiki"),
    ("gemma",    "nq",       "results/gemma_nq/signal_log_lp.parquet",          "nq"),
]


def load_qdata_distractor(data_path: str) -> dict[str, dict]:
    with open(data_path) as f:
        items = json.load(f)
    out: dict[str, dict] = {}
    for item in items:
        paragraphs = build_paragraphs(item)
        ranked = retrieve(item["question"], paragraphs, k=len(paragraphs))
        out[item["id"]] = {"question": item["question"], "ranked": ranked}
    return out


def load_qdata_pyserini(corpus: str, qids_needed: set[str]) -> dict[str, dict]:
    """For broadened-eval cells, load 300-Q subsample and re-run the cached
    Pyserini retrieval for each. Cache hits are free."""
    from data_loaders import LOADERS
    from retrieval_pyserini import PyseriniBM25

    items = LOADERS[corpus]()
    items = [it for it in items if it["id"] in qids_needed]
    retriever = PyseriniBM25(corpus, k=50)
    out: dict[str, dict] = {}
    for item in items:
        ranked = retriever.retrieve(item["question"])
        out[item["id"]] = {"question": item["question"], "ranked": ranked}
    return out


def run_cell(model: str, corpus: str, parquet: str, qdata: dict, workers: int) -> None:
    p = Path(parquet)
    if not p.exists():
        print(f"[skip] {model}/{corpus}: {parquet} missing")
        return
    df = pd.read_parquet(p)
    if "judge_entailment" in df.columns and df["judge_entailment"].notna().all() and (df["judge_entailment"] > 0).all():
        print(f"[done] {model}/{corpus}: judge_entailment already populated ({len(df)} rows)")
        return

    label = f"{model}/{corpus}"
    t0 = time.time()
    df = judge_split(df, qdata, label, workers=workers)
    report(df, label)
    df.to_parquet(p, index=False)
    print(f"[wrote] {parquet}  ({df.shape})  in {(time.time()-t0)/60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="all", help="all|distractor|broadened|<comma-separated model/corpus>")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.cells == "all":
        plan = [(m, c, p, ("distractor", d)) for m, c, p, d in DISTRACTOR_CELLS] + \
               [(m, c, p, ("pyserini", lc)) for m, c, p, lc in BROADENED_CELLS]
    elif args.cells == "distractor":
        plan = [(m, c, p, ("distractor", d)) for m, c, p, d in DISTRACTOR_CELLS]
    elif args.cells == "broadened":
        plan = [(m, c, p, ("pyserini", lc)) for m, c, p, lc in BROADENED_CELLS]
    else:
        wanted = set(args.cells.split(","))
        plan = []
        for m, c, p, d in DISTRACTOR_CELLS:
            if f"{m}/{c}" in wanted:
                plan.append((m, c, p, ("distractor", d)))
        for m, c, p, lc in BROADENED_CELLS:
            if f"{m}/{c}" in wanted:
                plan.append((m, c, p, ("pyserini", lc)))

    print(f"Plan: {len(plan)} cells")
    for m, c, p, _ in plan:
        print(f"  {m}/{c}  -> {p}")
    print()

    for model, corpus, parquet, (kind, arg) in plan:
        print(f"\n{'=' * 70}\n{model}/{corpus}\n{'=' * 70}")
        df = pd.read_parquet(parquet)
        qids_needed = set(df["qid"].astype(str).unique())
        if kind == "distractor":
            qdata = load_qdata_distractor(arg)
        else:
            qdata = load_qdata_pyserini(arg, qids_needed)
        missing = qids_needed - set(qdata.keys())
        if missing:
            print(f"  [warn] {len(missing)} qids in parquet not found in qdata; will fall back to empty evidence")
            for q in missing:
                qdata[q] = {"question": "", "ranked": []}
        run_cell(model, corpus, parquet, qdata, args.workers)


if __name__ == "__main__":
    main()
