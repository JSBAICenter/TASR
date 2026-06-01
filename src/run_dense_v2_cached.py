"""Cached version of run_dense_v2 — skips FAISS index load when retrieval cache is hot.

Used when running multiple model experiments on the same corpus after one has
already populated the retrieval cache. Avoids loading the 61GB DKRR FAISS index
into RAM for each parallel model run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("IR_DATASETS_HOME", str((Path(__file__).resolve().parent.parent / "data" / "ir_datasets_cache").resolve()))
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loaders import LOADERS
from eval_wrapper import exact_match_score, f1_score
from run_dense_v2 import (
    MODEL_CONFIGS, DENSE_CONFIGS, MAX_ROUNDS, ANSWER_PROMPT,
    LLM_CACHE_DIR, RETRIEVAL_CACHE_DIR, _cache_key, _score_alt,
    call_with_logprobs, parse_answer_confidence, format_evidence,
    tokenize, jaccard,
)
import json


class CachedRetriever:
    """Reads retrieval results from cache only — fails loudly on cache miss."""

    def __init__(self, corpus: str, k: int = 50):
        cfg = DENSE_CONFIGS[corpus]
        self.dense_index = cfg["dense_index"]
        self.retriever_label = cfg["retriever_label"]
        self.k = k

    def retrieve(self, question: str) -> list[dict]:
        key = _cache_key(index=self.dense_index, q=question, k=self.k, retriever=self.retriever_label)
        cache_file = RETRIEVAL_CACHE_DIR / f"{key}.json"
        if not cache_file.exists():
            raise RuntimeError(f"Cache miss for question: {question[:80]!r}. Run the non-cached version first to populate cache.")
        with open(cache_file) as f:
            return json.load(f)


def run_cell(model_key: str, corpus: str, retriever, out_path: Path) -> pd.DataFrame:
    cfg = MODEL_CONFIGS[model_key]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    items = LOADERS[corpus]()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done_qids = set()
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            done_qids = set(existing["qid"].astype(str).unique())
            rows = existing.to_dict("records")
            print(f"  Resume: {len(done_qids)} questions done")
        except Exception:
            pass

    t0 = time.time()
    for q_idx, item in enumerate(tqdm(items, desc=f"{model_key}/{corpus}")):
        qid = str(item["id"])
        if qid in done_qids:
            continue
        alt = item["alt_answers"]
        qtype = item.get("type", "")

        ranked = retriever.retrieve(item["question"])
        top_score = ranked[0]["score"] if ranked else 0.0
        rank2_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
        gap = top_score - rank2_score

        evidence = []
        prior_tokens = set()
        for r in range(MAX_ROUNDS):
            if r < len(ranked):
                evidence.append(ranked[r])

            new_tokens = set()
            if r < len(ranked):
                new_tokens = tokenize(ranked[r]["title"] + " " + ranked[r]["text"])
            overlap = jaccard(new_tokens, prior_tokens)

            prompt = ANSWER_PROMPT.format(
                question=item["question"],
                evidence_text=format_evidence(evidence),
            )
            text, margins = call_with_logprobs(client, cfg["model"], prompt, cfg["no_thinking"])
            answer, conf = parse_answer_confidence(text)
            em, f1 = _score_alt(answer, alt)

            rows.append({
                "qid": qid, "question_type": qtype, "round": r + 1,
                "top_bm25_score": top_score, "rank2_bm25_score": rank2_score,
                "gap_1_2": gap, "jaccard_overlap": overlap,
                "llm_confidence": conf, "current_answer": answer,
                "current_em": em, "current_f1": f1,
                "first_token_margin": margins["first_token_margin"],
                "answer_token_margin": margins["answer_token_margin"],
                "first_token_str": margins["first_token_str"],
                "answer_token_str": margins["answer_token_str"],
                "n_tokens": margins["n_tokens"], "judge_entailment": np.nan,
            })
            prior_tokens.update(new_tokens)

        if (q_idx + 1) % 25 == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            elapsed = time.time() - t0
            print(f"  checkpoint @ q={q_idx+1}  elapsed={elapsed/60:.1f}m  rows={len(rows)}")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--corpus", required=True, choices=list(DENSE_CONFIGS.keys()))
    args = ap.parse_args()

    cfg = DENSE_CONFIGS[args.corpus]
    out = f"results/{args.model}_{cfg['retriever_label']}_{args.corpus}/signal_log_lp.parquet"
    print(f"=== Cached dense experiment: model={args.model}  corpus={args.corpus} ===")
    print(f"  endpoint: {MODEL_CONFIGS[args.model]['base_url']}")
    print(f"  output:   {out}")

    retriever = CachedRetriever(args.corpus)
    t0 = time.time()
    df = run_cell(args.model, args.corpus, retriever, Path(out))
    elapsed = time.time() - t0
    print(f"\nDone: {len(df)} rows in {elapsed/60:.1f}m")

    r5 = df[df["round"] == 5]
    r3 = df[df["round"] == 3]
    print(f"  fixed-k=5 F1: {r5['current_f1'].mean()*100:.2f}")
    print(f"  fixed-k=3 F1: {r3['current_f1'].mean()*100:.2f}")


if __name__ == "__main__":
    main()
