"""Generalized dense retrieval experiment runner for TASR.

Supports multiple dense retriever families:
  - Contriever-MSMARCO (HotpotQA-fullwiki, NQ-Open)
  - DKRR-TQA (TriviaQA-Open, DPR-based, fine-tuned for TQA)

Each (model, corpus) combination produces a parquet with the same schema as
the BM25 runs, so replay_contriever.py and bootstrap_ci_opendomain.py work
unchanged.

Usage:
  python src/run_dense_v2.py --model qwen --corpus nq
  python src/run_dense_v2.py --model devstral --corpus trivia
  python src/run_dense_v2.py --model gemma --corpus nq
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
os.environ.setdefault("JVM_PATH", "/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so")
os.environ.setdefault("IR_DATASETS_HOME", str((Path(__file__).resolve().parent.parent / "data" / "ir_datasets_cache").resolve()))
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from openai import OpenAI
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder, DkrrDprQueryEncoder
from pyserini.search.lucene import LuceneSearcher

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loaders import LOADERS
from eval_wrapper import exact_match_score, f1_score

MAX_ROUNDS = 5

MODEL_CONFIGS = {
    "qwen": {
        "base_url": os.getenv("QWEN_BASE_URL", os.getenv("LLM_BASE_URL", "https://your-qwen-endpoint.example/v1")),
        "model": os.getenv("QWEN_MODEL", "qwen3.6-27b"),
        "api_key": os.getenv("QWEN_API_KEY", os.getenv("LLM_API_KEY", "REPLACE_ME")),
        "no_thinking": False,
    },
    "devstral": {
        "base_url": os.getenv("DEVSTRAL_BASE_URL", "https://your-devstral-endpoint.example/v1"),
        "model": os.getenv("DEVSTRAL_MODEL", "mistralai/Devstral-Small-2-24B-Instruct-2512"),
        "api_key": os.getenv("DEVSTRAL_API_KEY", "REPLACE_ME"),
        "no_thinking": True,
    },
    "gemma": {
        "base_url": os.getenv("GEMMA_BASE_URL", "https://your-gemma-endpoint.example/v1"),
        "model": os.getenv("GEMMA_MODEL", "google/gemma-4-31B-it"),
        "api_key": os.getenv("GEMMA_API_KEY", "REPLACE_ME"),
        "no_thinking": True,
    },
}

# Per-corpus dense retrieval config:
#   encoder: which query encoder to load
#   dense_index: prebuilt FAISS index name
#   sparse_index: prebuilt Lucene index name (for docstore text lookup)
#   retriever_label: short name used in output directory
DENSE_CONFIGS = {
    "fullwiki": {
        "encoder_type": "contriever",
        "dense_index": "beir-v1.0.0-hotpotqa.contriever-msmarco",
        "sparse_index": "beir-v1.0.0-hotpotqa.flat",
        "retriever_label": "contriever",
    },
    "nq": {
        "encoder_type": "contriever",
        "dense_index": "beir-v1.0.0-nq.contriever-msmarco",
        "sparse_index": "beir-v1.0.0-nq.flat",
        "retriever_label": "contriever",
    },
    "trivia": {
        "encoder_type": "dkrr",
        "dense_index": "wikipedia-dpr-100w.dkrr-tqa",
        "sparse_index": "wikipedia-dpr-100w",
        "retriever_label": "dkrr",
    },
}

ANSWER_PROMPT = """Answer the following question based on the provided evidence.

Question: {question}

Evidence:
{evidence_text}

Respond in this exact format:
Answer: <your short answer>
Confidence: <1-5>"""

LLM_CACHE_DIR = Path("data/dense_llm_cache")
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVAL_CACHE_DIR = Path("data/dense_retrieval_cache")
RETRIEVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(**kw) -> str:
    return hashlib.sha256(json.dumps(kw, sort_keys=True).encode()).hexdigest()


def _parse_doc_raw(raw: str) -> tuple[str, str]:
    try:
        d = json.loads(raw)
    except Exception:
        return "", raw.strip()
    title = (d.get("title") or "").strip()
    text = (d.get("text") or "").strip()
    if title or text:
        return title, text
    contents = (d.get("contents") or "").strip()
    if not contents:
        return "", ""
    first, _, body = contents.partition("\n")
    return first.strip().strip('"').strip(), body.strip()


class DenseRetriever:
    def __init__(self, corpus: str, k: int = 50):
        cfg = DENSE_CONFIGS[corpus]
        self.corpus = corpus
        self.dense_index = cfg["dense_index"]
        self.sparse_index = cfg["sparse_index"]
        self.retriever_label = cfg["retriever_label"]
        self.k = k

        print(f"  Loading encoder ({cfg['encoder_type']})...")
        if cfg["encoder_type"] == "contriever":
            self.encoder = AutoQueryEncoder("facebook/contriever-msmarco", pooling="mean")
        elif cfg["encoder_type"] == "dkrr":
            self.encoder = DkrrDprQueryEncoder("castorini/dkrr-dpr-tqa-retriever")
        else:
            raise ValueError(f"Unknown encoder type: {cfg['encoder_type']}")

        print(f"  Loading dense index: {self.dense_index}")
        self.searcher = FaissSearcher.from_prebuilt_index(self.dense_index, self.encoder)
        print(f"  Loading sparse docstore: {self.sparse_index}")
        self.docstore = LuceneSearcher.from_prebuilt_index(self.sparse_index)

    def retrieve(self, question: str) -> list[dict]:
        key = _cache_key(index=self.dense_index, q=question, k=self.k, retriever=self.retriever_label)
        cache_file = RETRIEVAL_CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        hits = self.searcher.search(question, self.k)
        out = []
        seen = set()
        for h in hits:
            doc = self.docstore.doc(h.docid)
            if doc is None or doc.raw() is None:
                continue
            title, text = _parse_doc_raw(doc.raw())
            display = title if title else f"doc_{h.docid}"
            if display in seen:
                continue
            seen.add(display)
            out.append({"title": display, "text": text, "score": float(h.score)})

        with open(cache_file, "w") as f:
            json.dump(out, f)
        return out


def _score_alt(pred: str, alt_answers: list[str]) -> tuple[float, float]:
    em = 0.0
    f1 = 0.0
    for g in alt_answers:
        em = max(em, float(exact_match_score(pred, g)))
        cur_f1, _, _ = f1_score(pred, g)
        f1 = max(f1, float(cur_f1))
    return em, f1


def call_with_logprobs(client: OpenAI, model_name: str, prompt: str, no_thinking: bool) -> tuple[str, dict]:
    key = _cache_key(prompt=prompt, model=model_name, temperature=0, retriever="dense_v2")
    cache_file = LLM_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        return cached["text"], cached["margins"]

    kwargs = dict(
        model=model_name,
        temperature=0,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=5,
    )
    if not no_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()

    margins = {"first_token_margin": np.nan, "answer_token_margin": np.nan,
               "first_token_str": "", "answer_token_str": "", "n_tokens": 0}

    if choice.logprobs and choice.logprobs.content:
        tokens = choice.logprobs.content
        margins["n_tokens"] = len(tokens)

        if tokens:
            t0 = tokens[0]
            top = sorted([lp.logprob for lp in t0.top_logprobs], reverse=True)
            margins["first_token_margin"] = top[0] - top[1] if len(top) > 1 else 0.0
            margins["first_token_str"] = t0.token

        ans_idx = None
        full = ""
        for i, t in enumerate(tokens):
            full += t.token
            if "answer:" in full.lower() and ans_idx is None:
                for j in range(i + 1, len(tokens)):
                    if tokens[j].token.strip():
                        ans_idx = j
                        break

        if ans_idx is not None and ans_idx < len(tokens):
            ta = tokens[ans_idx]
            top = sorted([lp.logprob for lp in ta.top_logprobs], reverse=True)
            margins["answer_token_margin"] = top[0] - top[1] if len(top) > 1 else 0.0
            margins["answer_token_str"] = ta.token

    with open(cache_file, "w") as f:
        json.dump({"text": text, "margins": margins, "prompt": prompt[:200]}, f)

    return text, margins


def parse_answer_confidence(text: str) -> tuple[str, int]:
    answer = ""
    conf = 3
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("answer:"):
            answer = line[len("answer:"):].strip()
        elif line.lower().startswith("confidence:"):
            try:
                conf = int(line[len("confidence:"):].strip())
                conf = max(1, min(5, conf))
            except ValueError:
                pass
    return answer, conf


def format_evidence(paragraphs: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {p['title']}: {p['text']}" for i, p in enumerate(paragraphs, 1))


def tokenize(text: str) -> set[str]:
    return set(re.findall(r'\w+', text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_cell(model_key: str, corpus: str, retriever: DenseRetriever, out_path: Path) -> pd.DataFrame:
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
                "qid": qid,
                "question_type": qtype,
                "round": r + 1,
                "top_bm25_score": top_score,
                "rank2_bm25_score": rank2_score,
                "gap_1_2": gap,
                "jaccard_overlap": overlap,
                "llm_confidence": conf,
                "current_answer": answer,
                "current_em": em,
                "current_f1": f1,
                "first_token_margin": margins["first_token_margin"],
                "answer_token_margin": margins["answer_token_margin"],
                "first_token_str": margins["first_token_str"],
                "answer_token_str": margins["answer_token_str"],
                "n_tokens": margins["n_tokens"],
                "judge_entailment": np.nan,
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
    print(f"=== Dense experiment: model={args.model}  corpus={args.corpus}  retriever={cfg['retriever_label']} ===")
    print(f"  endpoint: {MODEL_CONFIGS[args.model]['base_url']}")
    print(f"  model:    {MODEL_CONFIGS[args.model]['model']}")
    print(f"  output:   {out}")

    print(f"\nInitializing dense retriever...")
    retriever = DenseRetriever(args.corpus)

    print(f"\nRunning pipeline...")
    t0 = time.time()
    df = run_cell(args.model, args.corpus, retriever, Path(out))
    elapsed = time.time() - t0
    print(f"\nDone: {len(df)} rows in {elapsed/60:.1f}m")
    print(f"  margin range: [{df['answer_token_margin'].min():.2f}, {df['answer_token_margin'].max():.2f}]")

    r5 = df[df["round"] == 5]
    r3 = df[df["round"] == 3]
    print(f"\n  fixed-k=5 F1: {r5['current_f1'].mean()*100:.2f}")
    print(f"  fixed-k=3 F1: {r3['current_f1'].mean()*100:.2f}")


if __name__ == "__main__":
    main()
