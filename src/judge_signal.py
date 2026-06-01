"""Two-pass confidence: separate judge call scoring evidence -> answer entailment.

Instead of asking the model to rate its own answer
inline (confidence-first caused 5-7 F1 abstention loss), generate the answer
normally and then run a *separate* judge prompt that scores how well the
evidence supports the proposed answer.

Cost: one extra LLM call per (qid, round). 500 tune rows + 1500 eval rows.
Cache hits on re-runs. Writes `judge_entailment` (1-5) back to the enriched
parquets in place and reports Cohen's d.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import call_llm  # noqa: E402
from retrieval import build_paragraphs, retrieve  # noqa: E402


JUDGE_PROMPT = """You are evaluating whether a proposed answer is supported by the provided evidence.

Question: {question}
Evidence:
{evidence_text}
Proposed answer: {answer}

Rate how well the evidence supports this answer on a 1-5 scale:
  1 = evidence contradicts or is unrelated to the answer
  2 = evidence is weakly related but does not support the answer
  3 = partial / ambiguous support
  4 = evidence supports the answer with minor gaps
  5 = evidence clearly and fully supports the answer

Respond in exactly this format:
Support: <1-5>"""


def parse_support(response: str) -> int:
    m = re.search(r"Support:\s*(\d)", response)
    if m:
        return max(1, min(5, int(m.group(1))))
    m2 = re.search(r"\b([1-5])\b", response)
    return int(m2.group(1)) if m2 else 3


def format_evidence(passages: list[dict]) -> str:
    if not passages:
        return "(no evidence)"
    return "\n\n".join(f"[{p['title']}] {p['text']}" for p in passages)


def load_question_data(data_path: str) -> dict[str, dict]:
    with open(data_path) as f:
        items = json.load(f)
    out: dict[str, dict] = {}
    for item in items:
        paragraphs = build_paragraphs(item)
        ranked = retrieve(item["question"], paragraphs, k=len(paragraphs))
        out[item["id"]] = {"question": item["question"], "ranked": ranked}
    return out


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (a.std() + b.std()) / 2 + 1e-9
    return abs(a.mean() - b.mean()) / pooled


def _judge_one(args: tuple) -> tuple[int, int]:
    idx, prompt = args
    resp = call_llm(prompt, max_tokens=16)
    return idx, parse_support(resp)


def judge_split(df: pd.DataFrame, qdata: dict[str, dict], label: str, workers: int = 8) -> pd.DataFrame:
    n = len(df)
    scores = np.zeros(n, dtype=int)
    df = df.reset_index(drop=True)

    tasks = []
    for i, row in df.iterrows():
        meta = qdata[row["qid"]]
        evidence = meta["ranked"][: int(row["round"])]
        prompt = JUDGE_PROMPT.format(
            question=meta["question"],
            evidence_text=format_evidence(evidence),
            answer=row["current_answer"] or "",
        )
        tasks.append((i, prompt))

    t0 = time.time()
    done = 0
    print(f"[{label}] judging {n} rows with {workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_judge_one, t) for t in tasks]
        for fut in as_completed(futures):
            idx, score = fut.result()
            scores[idx] = score
            done += 1
            if done % 50 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n - done) / max(rate, 1e-6)
                print(f"  [{label}] {done}/{n}  ({rate:.2f} req/s, ETA {eta/60:.1f} min)", flush=True)
    df = df.copy()
    df["judge_entailment"] = scores
    return df


def report(df: pd.DataFrame, label: str) -> None:
    s = df["judge_entailment"].astype(float)
    print(f"\n[{label}] judge_entailment distribution: {dict(df['judge_entailment'].value_counts().sort_index())}")
    cor = df[df["current_em"] == 1]["judge_entailment"].astype(float)
    inc = df[df["current_em"] == 0]["judge_entailment"].astype(float)
    print(f"[{label}] Cohen's d (EM-correct vs incorrect): {cohens_d(cor, inc):.3f}")
    print(f"  mean(correct)   = {cor.mean():.3f}")
    print(f"  mean(incorrect) = {inc.mean():.3f}")
    print(f"[{label}] Per-round mean judge_entailment:")
    print(df.groupby("round")["judge_entailment"].mean().round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-parquet", default="results/signal_log_enriched.parquet")
    ap.add_argument("--tune-parquet", default="results/signal_log_enriched_tune.parquet")
    ap.add_argument("--data-eval", default="data/dev_eval.json")
    ap.add_argument("--data-tune", default="data/dev_tune.json")
    ap.add_argument("--splits", default="tune,eval", help="comma-separated subset of {tune,eval}")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    configs = {
        "tune": (args.tune_parquet, args.data_tune),
        "eval": (args.eval_parquet, args.data_eval),
    }

    for label in splits:
        parquet, data_path = configs[label]
        print(f"\n{'=' * 70}\nProcessing {label}: {parquet}\n{'=' * 70}")
        qdata = load_question_data(data_path)
        df = pd.read_parquet(parquet)
        df = judge_split(df, qdata, label, workers=args.workers)
        report(df, label)
        df.to_parquet(parquet, index=False)
        print(f"\nWrote {parquet}  ({df.shape})")


if __name__ == "__main__":
    main()
