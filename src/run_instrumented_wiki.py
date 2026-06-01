"""Instrumented k=5 trace with logprobs over fullwiki / NQ-Open / TriviaQA-Open.

Parallel to [src/run_instrumented_with_logprobs.py](src/run_instrumented_with_logprobs.py)
but retrieves over a Pyserini Lucene index (via
[src/retrieval_pyserini.py](src/retrieval_pyserini.py)) instead of the
per-item 10-paragraph distractor pool. Question/answer pairs come from
[src/data_loaders.py](src/data_loaders.py).

Open-domain corpora (NQ/Trivia) ship multiple canonical answer strings per Q;
we score with the standard max-over-alternatives F1/EM convention.

Output: results/qwen_<corpus>/signal_log_lp.parquet with the same schema as
the distractor headline parquet (so the downstream isotonic + rule-replay
scripts work unchanged).

Usage:
  python src/run_instrumented_wiki.py --corpus fullwiki
  python src/run_instrumented_wiki.py --corpus nq
  python src/run_instrumented_wiki.py --corpus trivia
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentState, ANSWER_PROMPT, format_evidence, parse_answer_confidence  # noqa: E402
from data_loaders import LOADERS  # noqa: E402
from eval_wrapper import exact_match_score, f1_score  # noqa: E402
from retrieval import tokenize  # noqa: E402
from retrieval_pyserini import PyseriniBM25  # noqa: E402
from run_instrumented_with_logprobs import (  # noqa: E402
    MAX_ROUNDS,
    call_with_logprobs,
    jaccard,
)


def _score_alt(pred: str, alt_answers: list[str]) -> tuple[float, float]:
    """Max EM / max F1 across the alternative gold strings (NQ/Trivia convention)."""
    em = 0.0
    f1 = 0.0
    for g in alt_answers:
        em = max(em, float(exact_match_score(pred, g)))
        cur_f1, _, _ = f1_score(pred, g)
        f1 = max(f1, float(cur_f1))
    return em, f1


def run(corpus: str, output_path: str, k_retrieve: int = 50, n_checkpoint: int = 25) -> pd.DataFrame:
    items = LOADERS[corpus]()
    retriever = PyseriniBM25(corpus, k=k_retrieve)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done_qids: set[str] = set()
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            done_qids = set(existing["qid"].astype(str).unique())
            rows = existing.to_dict("records")
            print(f"Resume: loaded {len(rows)} rows ({len(done_qids)} questions done)")
        except Exception as e:
            print(f"Resume failed ({e}); starting fresh")
            rows = []
            done_qids = set()

    t0 = time.time()
    for q_idx, item in enumerate(tqdm(items, desc=f"lp {corpus}")):
        qid = str(item["id"])
        if qid in done_qids:
            continue
        alt = item["alt_answers"]
        qtype = item.get("type", "")

        state = AgentState(question=item["question"], paragraphs=[], retriever=retriever)
        top_score = state.ranked[0]["score"] if state.ranked else 0.0
        rank2_score = state.ranked[1]["score"] if len(state.ranked) > 1 else 0.0
        gap_1_2 = top_score - rank2_score

        prior_tokens: set[str] = set()
        for _ in range(MAX_ROUNDS):
            added = state.search(n=1)
            new_tokens: set[str] = set()
            for p in added:
                new_tokens.update(tokenize(p["title"] + " " + p["text"]))
            overlap = jaccard(new_tokens, prior_tokens)

            prompt = ANSWER_PROMPT.format(
                question=state.question,
                evidence_text=format_evidence(state.evidence),
            )
            text, margins = call_with_logprobs(prompt)
            answer, conf = parse_answer_confidence(text)
            state.round_num += 1
            state.answers.append(answer)
            state.confidences.append(conf)
            em, f1 = _score_alt(answer, alt)

            rows.append({
                "qid": qid,
                "question_type": qtype,
                "round": state.round_num,
                "top_bm25_score": top_score,
                "rank2_bm25_score": rank2_score,
                "gap_1_2": gap_1_2,
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
            })
            prior_tokens.update(new_tokens)

        if (q_idx + 1) % n_checkpoint == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            elapsed = time.time() - t0
            print(f"  checkpoint @ q={q_idx+1}  elapsed={elapsed/60:.1f} min  rows={len(rows)}")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--out", default=None,
                    help="parquet output path (default: results/qwen_<corpus>/signal_log_lp.parquet)")
    ap.add_argument("--k-retrieve", type=int, default=50)
    args = ap.parse_args()

    out = args.out or f"results/qwen_{args.corpus}/signal_log_lp.parquet"
    print(f"=== corpus={args.corpus}  out={out} ===")
    t0 = time.time()
    df = run(args.corpus, out, k_retrieve=args.k_retrieve)
    elapsed = time.time() - t0
    print(f"\nwrote {out}: {len(df)} rows in {elapsed/60:.1f} min")
    for c in ["first_token_margin", "answer_token_margin"]:
        print(f"  {c}: NaN {df[c].isna().sum()}/{len(df)}")
    print(f"answer_token_margin: min={df['answer_token_margin'].min():.2f} "
          f"max={df['answer_token_margin'].max():.2f} "
          f"mean={df['answer_token_margin'].mean():.2f}")


if __name__ == "__main__":
    main()
