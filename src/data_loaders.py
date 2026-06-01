"""Question/answer loaders for the broadened-eval corpora.

Returns items in the same shape consumed by [src/agent.py](src/agent.py) and
[src/run_instrumented_with_logprobs.py](src/run_instrumented_with_logprobs.py),
namely a list of dicts with at least `id`, `question`, `answer`, `type`,
`level`. `context`/`supporting_facts` are absent here — retrieval happens over
the Pyserini index, not from a per-item paragraph pool.

For NQ-Open and TriviaQA-Open the dataset ships *multiple* canonical short
answers per question (e.g. `("Linda Davis",)` for NQ, `("Helicopters",
"Helicopter") `for Trivia). Our F1/EM scorer in eval_wrapper takes a single
gold string, so we pick the first answer as `answer` and stash the full
alternatives list under `alt_answers` for later token-level scoring if we
want to be more generous.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

# ir_datasets pulls in the BEIR/DPR text files. Keep the cache repo-local by
# default so the package can be moved without editing user-specific paths.
os.environ.setdefault("IR_DATASETS_HOME", str((Path(__file__).resolve().parent.parent / "data" / "ir_datasets_cache").resolve()))

import ir_datasets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUBSAMPLE_N = 300
SUBSAMPLE_SEED = 42


def _subsample(items: list[dict], n: int = SUBSAMPLE_N, seed: int = SUBSAMPLE_SEED) -> list[dict]:
    if len(items) <= n:
        return items
    rng = random.Random(seed)
    return rng.sample(items, n)


def load_hotpotqa_fullwiki(split: str = "eval") -> list[dict]:
    """HotpotQA dev set (300 Qs we already have), retrieved against fullwiki."""
    path = ROOT / "data" / f"dev_{split}.json"
    with open(path) as f:
        data = json.load(f)
    out = []
    for item in data:
        out.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "alt_answers": [item["answer"]],
            "type": item.get("type", ""),
            "level": item.get("level", ""),
            "supporting_facts": item.get("supporting_facts"),
        })
    return out


def _load_dpr_w100_split(ir_name: str) -> list[dict]:
    d = ir_datasets.load(ir_name)
    out: list[dict] = []
    for q in d.queries_iter():
        # DPR queries are namedtuples (query_id, text, answers).
        alt = list(q.answers)
        # Filter out emoji-only or empty answer strings; pick first non-empty.
        clean = [a for a in alt if a and a.strip() and any(ch.isalnum() for ch in a)]
        if not clean:
            continue
        out.append({
            "id": str(q.query_id),
            "question": q.text,
            "answer": clean[0],
            "alt_answers": clean,
            "type": "single",
            "level": "open",
            "supporting_facts": None,
        })
    return out


def load_nq_open() -> list[dict]:
    """NQ-Open dev (DPR split): 6,515 Qs with short answer strings."""
    items = _load_dpr_w100_split("dpr-w100/natural-questions/dev")
    return _subsample(items)


def load_trivia_open() -> list[dict]:
    """TriviaQA-Open dev (DPR split): 8,837 Qs with short answer strings."""
    items = _load_dpr_w100_split("dpr-w100/trivia-qa/dev")
    return _subsample(items)


LOADERS = {
    "fullwiki": load_hotpotqa_fullwiki,
    "nq": load_nq_open,
    "trivia": load_trivia_open,
}


if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "fullwiki"
    items = LOADERS[corpus]()
    print(f"{corpus}: {len(items)} items")
    for it in items[:3]:
        print(f"  qid={it['id']} q={it['question'][:80]!r} ans={it['answer']!r} "
              f"alt={it['alt_answers'][:3]}")
