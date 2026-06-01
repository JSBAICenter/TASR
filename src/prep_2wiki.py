"""Reshape 2WikiMultiHopQA into our HotpotQA-style JSON schema.

Samples 400 examples from the 2Wiki validation split with seed 42 and
partitions them into 100 tune + 300 eval, matching our HotpotQA protocol.

Output schema (per example) matches data/dev_eval.json:
  id, question, answer, type, level, supporting_facts{title,sent_id}, context{title,sentences}

Writes:
  data/2wiki_dev_tune.json  (100 questions)
  data/2wiki_dev_eval.json  (300 questions)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset


def reshape(ex: dict) -> dict:
    # Coerce titles to str. 2Wiki occasionally yields integer-only titles (e.g.
    # postal-code or year articles loaded as int by the dataset parser); BM25
    # tokenization expects strings.
    titles = [str(c[0]) for c in ex["context"]]
    sentences = [[str(s) for s in c[1]] for c in ex["context"]]

    sf_titles, sf_sent_ids = [], []
    sf = ex.get("supporting_facts") or {}
    if isinstance(sf, dict) and "title" in sf and "sent_id" in sf:
        sf_titles = list(sf["title"])
        sf_sent_ids = [int(x) for x in sf["sent_id"]]
    elif isinstance(sf, list):
        for entry in sf:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                sf_titles.append(str(entry[0]))
                sf_sent_ids.append(int(entry[1]))

    return {
        "id": ex["_id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex["type"],
        "level": "hard",
        "supporting_facts": {"title": sf_titles, "sent_id": sf_sent_ids},
        "context": {"title": titles, "sentences": sentences},
    }


def main() -> None:
    ds = load_dataset("voidful/2WikiMultihopQA", split="validation")
    rng = random.Random(42)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    sample = idxs[:400]

    tune_idx = sample[:100]
    eval_idx = sample[100:400]

    tune = [reshape(ds[i]) for i in tune_idx]
    eval_ = [reshape(ds[i]) for i in eval_idx]

    Path("data/2wiki_dev_tune.json").write_text(json.dumps(tune, indent=2, ensure_ascii=False))
    Path("data/2wiki_dev_eval.json").write_text(json.dumps(eval_, indent=2, ensure_ascii=False))

    from collections import Counter
    print(f"wrote data/2wiki_dev_tune.json  n={len(tune)}  types={dict(Counter(x['type'] for x in tune))}")
    print(f"wrote data/2wiki_dev_eval.json  n={len(eval_)}  types={dict(Counter(x['type'] for x in eval_))}")
    print(f"context size (paragraphs): tune mean={sum(len(x['context']['title']) for x in tune)/len(tune):.1f}  "
          f"eval mean={sum(len(x['context']['title']) for x in eval_)/len(eval_):.1f}")


if __name__ == "__main__":
    main()
