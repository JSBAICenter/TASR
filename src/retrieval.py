import json
import random
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


_PUNCT_RE = re.compile(r"[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()


def build_paragraphs(item: dict) -> list[dict]:
    titles = item["context"]["title"]
    sentences = item["context"]["sentences"]
    return [
        {"title": title, "text": " ".join(sents)}
        for title, sents in zip(titles, sentences)
    ]


def retrieve(question: str, paragraphs: list[dict], k: int) -> list[dict]:
    tokenized_corpus = [tokenize(p["title"] + " " + p["text"]) for p in paragraphs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(
        zip(scores, paragraphs), key=lambda x: x[0], reverse=True
    )
    return [
        {"title": p["title"], "text": p["text"], "score": float(s)}
        for s, p in ranked[:k]
    ]


def retrieve_all(question: str, paragraphs: list[dict]) -> list[dict]:
    return retrieve(question, paragraphs, k=len(paragraphs))


if __name__ == "__main__":
    with open("data/dev_tune.json") as f:
        data = json.load(f)

    random.seed(0)
    sample = random.sample(data, 5)
    for item in sample:
        paragraphs = build_paragraphs(item)
        top3 = retrieve(item["question"], paragraphs, k=3)
        gold_titles = {f for f, _ in zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"])}

        print(f"\nQ: {item['question']}")
        print(f"Gold answer: {item['answer']}")
        print(f"Gold titles: {sorted(gold_titles)}")
        print("Top-3 retrieved:")
        for r in top3:
            mark = "*" if r["title"] in gold_titles else " "
            print(f"  {mark} [{r['score']:.2f}] {r['title']}")
