"""Dense (Contriever) retriever for TASR external-validity experiments.

Drop-in replacement for retrieval_pyserini.py using Contriever embeddings
via Pyserini's FaissSearcher over prebuilt dense indexes.

Prebuilt Contriever indexes available in Pyserini:
  - wikipedia-dpr-100w.contriever-msmarco   (NQ, TriviaQA)
  - beir-v1.0.0-hotpotqa.contriever-msmarco (HotpotQA fullwiki)

Setup:
  pip install pyserini faiss-cpu   # or faiss-gpu
  # Indexes auto-download on first use via Pyserini.

Usage:
  python src/run_instrumented_wiki.py --corpus fullwiki --retriever contriever
  python src/run_instrumented_wiki.py --corpus nq --retriever contriever
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_JAVA_HOME = "/usr/lib/jvm/java-21-openjdk-amd64"
_DEFAULT_JVM_PATH = f"{_DEFAULT_JAVA_HOME}/lib/server/libjvm.so"
_DEFAULT_PYSERINI_CACHE = str(_REPO_ROOT / "data" / "pyserini_indexes")

os.environ.setdefault("JAVA_HOME", _DEFAULT_JAVA_HOME)
os.environ.setdefault("JVM_PATH", _DEFAULT_JVM_PATH)
os.environ.setdefault("PYSERINI_CACHE", _DEFAULT_PYSERINI_CACHE)
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder  # noqa: E402
from pyserini.search.lucene import LuceneSearcher  # noqa: E402

DENSE_INDEX_NAMES = {
    "fullwiki": "beir-v1.0.0-hotpotqa.contriever-msmarco",
    "nq":       "wikipedia-dpr-100w.contriever-msmarco",
    "trivia":   "wikipedia-dpr-100w.contriever-msmarco",
}

SPARSE_INDEX_NAMES = {
    "fullwiki": "beir-v1.0.0-hotpotqa.flat",
    "nq":       "wikipedia-dpr-100w",
    "trivia":   "wikipedia-dpr-100w",
}

CACHE_DIR = _REPO_ROOT / "data" / "contriever_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SEARCHER_CACHE: dict[str, "FaissSearcher"] = {}
_LUCENE_CACHE: dict[str, "LuceneSearcher"] = {}


def _get_searcher(index_name: str):
    if index_name not in _SEARCHER_CACHE:
        encoder = AutoQueryEncoder("facebook/contriever-msmarco", pooling="mean")
        _SEARCHER_CACHE[index_name] = FaissSearcher.from_prebuilt_index(index_name, encoder)
    return _SEARCHER_CACHE[index_name]


def _get_sparse_searcher(index_name: str):
    if index_name not in _LUCENE_CACHE:
        _LUCENE_CACHE[index_name] = LuceneSearcher.from_prebuilt_index(index_name)
    return _LUCENE_CACHE[index_name]


def _cache_key(index_name: str, question: str, k: int) -> str:
    raw = json.dumps({"index": index_name, "q": question, "k": k, "retriever": "contriever"}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


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
    title = first.strip().strip('"').strip()
    return title, body.strip()


class ContrieverRetriever:
    def __init__(self, corpus: str, k: int = 50):
        if corpus not in DENSE_INDEX_NAMES:
            raise ValueError(f"Unknown corpus '{corpus}'. Choices: {list(DENSE_INDEX_NAMES)}")
        self.corpus = corpus
        self.index_name = DENSE_INDEX_NAMES[corpus]
        self.sparse_index_name = SPARSE_INDEX_NAMES[corpus]
        self.k = k
        self.searcher = _get_searcher(self.index_name)
        self.sparse_searcher = _get_sparse_searcher(self.sparse_index_name)

    def retrieve(self, question: str) -> list[dict]:
        key = _cache_key(self.index_name, question, self.k)
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        hits = self.searcher.search(question, self.k)
        out: list[dict] = []
        seen_titles: set[str] = set()
        for h in hits:
            doc = self.sparse_searcher.doc(h.docid)
            if doc is None or doc.raw() is None:
                continue
            title, text = _parse_doc_raw(doc.raw())
            display_title = title if title else f"doc_{h.docid}"
            if display_title in seen_titles:
                continue
            seen_titles.add(display_title)
            out.append({
                "title": display_title,
                "text": text,
                "score": float(h.score),
            })

        with open(cache_file, "w") as f:
            json.dump(out, f)
        return out


if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "fullwiki"
    q = sys.argv[2] if len(sys.argv) > 2 else "Were Scott Derrickson and Ed Wood of the same nationality?"
    r = ContrieverRetriever(corpus, k=5)
    print(f"corpus={corpus}, index={r.index_name}")
    for p in r.retrieve(q):
        snippet = p["text"][:120].replace("\n", " ")
        print(f"  [{p['score']:.4f}] {p['title'][:40]}  {snippet}")
