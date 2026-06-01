"""Pyserini-backed BM25 retriever for the broadened-eval (wiki) experiments.

Replaces the per-call rank_bm25 build in [src/retrieval.py](src/retrieval.py)
with a single load of a prebuilt Lucene BM25 index. Used by the fullwiki +
BEIR pipeline; the distractor pipeline keeps using rank_bm25.

Returns paragraphs in the same `{title, text, score}` shape that the rest of
the agent expects, so [src/agent.py](src/agent.py)'s `search(n=1)` walk over
`state.ranked` works unchanged.

Per-query results are cached to JSON on disk so reruns are free.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Set Pyserini's JVM / cache / OpenAI shim envs before importing pyserini.
# Doing it here means callers don't have to remember the env dance.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_JAVA_HOME = "/usr/lib/jvm/java-21-openjdk-amd64"
_DEFAULT_JVM_PATH = f"{_DEFAULT_JAVA_HOME}/lib/server/libjvm.so"
_DEFAULT_PYSERINI_CACHE = str(_REPO_ROOT / "data" / "pyserini_indexes")

os.environ.setdefault("JAVA_HOME", _DEFAULT_JAVA_HOME)
os.environ.setdefault("JVM_PATH", _DEFAULT_JVM_PATH)
os.environ.setdefault("PYSERINI_CACHE", _DEFAULT_PYSERINI_CACHE)
# Pyserini 0.24 eagerly instantiates an OpenAI client at import time; we don't
# use any encoder paths, so a stub key keeps the import from blowing up.
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from pyserini.search.lucene import LuceneSearcher  # noqa: E402

# Mapping from logical corpus name to Pyserini prebuilt index name.
# `nq` and `trivia` share `wikipedia-dpr-100w` (DPR's 21M-passage Wikipedia 100w
# chunks), matching the corpus used by NQ-Open / TriviaQA-Open in the DPR paper.
INDEX_NAMES = {
    "fullwiki": "beir-v1.0.0-hotpotqa.flat",   # HotpotQA Wikipedia abstracts
    "nq":       "wikipedia-dpr-100w",          # NQ-Open / DPR Wikipedia
    "trivia":   "wikipedia-dpr-100w",          # TriviaQA-Open / same corpus
}

CACHE_DIR = _REPO_ROOT / "data" / "pyserini_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# nq + trivia both point at the same index_name; only load the JVM-side searcher once.
_SEARCHER_CACHE: dict[str, "LuceneSearcher"] = {}


def _get_searcher(index_name: str):
    if index_name not in _SEARCHER_CACHE:
        _SEARCHER_CACHE[index_name] = LuceneSearcher.from_prebuilt_index(index_name)
    return _SEARCHER_CACHE[index_name]


def _cache_key(index_name: str, question: str, k: int) -> str:
    raw = json.dumps({"index": index_name, "q": question, "k": k}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _parse_doc_raw(raw: str) -> tuple[str, str]:
    """Pyserini docs come in two flavors here:

    BEIR (used by beir-v1.0.0-hotpotqa.flat): JSON with `_id`, `title`, `text`.
    DPR (used by wikipedia-dpr-100w): JSON with `id`, `contents`, where contents
    is `"\"Title\"\nbody text...` (title quoted on first line, body after).
    """
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
    # DPR titles are wrapped in literal double quotes; strip them.
    title = first.strip().strip('"').strip()
    return title, body.strip()


class PyseriniBM25:
    def __init__(self, corpus: str, k: int = 50):
        if corpus not in INDEX_NAMES:
            raise ValueError(f"Unknown corpus '{corpus}'. Choices: {list(INDEX_NAMES)}")
        self.corpus = corpus
        self.index_name = INDEX_NAMES[corpus]
        self.k = k
        self.searcher = _get_searcher(self.index_name)

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
            doc = self.searcher.doc(h.docid)
            if doc is None or doc.raw() is None:
                continue
            title, text = _parse_doc_raw(doc.raw())
            # If title is empty (FiQA), fall back to docid so dedup-by-title in
            # AgentState.search() still works.
            display_title = title if title else f"doc_{h.docid}"
            # Skip exact-title duplicates so search(n=1) makes progress.
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
    corpus = sys.argv[1] if len(sys.argv) > 1 else "fiqa"
    q = sys.argv[2] if len(sys.argv) > 2 else "what is a hedge fund"
    r = PyseriniBM25(corpus, k=5)
    print(f"corpus={corpus}, index={r.index_name}")
    for p in r.retrieve(q):
        snippet = p["text"][:120].replace("\n", " ")
        print(f"  [{p['score']:.2f}] {p['title'][:40]}  {snippet}")
