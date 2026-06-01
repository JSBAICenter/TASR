"""Per-question agent state for the budget-aware retrieval method.

`search()` walks deeper into the BM25 ranking from the original question — no LLM
call. This is the headline behaviour for the distractor setting (round k = top-k
paragraphs as evidence).

`reformulate()` rewrites the query via the LLM and re-ranks the candidate pool.
Unused in the distractor headline; reserved for the fullwiki stretch where new
queries can surface different evidence.

`answer()` asks the LLM for (answer, confidence) over the current evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_client import call_llm
from retrieval import retrieve


ANSWER_PROMPT = """Based on the evidence below, answer the question with a short phrase (a few words only).
Then rate your confidence from 1 to 5:
  1 = pure guess, 2 = unlikely correct, 3 = maybe, 4 = fairly confident, 5 = certain

Question: {question}
Evidence:
{evidence_text}

Respond in exactly this format:
Answer: <your answer>
Confidence: <1-5>"""


REFORMULATE_PROMPT = """You are helping answer a question by searching for evidence. Generate a new, different
search query to find missing information.
Question: {question}
Evidence found so far: {evidence_text}
Previous queries: {previous_queries}
Generate ONE new search query (just the query, nothing else):"""


def parse_answer_confidence(response: str) -> tuple[str, int]:
    ans = re.search(r"Answer:\s*(.+)", response)
    conf = re.search(r"Confidence:\s*(\d)", response)
    answer = ans.group(1).strip() if ans else response.strip().split("\n")[0]
    confidence = int(conf.group(1)) if conf else 3
    return answer, max(1, min(5, confidence))


def format_evidence(passages: list[dict]) -> str:
    if not passages:
        return "(no evidence)"
    return "\n\n".join(f"[{p['title']}] {p['text']}" for p in passages)


@dataclass
class AgentState:
    question: str
    paragraphs: list[dict]
    retriever: object | None = None  # PyseriniBM25 instance for fullwiki/open-domain

    ranked: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    round_num: int = 0
    answers: list[str] = field(default_factory=list)
    confidences: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.retriever is not None:
            self.ranked = self.retriever.retrieve(self.question)
        else:
            self.ranked = retrieve(self.question, self.paragraphs, k=len(self.paragraphs))
        self.queries.append(self.question)

    def search(self, n: int = 1) -> list[dict]:
        seen = {p["title"] for p in self.evidence}
        added: list[dict] = []
        for p in self.ranked:
            if len(added) >= n:
                break
            if p["title"] in seen:
                continue
            self.evidence.append(p)
            added.append(p)
        return added

    def reformulate(self) -> str:
        prompt = REFORMULATE_PROMPT.format(
            question=self.question,
            evidence_text=format_evidence(self.evidence),
            previous_queries="; ".join(self.queries),
        )
        new_query = call_llm(prompt, max_tokens=64).strip()
        self.queries.append(new_query)
        self.ranked = retrieve(new_query, self.paragraphs, k=len(self.paragraphs))
        return new_query

    def answer(self) -> tuple[str, int]:
        self.round_num += 1
        prompt = ANSWER_PROMPT.format(
            question=self.question,
            evidence_text=format_evidence(self.evidence),
        )
        response = call_llm(prompt, max_tokens=128)
        ans, conf = parse_answer_confidence(response)
        self.answers.append(ans)
        self.confidences.append(conf)
        return ans, conf


if __name__ == "__main__":
    import json

    from retrieval import build_paragraphs

    with open("data/dev_tune.json") as f:
        data = json.load(f)
    item = data[0]
    paragraphs = build_paragraphs(item)
    state = AgentState(question=item["question"], paragraphs=paragraphs)
    print(f"Question: {state.question}")
    print(f"Gold: {item['answer']}")
    print()
    for _ in range(3):
        added = state.search(n=1)
        ans, conf = state.answer()
        title = added[0]["title"] if added else "(no new passage)"
        print(f"Round {state.round_num}: +[{title}] -> {ans} (conf={conf})")
