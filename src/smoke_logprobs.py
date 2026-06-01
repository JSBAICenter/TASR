"""Smoke test: extract logit margin from 20 dev_tune questions, 5 rounds each.

Verifies before the full ~2000-call run:
1. Endpoint reliably returns logprobs for answer-style prompts (not just '2+2').
2. Margin distribution spreads across questions (continuous, not collapsed).
3. The 'first token after Answer:' extraction works on real responses.
4. Margin correlates with current_em (sanity check that it tracks correctness).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentState, ANSWER_PROMPT, format_evidence, parse_answer_confidence  # noqa: E402
from eval_wrapper import exact_match_score, f1_score  # noqa: E402
from llm_client import client, DEFAULT_MODEL  # noqa: E402
from retrieval import build_paragraphs  # noqa: E402


N_QUESTIONS = 20
MAX_ROUNDS = 5
ANSWER_MARKER = "Answer:"


def extract_margins(top_logprobs_per_token: list, generated_text: str) -> dict:
    """Return margins at two positions: first generated token, and first token
    after the literal 'Answer:' marker in the generation.

    Each entry in top_logprobs_per_token has .token and .top_logprobs (list of
    objects with .token and .logprob).
    """
    out = {
        "first_token_margin": float("nan"),
        "answer_token_margin": float("nan"),
        "first_token_str": None,
        "answer_token_str": None,
        "answer_token_index": -1,
    }

    if not top_logprobs_per_token:
        return out

    # First-token margin (top1 - top2 logprob == top1 - top2 logit).
    first = top_logprobs_per_token[0]
    if len(first.top_logprobs) >= 2:
        out["first_token_margin"] = first.top_logprobs[0].logprob - first.top_logprobs[1].logprob
        out["first_token_str"] = first.top_logprobs[0].token

    # Find the token *after* 'Answer:' appears in the running concatenation.
    running = ""
    found_at = -1
    for i, tok in enumerate(top_logprobs_per_token):
        running += tok.token
        if ANSWER_MARKER in running and found_at == -1:
            # Look at the next token (i+1) — that's where the answer commitment lives.
            if i + 1 < len(top_logprobs_per_token):
                found_at = i + 1
                break
    if found_at >= 0:
        target = top_logprobs_per_token[found_at]
        # Skip leading whitespace tokens (very common: model emits " " then "Paris").
        while found_at < len(top_logprobs_per_token) and target.token.strip() == "":
            found_at += 1
            if found_at >= len(top_logprobs_per_token):
                break
            target = top_logprobs_per_token[found_at]
        if found_at < len(top_logprobs_per_token) and len(target.top_logprobs) >= 2:
            out["answer_token_margin"] = target.top_logprobs[0].logprob - target.top_logprobs[1].logprob
            out["answer_token_str"] = target.top_logprobs[0].token
            out["answer_token_index"] = found_at

    return out


def call_with_logprobs(prompt: str) -> tuple[str, dict]:
    """Direct API call (bypasses llm_client cache). Returns (text, margins)."""
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.0,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (response.choices[0].message.content or "").strip()
    lp = response.choices[0].logprobs
    tokens = lp.content if (lp is not None and lp.content) else []
    margins = extract_margins(tokens, text)
    margins["n_tokens"] = len(tokens)
    return text, margins


def main() -> None:
    with open("data/dev_tune.json") as f:
        data = json.load(f)
    questions = data[:N_QUESTIONS]
    print(f"Smoke test on {len(questions)} dev_tune questions, {MAX_ROUNDS} rounds each = {len(questions)*MAX_ROUNDS} calls\n")

    rows: list[dict] = []
    for item in tqdm(questions, desc="smoke"):
        paragraphs = build_paragraphs(item)
        state = AgentState(question=item["question"], paragraphs=paragraphs)
        gold = item["answer"]
        for _ in range(MAX_ROUNDS):
            state.search(n=1)
            prompt = ANSWER_PROMPT.format(
                question=state.question,
                evidence_text=format_evidence(state.evidence),
            )
            text, margins = call_with_logprobs(prompt)
            answer, conf = parse_answer_confidence(text)
            em = exact_match_score(answer, gold)
            f1_v, _, _ = f1_score(answer, gold)
            rows.append({
                "qid": item["id"],
                "round": state.round_num + 1,  # round_num is bumped inside answer(), we skipped it
                "answer": answer,
                "confidence": conf,
                "em": float(em),
                "f1": float(f1_v),
                **margins,
                "raw_text": text[:80],
            })
            state.round_num += 1  # keep numbering consistent
            state.answers.append(answer)
            state.confidences.append(conf)

    # Diagnostics.
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== margins distribution ===")
    for col in ["first_token_margin", "answer_token_margin"]:
        v = df[col].dropna().values
        if len(v) == 0:
            print(f"  {col}: all NaN")
            continue
        print(f"  {col}: n={len(v)}  min={v.min():.3f}  max={v.max():.3f}  "
              f"mean={v.mean():.3f}  std={v.std():.3f}")
        print(f"    percentiles 10/50/90: {np.percentile(v, 10):.2f} / {np.percentile(v, 50):.2f} / {np.percentile(v, 90):.2f}")

    print("\n=== NaN counts ===")
    print(f"  first_token_margin: {df['first_token_margin'].isna().sum()}/{len(df)}")
    print(f"  answer_token_margin: {df['answer_token_margin'].isna().sum()}/{len(df)}")

    print("\n=== correlation with EM ===")
    for col in ["first_token_margin", "answer_token_margin", "confidence"]:
        sub = df.dropna(subset=[col])
        if len(sub) < 2:
            print(f"  {col}: too few non-NaN rows")
            continue
        corr = np.corrcoef(sub[col].astype(float), sub["em"].astype(float))[0, 1]
        # Mean margin among correct vs incorrect.
        mean_correct = sub[sub["em"] == 1][col].astype(float).mean()
        mean_wrong = sub[sub["em"] == 0][col].astype(float).mean()
        print(f"  {col}: corr={corr:+.3f}  mean(em=1)={mean_correct:.2f}  mean(em=0)={mean_wrong:.2f}")

    print("\n=== sample rows (first 10) ===")
    print(df[["qid", "round", "answer", "em", "confidence", "first_token_str",
              "answer_token_str", "answer_token_margin"]].head(10).to_string(index=False))

    out = "results/smoke_logprobs.parquet"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
