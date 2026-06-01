"""Re-run the instrumented k=5 trace with logprobs=True to capture logit margins.

Produces signal_log_lp{,_tune}.parquet with the same columns as the original
instrumented parquets plus three margin columns:
  - first_token_margin     : logit margin at the very first generated token
                             (always 'Answer' under our prompt; useless but
                             logged for ablation)
  - answer_token_margin    : logit margin at the first non-whitespace token
                             AFTER 'Answer:' (the model's commitment point)
  - margin_token_str       : the actual top-1 token string at the answer position
                             (sanity check during analysis)

Uses its own on-disk cache (data/lp_cache/) keyed by call parameters; the
shared llm_cache only stores text and would drop the logprobs. Writes a parquet
checkpoint every N_CHECKPOINT questions so a partial run is recoverable.

Usage:
  python src/run_instrumented_with_logprobs.py --split tune     # ~3.5 hr
  python src/run_instrumented_with_logprobs.py --split eval     # ~10.5 hr
  python src/run_instrumented_with_logprobs.py --split both     # ~14 hr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentState, ANSWER_PROMPT, format_evidence, parse_answer_confidence  # noqa: E402
from eval_wrapper import exact_match_score, f1_score  # noqa: E402
from llm_client import client, DEFAULT_MODEL, NO_THINKING_KWARG  # noqa: E402
from retrieval import build_paragraphs, tokenize  # noqa: E402


MAX_ROUNDS = 5
ANSWER_MARKER = "Answer:"
N_CHECKPOINT = 25  # write parquet every N questions

LP_CACHE_DIR = Path("data/lp_cache")
LP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _lp_cache_key(prompt: str, model: str, temperature: float, max_tokens: int,
                  top_logprobs: int, thinking: bool) -> str:
    raw = json.dumps(
        {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_logprobs": top_logprobs,
            "thinking": thinking,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def extract_margins(top_logprobs_per_token: list) -> dict:
    """Returns first-token margin, answer-token margin, and tokens for debugging.

    answer_token_margin is the margin at the first non-whitespace token AFTER
    the literal 'Answer:' substring appears in the running concatenation.
    """
    out = {
        "first_token_margin": float("nan"),
        "answer_token_margin": float("nan"),
        "first_token_str": None,
        "answer_token_str": None,
        "answer_token_index": -1,
        "n_tokens": len(top_logprobs_per_token),
    }
    if not top_logprobs_per_token:
        return out

    first = top_logprobs_per_token[0]
    if len(first.top_logprobs) >= 2:
        out["first_token_margin"] = first.top_logprobs[0].logprob - first.top_logprobs[1].logprob
        out["first_token_str"] = first.top_logprobs[0].token

    running = ""
    answer_seen_at = -1
    for i, tok in enumerate(top_logprobs_per_token):
        running += tok.token
        if ANSWER_MARKER in running and answer_seen_at == -1:
            answer_seen_at = i + 1
            break

    if 0 <= answer_seen_at < len(top_logprobs_per_token):
        # Skip whitespace tokens.
        idx = answer_seen_at
        while idx < len(top_logprobs_per_token) and top_logprobs_per_token[idx].token.strip() == "":
            idx += 1
        if idx < len(top_logprobs_per_token):
            target = top_logprobs_per_token[idx]
            if len(target.top_logprobs) >= 2:
                out["answer_token_margin"] = target.top_logprobs[0].logprob - target.top_logprobs[1].logprob
                out["answer_token_str"] = target.top_logprobs[0].token
                out["answer_token_index"] = idx
    return out


def call_with_logprobs(prompt: str, retries: int = 3, backoff: float = 2.0) -> tuple[str, dict]:
    """Direct OpenAI client call with logprobs=True. Retries on transient errors.

    Caches (text, margins) on disk so a re-run is byte-identical even though the
    GPU inference path isn't bit-deterministic across calls.
    """
    model = DEFAULT_MODEL
    temperature = 0.0
    max_tokens = 128
    top_logprobs = 5
    thinking = False
    key = _lp_cache_key(prompt, model, temperature, max_tokens, top_logprobs, thinking)
    cache_file = LP_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        return cached["text"], cached["margins"]

    last_err = None
    for attempt in range(retries):
        try:
            req_kwargs = dict(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            if not NO_THINKING_KWARG:
                req_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
            response = client.chat.completions.create(**req_kwargs)
            text = (response.choices[0].message.content or "").strip()
            lp = response.choices[0].logprobs
            tokens = lp.content if (lp is not None and lp.content) else []
            margins = extract_margins(tokens)
            with open(cache_file, "w") as f:
                json.dump({"prompt": prompt, "text": text, "margins": margins}, f)
            return text, margins
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"API call failed after {retries} retries: {last_err}")


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def run_instrumented_lp(data_path: str, output_path: str) -> pd.DataFrame:
    with open(data_path) as f:
        data = json.load(f)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    # Resume support: if a checkpoint exists, skip questions already done.
    done_qids: set[str] = set()
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            done_qids = set(existing["qid"].unique())
            rows = existing.to_dict("records")
            print(f"Resume: loaded {len(rows)} rows ({len(done_qids)} questions done)")
        except Exception as e:
            print(f"Resume failed ({e}); starting fresh")
            rows = []
            done_qids = set()

    t0 = time.time()
    for q_idx, item in enumerate(tqdm(data, desc=f"lp {Path(data_path).stem}")):
        if item["id"] in done_qids:
            continue

        qid = item["id"]
        gold = item["answer"]
        qtype = item.get("type", "")
        paragraphs = build_paragraphs(item)
        state = AgentState(question=item["question"], paragraphs=paragraphs)
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
            em = exact_match_score(answer, gold)
            f1, _, _ = f1_score(answer, gold)

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
                "current_em": float(em),
                "current_f1": float(f1),
                "first_token_margin": margins["first_token_margin"],
                "answer_token_margin": margins["answer_token_margin"],
                "first_token_str": margins["first_token_str"],
                "answer_token_str": margins["answer_token_str"],
                "n_tokens": margins["n_tokens"],
            })

            prior_tokens.update(new_tokens)

        # Checkpoint.
        if (q_idx + 1) % N_CHECKPOINT == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            elapsed = time.time() - t0
            done = q_idx + 1 - len([d for d in done_qids if d in {it["id"] for it in data[: q_idx + 1]}])
            print(f"  checkpoint @ q={q_idx+1}  elapsed={elapsed/60:.1f} min  rows={len(rows)}")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "eval", "both"], default="tune")
    ap.add_argument("--out-tune", default="results/signal_log_lp_tune.parquet")
    ap.add_argument("--out-eval", default="results/signal_log_lp.parquet")
    args = ap.parse_args()

    plan = []
    if args.split in {"tune", "both"}:
        plan.append(("data/dev_tune.json", args.out_tune))
    if args.split in {"eval", "both"}:
        plan.append(("data/dev_eval.json", args.out_eval))

    for data_path, out_path in plan:
        print(f"\n=== {data_path} -> {out_path} ===")
        t0 = time.time()
        df = run_instrumented_lp(data_path, out_path)
        elapsed = time.time() - t0

        print(f"\nwrote {out_path}: {len(df)} rows in {elapsed/60:.1f} min")
        print(f"NaN per margin column:")
        for c in ["first_token_margin", "answer_token_margin"]:
            print(f"  {c}: {df[c].isna().sum()}/{len(df)}")
        print(f"answer_token_margin: min={df['answer_token_margin'].min():.2f} "
              f"max={df['answer_token_margin'].max():.2f} mean={df['answer_token_margin'].mean():.2f}")


if __name__ == "__main__":
    main()
