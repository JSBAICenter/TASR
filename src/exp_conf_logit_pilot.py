"""Pilot: confidence-digit logit margin vs answer-token logit margin.

Hypothesis: the calibrated_logit_margin we currently use looks at the model's
answer-token. The model also emits a 1-5 confidence digit. If it says "5" but
only 30% of the probability mass at that position is on "5" (the rest spread
across 1-4), that's not really confident. Maybe the confidence-digit logit
margin is a stronger correctness signal than the answer-token margin.

Pilot scope: 50 qids x 5 rounds = 250 calls on Qwen3.6-27B (HotpotQA dev_eval,
first 50). top_logprobs=20 so digits 0-5 are always present in the top set.

Captured signals (per row):
  answer_token_margin            -- existing (top1 - top2 logprob at answer pos)
  conf_top1_minus_top2           -- new (top1 - top2 logprob at conf-digit pos,
                                         over ALL tokens, not just digits)
  conf_top1_minus_2nd_digit      -- new (top1 - next-highest digit in {0..5})
  conf_chosen_prob               -- new (prob of the chosen digit, after softmax
                                         over top-20)
  conf_entropy_digits            -- new (Shannon entropy over digits 0-5, after
                                         renormalizing their probs)
  digit_logprobs                 -- dict of {0..5} -> logprob (or None)

Output: results/conf_logit_pilot.parquet
Report: correlations of each signal with current_em.
"""

from __future__ import annotations
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from agent import AgentState, ANSWER_PROMPT, format_evidence, parse_answer_confidence  # noqa: E402
from eval_wrapper import exact_match_score, f1_score  # noqa: E402
from llm_client import client, DEFAULT_MODEL, NO_THINKING_KWARG  # noqa: E402
from retrieval import build_paragraphs, tokenize  # noqa: E402


N_QIDS = 50
MAX_ROUNDS = 5
ANSWER_MARKER = "Answer:"
CONF_MARKER = "Confidence:"
TOP_LOGPROBS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 128
DIGITS = ['0', '1', '2', '3', '4', '5']

CACHE_DIR = ROOT / 'data' / 'lp_cache_pilot_conf'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_key(prompt: str) -> str:
    raw = json.dumps({
        'prompt': prompt, 'model': DEFAULT_MODEL,
        'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS,
        'top_logprobs': TOP_LOGPROBS, 'thinking': False,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def call_llm(prompt: str, retries: int = 3) -> dict:
    key = cache_key(prompt)
    fp = CACHE_DIR / f"{key}.json"
    if fp.exists():
        return json.loads(fp.read_text())

    last_err = None
    for attempt in range(retries):
        try:
            req = dict(
                model=DEFAULT_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                messages=[{'role': 'user', 'content': prompt}],
                logprobs=True, top_logprobs=TOP_LOGPROBS,
            )
            if not NO_THINKING_KWARG:
                req['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
            resp = client.chat.completions.create(**req)
            text = (resp.choices[0].message.content or '').strip()
            lp = resp.choices[0].logprobs
            tokens = lp.content if (lp is not None and lp.content) else []
            # Serialize: list of {token, logprob, top_logprobs: [{token, logprob}, ...]}
            ser = []
            for t in tokens:
                ser.append({
                    'token': t.token,
                    'logprob': t.logprob,
                    'top_logprobs': [{'token': c.token, 'logprob': c.logprob}
                                     for c in (t.top_logprobs or [])],
                })
            out = {'text': text, 'tokens': ser}
            fp.write_text(json.dumps(out))
            return out
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"call failed: {last_err}")


def find_first_nonspace_after_marker(tokens: list[dict], marker: str) -> int:
    running = ""
    seen_at = -1
    for i, t in enumerate(tokens):
        running += t['token']
        if marker in running and seen_at == -1:
            seen_at = i + 1
            break
    if seen_at < 0:
        return -1
    idx = seen_at
    while idx < len(tokens) and tokens[idx]['token'].strip() == "":
        idx += 1
    return idx if idx < len(tokens) else -1


def extract_signals(tokens: list[dict]) -> dict:
    out = {
        'answer_token_margin':       float('nan'),
        'answer_token_str':          None,
        'conf_token_str':            None,
        'conf_top1_minus_top2':      float('nan'),
        'conf_top1_minus_2nd_digit': float('nan'),
        'conf_chosen_prob':          float('nan'),
        'conf_entropy_digits':       float('nan'),
        'digit_logprobs':            json.dumps({d: None for d in DIGITS}),
        'n_tokens':                  len(tokens),
    }
    if not tokens:
        return out

    # Answer-token margin (existing).
    ai = find_first_nonspace_after_marker(tokens, ANSWER_MARKER)
    if 0 <= ai < len(tokens) and len(tokens[ai]['top_logprobs']) >= 2:
        a = tokens[ai]
        out['answer_token_margin'] = a['top_logprobs'][0]['logprob'] - a['top_logprobs'][1]['logprob']
        out['answer_token_str'] = a['top_logprobs'][0]['token']

    # Confidence-digit signals (new).
    ci = find_first_nonspace_after_marker(tokens, CONF_MARKER)
    if 0 <= ci < len(tokens):
        c = tokens[ci]
        out['conf_token_str'] = c['top_logprobs'][0]['token'] if c['top_logprobs'] else c['token']

        tl = c['top_logprobs']
        if len(tl) >= 2:
            out['conf_top1_minus_top2'] = tl[0]['logprob'] - tl[1]['logprob']

        # Restrict to digits 0-5.
        digit_lp = {d: None for d in DIGITS}
        for cand in tl:
            tok = cand['token'].strip()
            if tok in DIGITS and digit_lp[tok] is None:
                digit_lp[tok] = cand['logprob']
        out['digit_logprobs'] = json.dumps(digit_lp)

        chosen = out['conf_token_str'].strip() if out['conf_token_str'] else None
        chosen_lp = digit_lp.get(chosen) if chosen in DIGITS else None

        digit_vals = sorted((v for v in digit_lp.values() if v is not None), reverse=True)
        if chosen in DIGITS and chosen_lp is not None and len(digit_vals) >= 2:
            # next-highest digit logprob != chosen
            for v in digit_vals:
                if v != chosen_lp:
                    out['conf_top1_minus_2nd_digit'] = chosen_lp - v
                    break

        # chosen_prob: softmax over top-20 then read chosen
        if tl and chosen_lp is not None:
            lps = np.array([cand['logprob'] for cand in tl])
            mx = lps.max()
            exps = np.exp(lps - mx)
            probs = exps / exps.sum()
            # find which top_logprob entry is the chosen digit (first match)
            for j, cand in enumerate(tl):
                if cand['token'].strip() == chosen and cand['logprob'] == chosen_lp:
                    out['conf_chosen_prob'] = float(probs[j])
                    break

        # entropy over digits (renormalized)
        present_lps = [v for v in digit_lp.values() if v is not None]
        if len(present_lps) >= 2:
            lps = np.array(present_lps)
            mx = lps.max()
            exps = np.exp(lps - mx)
            p = exps / exps.sum()
            ent = -float((p * np.log(p + 1e-12)).sum())
            out['conf_entropy_digits'] = ent

    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


def run_pilot():
    data_path = ROOT / 'data' / 'dev_eval.json'
    with open(data_path) as f:
        data = json.load(f)
    items = data[:N_QIDS]

    rows = []
    t0 = time.time()
    for item in tqdm(items, desc='conf-pilot'):
        qid = item['id']
        gold = item['answer']
        paragraphs = build_paragraphs(item)
        state = AgentState(question=item['question'], paragraphs=paragraphs)
        prior_tokens: set[str] = set()
        for _ in range(MAX_ROUNDS):
            added = state.search(n=1)
            new_tokens: set[str] = set()
            for p in added:
                new_tokens.update(tokenize(p['title'] + ' ' + p['text']))
            overlap = jaccard(new_tokens, prior_tokens)

            prompt = ANSWER_PROMPT.format(
                question=state.question,
                evidence_text=format_evidence(state.evidence),
            )
            resp = call_llm(prompt)
            text = resp['text']
            signals = extract_signals(resp['tokens'])
            answer, conf = parse_answer_confidence(text)
            state.round_num += 1
            state.answers.append(answer)
            state.confidences.append(conf)
            em = exact_match_score(answer, gold)
            f1, _, _ = f1_score(answer, gold)

            rows.append({
                'qid': qid,
                'round': state.round_num,
                'jaccard_overlap': overlap,
                'llm_confidence': conf,
                'current_answer': answer,
                'current_em': float(em),
                'current_f1': float(f1),
                **signals,
            })
            prior_tokens.update(new_tokens)

    df = pd.DataFrame(rows)
    out_p = ROOT / 'results' / 'conf_logit_pilot.parquet'
    df.to_parquet(out_p, index=False)
    elapsed = time.time() - t0
    print(f"\nWrote {out_p}: {len(df)} rows in {elapsed/60:.1f} min\n")
    return df


def report(df: pd.DataFrame):
    print("=" * 70)
    print(f"Pilot: {df['qid'].nunique()} qids x ~{int(len(df)/df['qid'].nunique())} rounds = {len(df)} rows")
    print(f"Overall EM rate: {df['current_em'].mean():.3f}")
    print("\nCorrelation of each signal with current_em (Pearson, Spearman):")
    print(f"{'signal':30s} | {'pearson':>8s} | {'spearman':>9s} | {'n_valid':>8s}")
    print('-' * 70)
    signals = [
        'answer_token_margin',
        'conf_top1_minus_top2',
        'conf_top1_minus_2nd_digit',
        'conf_chosen_prob',
        'conf_entropy_digits',
        'llm_confidence',
    ]
    for s in signals:
        v = df[[s, 'current_em']].dropna()
        if len(v) == 0:
            print(f"{s:30s} | {'nan':>8s} | {'nan':>9s} | {'0':>8s}")
            continue
        p = v[s].corr(v['current_em'])
        sp = v[s].corr(v['current_em'], method='spearman')
        # For entropy: higher entropy = less confident, so we expect NEGATIVE corr
        print(f"{s:30s} | {p:+8.3f} | {sp:+9.3f} | {len(v):>8d}")

    print("\nPer-round breakdown of answer_token_margin vs conf_top1_minus_2nd_digit:")
    for r in sorted(df['round'].unique()):
        sub = df[df['round'] == r]
        em = sub['current_em'].mean()
        a = sub['answer_token_margin'].mean()
        c = sub['conf_top1_minus_2nd_digit'].mean()
        cp = sub['conf_chosen_prob'].mean()
        print(f"  round {r}: n={len(sub):3d}  EM={em:.3f}  "
              f"ans_margin={a:.2f}  conf_t1-t2d={c:.2f}  conf_chosen_prob={cp:.3f}")

    print("\nllm_confidence distribution (raw 1-5):")
    for c in sorted(df['llm_confidence'].dropna().unique()):
        sub = df[df['llm_confidence'] == c]
        print(f"  conf={int(c)}: n={len(sub):3d}, EM={sub['current_em'].mean():.3f}, "
              f"mean chosen_prob={sub['conf_chosen_prob'].mean():.3f}")

    # Save report
    out_j = ROOT / 'results' / 'conf_logit_pilot_report.json'
    summary = {
        'n_qids': int(df['qid'].nunique()),
        'n_rows': int(len(df)),
        'em_rate': float(df['current_em'].mean()),
        'correlations_pearson': {},
        'correlations_spearman': {},
    }
    for s in signals:
        v = df[[s, 'current_em']].dropna()
        if len(v) > 0:
            summary['correlations_pearson'][s] = float(v[s].corr(v['current_em']))
            summary['correlations_spearman'][s] = float(v[s].corr(v['current_em'], method='spearman'))
    out_j.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_j}")


def main():
    df = run_pilot()
    report(df)


if __name__ == '__main__':
    main()
