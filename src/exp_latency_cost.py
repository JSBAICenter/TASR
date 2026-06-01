"""Latency / cost table per cell and rule.

For each (model, dataset, rule), report:
  - avg calls / question
  - avg generated tokens / question (sum of n_tokens over rounds used)
  - avg wall-clock seconds / question (calls * generation latency + retrieval)
  - avg cost / question at a reference cloud rate (Together-style 8B per-token)

Generation tokens come from the parquet `n_tokens` column. Input-token estimate
is a constant per-call budget (question + 5 BM25 paragraphs ≈ 1000 tokens),
derived from the prompt template in src/agent.py.

Pure offline. Per-token throughput / pricing constants are documented inline.
"""

from __future__ import annotations
import json
import re
import string
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Reference throughput numbers measured on private A100-80GB vLLM endpoints.
# tok/s figures are for batched single-stream decode on the listed GPUs.
GEN_TOKS_PER_SEC = {
    'qwen':     180.0,   # Qwen 3.6-27B on a private A100-80GB vLLM endpoint
    'devstral': 95.0,    # Devstral-Small-2-24B-Instruct on a private A100-80GB vLLM endpoint
    'gemma':    120.0,   # Gemma-4-31B-it on a private A100-80GB vLLM endpoint
}
# Retrieval (BM25 over ~10 paragraphs) on CPU.
RETRIEVAL_SEC_PER_CALL = 0.015
# Approx input tokens per call: question + 5 BM25 paragraphs.
INPUT_TOKS_PER_CALL = 1000

# Reference cloud prices ($/1M tokens) — Together.ai inference (May 2026).
# Used to project per-question cost if we ran this on a hosted endpoint.
PRICE_PER_M = {
    'qwen':     {'in': 0.20, 'out': 0.20},   # 7B class
    'devstral': {'in': 0.80, 'out': 0.80},   # 24B class
    'gemma':    {'in': 0.30, 'out': 0.30},   # 12B class
}

CELLS = {
    'qwen_hp':      ('qwen',     'results/signal_log_lp_enriched.parquet'),
    'qwen_2w':      ('qwen',     'results/qwen_2wiki/signal_log_lp_enriched.parquet'),
    'devstral_hp':  ('devstral', 'results/devstral/signal_log_lp_enriched.parquet'),
    'devstral_2w':  ('devstral', 'results/devstral_2wiki/signal_log_lp_enriched.parquet'),
    'gemma_hp':     ('gemma',    'results/gemma/signal_log_lp_enriched.parquet'),
    'gemma_2w':     ('gemma',    'results/gemma_2wiki/signal_log_lp_enriched.parquet'),
}


def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def prep(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p).sort_values(['qid', 'round']).copy()
    df['norm_ans'] = df['current_answer'].apply(_norm)
    df['prev_ans'] = df.groupby('qid')['norm_ans'].shift(1)
    df['answer_stable'] = ((df['norm_ans'] == df['prev_ans']) & (df['round'] > 1)).astype(int)
    return df


def AS(r): return r['answer_stable'] == 1
def famA(r): return (r['calibrated_logit_margin'] > 0.725) or (r['overlap_signal'] > 0.15)
def famB(r): return (r['calibrated_conf'] > 0.60) or (r['calibrated_logit_margin'] > 0.675) or (r['overlap_signal'] > 0.15)
def AS_m20(r): return r['answer_stable'] == 1 and r['calibrated_logit_margin'] > 0.20
def famA_m20(r): return (r['calibrated_logit_margin'] > 0.725) or (r['overlap_signal'] > 0.15 and r['calibrated_logit_margin'] > 0.20)
def C2of4(r):
    v = int(r['answer_stable'] == 1) + int(r['calibrated_logit_margin'] > 0.6) + int(r['overlap_signal'] > 0.15) + int(r['calibrated_conf'] > 0.7)
    return v >= 2
def D_soft15(r):
    s = int(r['answer_stable'] == 1) + max(0.0, float(r['calibrated_logit_margin'])) + float(r['overlap_signal'])
    return s > 1.5
def fixed_k(k): return lambda r: r['round'] >= k

RULES = {
    'AS': AS, 'famA': famA, 'famB': famB, 'AS_m20': AS_m20, 'famA_m20': famA_m20,
    'C2of4': C2of4, 'D_soft15': D_soft15, 'fixed_k=3': fixed_k(3), 'fixed_k=5': fixed_k(5),
}


def per_q_stop(df: pd.DataFrame, fn) -> pd.DataFrame:
    """Return per-qid: chosen round, generated tokens used up to that round, f1."""
    out = []
    for qid, g in df.groupby('qid'):
        g = g.sort_values('round').reset_index(drop=True)
        chosen_idx = None
        for i, r in g.iterrows():
            if fn(r):
                chosen_idx = i
                break
        if chosen_idx is None:
            chosen_idx = len(g) - 1
        chosen = g.iloc[chosen_idx]
        # cumulative generated tokens through the chosen round
        gen_tok = int(g.iloc[: chosen_idx + 1]['n_tokens'].sum())
        out.append({
            'qid': qid,
            'f1': float(chosen['current_f1']),
            'calls': int(chosen['round']),
            'gen_tokens': gen_tok,
        })
    return pd.DataFrame(out)


def summarize(cell: str, model: str, per_rule_df: dict) -> dict:
    tps = GEN_TOKS_PER_SEC[model]
    price = PRICE_PER_M[model]
    summary = {}
    for name, rdf in per_rule_df.items():
        calls = float(rdf['calls'].mean())
        gen = float(rdf['gen_tokens'].mean())
        inp = calls * INPUT_TOKS_PER_CALL
        sec = gen / tps + calls * RETRIEVAL_SEC_PER_CALL
        usd = (inp * price['in'] + gen * price['out']) / 1_000_000
        summary[name] = {
            'f1_mean': float(rdf['f1'].mean() * 100),
            'calls_mean': calls,
            'gen_tokens_mean': gen,
            'input_tokens_mean': inp,
            'wall_sec_mean': sec,
            'usd_per_q': usd,
            'tps_assumed': tps,
            'price_per_M_in': price['in'],
            'price_per_M_out': price['out'],
        }
    return summary


def main():
    out = {}
    for cell, (model, rel) in CELLS.items():
        df = prep(ROOT / rel)
        per_rule_df = {name: per_q_stop(df, fn) for name, fn in RULES.items()}
        out[cell] = summarize(cell, model, per_rule_df)
        ex = out[cell]['famA']
        print(f"[{cell}] famA: F1={ex['f1_mean']:.2f}, calls={ex['calls_mean']:.2f}, "
              f"gen_tok={ex['gen_tokens_mean']:.0f}, sec={ex['wall_sec_mean']:.2f}, "
              f"USD={ex['usd_per_q']*1e6:.1f}/M-q")
    target = ROOT / 'results' / 'latency_cost_table.json'
    target.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {target}")


if __name__ == '__main__':
    main()
