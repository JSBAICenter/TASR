"""Big comparison table: all rules vs fixed_k=3, with paired bootstrap CIs.

Rules: AS, AS_m20, famA, famA_m20, famB, famB_m20, C2of4, D_soft15,
       fixed_k=1, fixed_k=3, fixed_k=5, oracle.
Per-cell: F1, calls, ΔF1 vs fixed_k=3 with 95% CI.
Also: macro across the 6 cells.
"""

from __future__ import annotations
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CELLS = {
    'qwen_hp':      'results/signal_log_lp_enriched.parquet',
    'qwen_2w':      'results/qwen_2wiki/signal_log_lp_enriched.parquet',
    'devstral_hp':  'results/devstral/signal_log_lp_enriched.parquet',
    'devstral_2w':  'results/devstral_2wiki/signal_log_lp_enriched.parquet',
    'gemma_hp':     'results/gemma/signal_log_lp_enriched.parquet',
    'gemma_2w':     'results/gemma_2wiki/signal_log_lp_enriched.parquet',
}
N_BOOT = 1000
SEED = 42


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
def AS_m20(r): return r['answer_stable'] == 1 and r['calibrated_logit_margin'] > 0.20
def AS_m25(r): return r['answer_stable'] == 1 and r['calibrated_logit_margin'] > 0.25
# Guardrail family: stable + supporting-sentence overlap and/or
# evidence entailment. Framing: "require AS plus either overlap or
# entailment from retrieved evidence". `judge_entailment` is the LLM-judge
# 1-5 score over (question, evidence, candidate answer); cutoff 4 = "supports
# the answer with minor gaps" or stronger.
def AS_ov(r): return r['answer_stable'] == 1 and r['overlap_signal'] >= 0.15
def AS_en(r): return r['answer_stable'] == 1 and r.get('judge_entailment', 0) >= 4
def AS_ov_or_en(r): return r['answer_stable'] == 1 and (r['overlap_signal'] >= 0.15 or r.get('judge_entailment', 0) >= 4)
def AS_m25_ov_or_en(r): return r['answer_stable'] == 1 and r['calibrated_logit_margin'] > 0.25 and (r['overlap_signal'] >= 0.15 or r.get('judge_entailment', 0) >= 4)
def famA(r): return (r['calibrated_logit_margin'] > 0.725) or (r['overlap_signal'] > 0.15)
def famA_m20(r): return (r['calibrated_logit_margin'] > 0.725) or (r['overlap_signal'] > 0.15 and r['calibrated_logit_margin'] > 0.20)
def famB(r): return (r['calibrated_conf'] > 0.60) or (r['calibrated_logit_margin'] > 0.675) or (r['overlap_signal'] > 0.15)
def famB_m20(r): return (r['calibrated_logit_margin'] > 0.675) or (r['calibrated_conf'] > 0.60 and r['calibrated_logit_margin'] > 0.20) or (r['overlap_signal'] > 0.15 and r['calibrated_logit_margin'] > 0.20)
def C2of4(r):
    v = int(r['answer_stable'] == 1) + int(r['calibrated_logit_margin'] > 0.6) + int(r['overlap_signal'] > 0.15) + int(r['calibrated_conf'] > 0.7)
    return v >= 2
def D_soft15(r):
    s = int(r['answer_stable'] == 1) + max(0.0, float(r['calibrated_logit_margin'])) + float(r['overlap_signal'])
    return s > 1.5
def fixed_k(k): return lambda r: r['round'] >= k


RULES = {
    'AS': AS, 'AS_m20': AS_m20, 'AS_m25': AS_m25,
    'AS_ov': AS_ov, 'AS_en': AS_en, 'AS_ov_or_en': AS_ov_or_en,
    'AS_m25_ov_or_en': AS_m25_ov_or_en,
    'famA': famA, 'famA_m20': famA_m20,
    'famB': famB, 'famB_m20': famB_m20,
    'C2of4': C2of4, 'D_soft15': D_soft15,
    'fixed_k=1': fixed_k(1), 'fixed_k=3': fixed_k(3), 'fixed_k=5': fixed_k(5),
}
BASELINE = 'fixed_k=3'


def per_q_stop(df: pd.DataFrame, fn) -> pd.DataFrame:
    out = []
    for qid, g in df.groupby('qid'):
        g = g.sort_values('round').reset_index(drop=True)
        chosen = None
        for _, r in g.iterrows():
            if fn(r):
                chosen = r
                break
        if chosen is None:
            chosen = g.iloc[-1]
        out.append({'qid': qid, 'f1': float(chosen['current_f1']), 'calls': int(chosen['round'])})
    return pd.DataFrame(out).sort_values('qid').reset_index(drop=True)


def per_q_oracle(df: pd.DataFrame) -> pd.DataFrame:
    """Per-qid best F1; round = first round that achieves the per-qid max F1."""
    out = []
    for qid, g in df.groupby('qid'):
        g = g.sort_values('round').reset_index(drop=True)
        best_f1 = g['current_f1'].max()
        first = g[g['current_f1'] == best_f1].iloc[0]
        out.append({'qid': qid, 'f1': float(first['current_f1']), 'calls': int(first['round'])})
    return pd.DataFrame(out).sort_values('qid').reset_index(drop=True)


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    assert (a['qid'].values == b['qid'].values).all()
    n = len(a)
    af1, bf1 = a['f1'].values * 100, b['f1'].values * 100
    ac, bc = a['calls'].values.astype(float), b['calls'].values.astype(float)
    df1, dc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        df1.append((af1[idx] - bf1[idx]).mean())
        dc.append((ac[idx] - bc[idx]).mean())
    df1 = np.array(df1); dc = np.array(dc)
    return {
        'diff_f1_mean': float(df1.mean()),
        'diff_f1_ci95': [float(np.percentile(df1, 2.5)), float(np.percentile(df1, 97.5))],
        'diff_calls_mean': float(dc.mean()),
        'diff_calls_ci95': [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
    }


def main():
    out = {}
    rule_names = list(RULES.keys()) + ['oracle']
    for cell, rel in CELLS.items():
        df = prep(ROOT / rel)
        per_rule = {name: per_q_stop(df, fn) for name, fn in RULES.items()}
        per_rule['oracle'] = per_q_oracle(df)
        cell_out = {'point': {}, 'vs_fixed3': {}}
        for name in rule_names:
            cell_out['point'][name] = {
                'f1': float(per_rule[name]['f1'].mean() * 100),
                'calls': float(per_rule[name]['calls'].mean()),
            }
            cell_out['vs_fixed3'][name] = paired_bootstrap(per_rule[name], per_rule[BASELINE])
        out[cell] = cell_out

    # macro
    macro = {'point': {}, 'vs_fixed3': {}}
    for name in rule_names:
        f1s = [out[c]['point'][name]['f1'] for c in CELLS]
        cs = [out[c]['point'][name]['calls'] for c in CELLS]
        df1s = [out[c]['vs_fixed3'][name]['diff_f1_mean'] for c in CELLS]
        dcs = [out[c]['vs_fixed3'][name]['diff_calls_mean'] for c in CELLS]
        macro['point'][name] = {'f1': float(np.mean(f1s)), 'calls': float(np.mean(cs))}
        macro['vs_fixed3'][name] = {'diff_f1_mean': float(np.mean(df1s)),
                                    'diff_calls_mean': float(np.mean(dcs))}
    out['macro'] = macro

    target = ROOT / 'results' / 'headline_table.json'
    target.write_text(json.dumps(out, indent=2))
    print(f"Saved: {target}\n")

    # print pretty table
    cols = ['AS', 'AS_m20', 'AS_m25', 'AS_ov', 'AS_en', 'AS_ov_or_en', 'AS_m25_ov_or_en',
            'famA', 'famA_m20', 'famB', 'famB_m20',
            'C2of4', 'D_soft15', 'fixed_k=1', 'fixed_k=3', 'fixed_k=5', 'oracle']
    for cell in list(CELLS.keys()) + ['macro']:
        print(f"\n=== {cell} (baseline = fixed_k=3) ===")
        print(f"{'rule':12s} | {'F1':>6s} | {'calls':>5s} | {'ΔF1':>7s} | {'CI95':>16s} | {'Δcalls':>7s}")
        print('-' * 78)
        for name in cols:
            p = out[cell]['point'][name]
            d = out[cell]['vs_fixed3'][name]
            if cell == 'macro' or 'diff_f1_ci95' not in d:
                ci = '  (macro)        '
            else:
                ci = f"[{d['diff_f1_ci95'][0]:+5.2f},{d['diff_f1_ci95'][1]:+5.2f}]"
            print(f"{name:12s} | {p['f1']:6.2f} | {p['calls']:5.2f} | "
                  f"{d['diff_f1_mean']:+7.2f} | {ci:>16s} | {d['diff_calls_mean']:+7.2f}")


if __name__ == '__main__':
    main()
