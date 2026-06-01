"""Paired bootstrap CIs for new rules (margin-floor, C.2-of-4, D.Soft1.5) vs
baselines (AS, famA, famB, fixed_k=3) across all 6 cells.

1000 resamples, seed 42, n=300 per cell. Outputs JSON with per-cell CIs and
significance flags. Pure offline; reads lp_enriched parquets only.
"""

from __future__ import annotations
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = {
    'qwen_hp':      'results/signal_log_lp_enriched.parquet',
    'qwen_2w':      'results/qwen_2wiki/signal_log_lp_enriched.parquet',
    'devstral_hp':  'results/devstral/signal_log_lp_enriched.parquet',
    'devstral_2w':  'results/devstral_2wiki/signal_log_lp_enriched.parquet',
    'gemma_hp':     'results/gemma/signal_log_lp_enriched.parquet',
    'gemma_2w':     'results/gemma_2wiki/signal_log_lp_enriched.parquet',
}
ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 1000
SEED = 42


def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def prep(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values(['qid', 'round']).copy()
    df['norm_ans'] = df['current_answer'].apply(_norm)
    df['prev_ans'] = df.groupby('qid')['norm_ans'].shift(1)
    df['answer_stable'] = ((df['norm_ans'] == df['prev_ans']) & (df['round'] > 1)).astype(int)
    return df


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


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    assert (df_a['qid'].values == df_b['qid'].values).all()
    n = len(df_a)
    da_f1 = df_a['f1'].values * 100
    db_f1 = df_b['f1'].values * 100
    da_c = df_a['calls'].values.astype(float)
    db_c = df_b['calls'].values.astype(float)
    df1, dc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        df1.append((da_f1[idx] - db_f1[idx]).mean())
        dc.append((da_c[idx] - db_c[idx]).mean())
    df1 = np.array(df1)
    dc = np.array(dc)
    return {
        'mean_f1_a': float(da_f1.mean()),
        'mean_f1_b': float(db_f1.mean()),
        'mean_calls_a': float(da_c.mean()),
        'mean_calls_b': float(db_c.mean()),
        'diff_f1_mean': float(df1.mean()),
        'diff_f1_ci95': [float(np.percentile(df1, 2.5)), float(np.percentile(df1, 97.5))],
        'diff_calls_mean': float(dc.mean()),
        'diff_calls_ci95': [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
        'p_a_better_f1': float((df1 > 0).mean()),
        'p_a_worse_f1': float((df1 < 0).mean()),
    }


def AS(r): return r['answer_stable'] == 1
def famA(r): return (r['calibrated_logit_margin'] > 0.725) or (r['overlap_signal'] > 0.15)
def famB(r): return (r['calibrated_conf'] > 0.60) or (r['calibrated_logit_margin'] > 0.675) or (r['overlap_signal'] > 0.15)
def AS_m20(r): return r['answer_stable'] == 1 and r['calibrated_logit_margin'] > 0.20
def famA_m20(r): return (r['calibrated_logit_margin'] > 0.725) or (r['overlap_signal'] > 0.15 and r['calibrated_logit_margin'] > 0.20)
def famB_m20(r): return (r['calibrated_logit_margin'] > 0.675) or (r['calibrated_conf'] > 0.60 and r['calibrated_logit_margin'] > 0.20) or (r['overlap_signal'] > 0.15 and r['calibrated_logit_margin'] > 0.20)


def C2of4(r):
    v = int(r['answer_stable'] == 1) + int(r['calibrated_logit_margin'] > 0.6) + int(r['overlap_signal'] > 0.15) + int(r['calibrated_conf'] > 0.7)
    return v >= 2


def D_soft15(r):
    s = int(r['answer_stable'] == 1) + max(0.0, float(r['calibrated_logit_margin'])) + float(r['overlap_signal'])
    return s > 1.5


def fixed_k(k):
    return lambda r: r['round'] >= k


RULES = {
    'AS': AS,
    'famA': famA,
    'famB': famB,
    'AS_m20': AS_m20,
    'famA_m20': famA_m20,
    'famB_m20': famB_m20,
    'C2of4': C2of4,
    'D_soft15': D_soft15,
    'fixed_k=3': fixed_k(3),
    'fixed_k=5': fixed_k(5),
}

# (a, b) pairs where we test a vs b (a > b is "a wins on F1")
COMPARISONS = [
    ('AS_m20',   'AS'),
    ('AS_m20',   'famA'),
    ('AS_m20',   'fixed_k=3'),
    ('famA_m20', 'famA'),
    ('famA_m20', 'fixed_k=3'),
    ('famB_m20', 'famB'),
    ('C2of4',    'famA'),
    ('C2of4',    'famB'),
    ('C2of4',    'fixed_k=3'),
    ('D_soft15', 'famA'),
    ('D_soft15', 'famB'),
    ('D_soft15', 'fixed_k=3'),
]


def main():
    out = {}
    for cell, rel in CELLS.items():
        df = prep(ROOT / rel)
        per_rule = {name: per_q_stop(df, fn) for name, fn in RULES.items()}
        cell_out = {
            'point_estimates': {name: {
                'f1_mean': float(per_rule[name]['f1'].mean() * 100),
                'calls_mean': float(per_rule[name]['calls'].mean()),
                'n': int(len(per_rule[name])),
            } for name in RULES},
            'comparisons': {},
        }
        for a, b in COMPARISONS:
            cell_out['comparisons'][f'{a}_vs_{b}'] = paired_bootstrap(per_rule[a], per_rule[b])
        out[cell] = cell_out
        print(f"[{cell}] done.  AS_m20 vs AS: ΔF1={cell_out['comparisons']['AS_m20_vs_AS']['diff_f1_mean']:+.2f}, "
              f"CI=[{cell_out['comparisons']['AS_m20_vs_AS']['diff_f1_ci95'][0]:+.2f}, "
              f"{cell_out['comparisons']['AS_m20_vs_AS']['diff_f1_ci95'][1]:+.2f}]")
    target = ROOT / 'results' / 'bootstrap_ci_new_rules.json'
    target.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {target}")


if __name__ == '__main__':
    main()
