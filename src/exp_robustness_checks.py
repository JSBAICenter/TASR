"""Two robustness checks for AS_m20 before committing the headline.

1. Margin-floor sweep: AS with floor in {0.10, 0.15, 0.20, 0.25, 0.30}, all 6
   cells. Reports F1/calls + paired-bootstrap ΔF1 vs AS_m20 and vs fixed_k=3.
   Defends the hand-picked 0.20 against the obvious "why 0.20?" attack.

2. CE z-score gate as add-on to AS_m20: stop if AS_m20 fires OR
   `ce_top_score_z_intra > T` for T in {0.5, 1.0, 1.5}. Uses the enriched
   `*_ce.parquet` files written by exp_reranker_signal.py. Tests whether the
   reranker, when used as a per-qid "this round's evidence is unusually
   relevant" signal (rather than absolute CE>0), adds a fast-exit path.

Pure offline. No LLM calls. Outputs to results/robustness_checks.json.
"""

from __future__ import annotations
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CELLS_BASE = {
    'qwen_hp':      'results/signal_log_lp_enriched.parquet',
    'qwen_2w':      'results/qwen_2wiki/signal_log_lp_enriched.parquet',
    'devstral_hp':  'results/devstral/signal_log_lp_enriched.parquet',
    'devstral_2w':  'results/devstral_2wiki/signal_log_lp_enriched.parquet',
    'gemma_hp':     'results/gemma/signal_log_lp_enriched.parquet',
    'gemma_2w':     'results/gemma_2wiki/signal_log_lp_enriched.parquet',
}
CELLS_CE = {
    'qwen_hp':      'results/signal_log_lp_enriched_ce.parquet',
    'qwen_2w':      'results/qwen_2wiki/signal_log_lp_enriched_ce.parquet',
    'devstral_hp':  'results/devstral/signal_log_lp_enriched_ce.parquet',
    'devstral_2w':  'results/devstral_2wiki/signal_log_lp_enriched_ce.parquet',
    'gemma_hp':     'results/gemma/signal_log_lp_enriched_ce.parquet',
    'gemma_2w':     'results/gemma_2wiki/signal_log_lp_enriched_ce.parquet',
}
N_BOOT = 1000
SEED = 42
FLOORS = [round(0.20 + 0.01 * i, 2) for i in range(16)]  # 0.20, 0.21, ..., 0.35
Z_THRESHOLDS = [0.5, 1.0, 1.5]


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


def AS(r): return r['answer_stable'] == 1
def AS_floor(t):
    return lambda r: r['answer_stable'] == 1 and r['calibrated_logit_margin'] > t
def AS_m20_or_ce_z(t):
    return lambda r: (r['answer_stable'] == 1 and r['calibrated_logit_margin'] > 0.20) or (r['ce_top_score_z_intra'] > t)
def fixed_k(k): return lambda r: r['round'] >= k


# ----- (1) Margin-floor sweep -----
def run_floor_sweep():
    print("\n=== (1) Margin-floor sweep ===\n")
    print(f"{'cell':12s} | " + " | ".join(f"floor={t:.2f}".ljust(13) for t in FLOORS))
    print("-" * 100)
    out = {}
    macro_acc = {f"floor={t:.2f}": {'f1': [], 'calls': []} for t in FLOORS}
    for cell, rel in CELLS_BASE.items():
        df = prep(ROOT / rel)
        as_df = per_q_stop(df, AS)
        fk3_df = per_q_stop(df, fixed_k(3))
        cell_out = {'AS': {
            'f1': float(as_df['f1'].mean() * 100),
            'calls': float(as_df['calls'].mean()),
        }}
        row = [f"{cell:12s} |"]
        for t in FLOORS:
            r = per_q_stop(df, AS_floor(t))
            f1, c = r['f1'].mean() * 100, r['calls'].mean()
            b_as = paired_bootstrap(r, as_df)
            b_fk3 = paired_bootstrap(r, fk3_df)
            key = f"floor={t:.2f}"
            cell_out[key] = {
                'f1': float(f1), 'calls': float(c),
                'vs_AS': b_as, 'vs_fixed3': b_fk3,
            }
            macro_acc[key]['f1'].append(f1)
            macro_acc[key]['calls'].append(c)
            row.append(f"{f1:5.2f}/{c:.2f}".ljust(13))
        print(" ".join(row))
        out[cell] = cell_out
    # macro
    print(f"\n{'MACRO':12s} | " + " | ".join(
        f"{np.mean(macro_acc[k]['f1']):5.2f}/{np.mean(macro_acc[k]['calls']):.2f}".ljust(13)
        for k in macro_acc))
    out['macro'] = {k: {'f1': float(np.mean(v['f1'])),
                        'calls': float(np.mean(v['calls']))}
                    for k, v in macro_acc.items()}
    return out


# ----- (2) CE z-score gate as AS_m20 add-on -----
def run_ce_z_addon():
    print("\n\n=== (2) CE z-score gate as add-on to AS_m20 ===\n")
    print(f"{'cell':12s} | " + f"{'AS_m20 F1/c':>13s} | " + " | ".join(
        f"+CE_z>{t:.1f} F1/c".ljust(15) for t in Z_THRESHOLDS))
    print("-" * 100)
    out = {}
    macro_acc = {'AS_m20': {'f1': [], 'calls': []}}
    for t in Z_THRESHOLDS:
        macro_acc[f"AS_m20|CE_z>{t}"] = {'f1': [], 'calls': []}
    for cell, rel in CELLS_CE.items():
        df = prep(ROOT / rel)
        asm20 = per_q_stop(df, AS_floor(0.20))
        fk3_df = per_q_stop(df, fixed_k(3))
        cell_out = {'AS_m20': {
            'f1': float(asm20['f1'].mean() * 100),
            'calls': float(asm20['calls'].mean()),
        }}
        macro_acc['AS_m20']['f1'].append(asm20['f1'].mean() * 100)
        macro_acc['AS_m20']['calls'].append(asm20['calls'].mean())
        row = [f"{cell:12s} |", f"{asm20['f1'].mean()*100:5.2f}/{asm20['calls'].mean():.2f}".rjust(13), "|"]
        for t in Z_THRESHOLDS:
            r = per_q_stop(df, AS_m20_or_ce_z(t))
            f1, c = r['f1'].mean() * 100, r['calls'].mean()
            b_asm20 = paired_bootstrap(r, asm20)
            b_fk3 = paired_bootstrap(r, fk3_df)
            key = f"AS_m20|CE_z>{t}"
            cell_out[key] = {
                'f1': float(f1), 'calls': float(c),
                'vs_AS_m20': b_asm20, 'vs_fixed3': b_fk3,
            }
            macro_acc[key]['f1'].append(f1)
            macro_acc[key]['calls'].append(c)
            row.append(f"{f1:5.2f}/{c:.2f}".ljust(15))
        print(" ".join(row))
        out[cell] = cell_out
    print(f"\n{'MACRO':12s} | " + f"{np.mean(macro_acc['AS_m20']['f1']):5.2f}/{np.mean(macro_acc['AS_m20']['calls']):.2f}".rjust(13) + " | " + " | ".join(
        f"{np.mean(macro_acc[f'AS_m20|CE_z>{t}']['f1']):5.2f}/{np.mean(macro_acc[f'AS_m20|CE_z>{t}']['calls']):.2f}".ljust(15)
        for t in Z_THRESHOLDS))
    out['macro'] = {k: {'f1': float(np.mean(v['f1'])),
                        'calls': float(np.mean(v['calls']))}
                    for k, v in macro_acc.items()}
    return out


def main():
    floor_results = run_floor_sweep()
    ce_z_results = run_ce_z_addon()
    target = ROOT / 'results' / 'robustness_checks.json'
    target.write_text(json.dumps({
        'floor_sweep': floor_results,
        'ce_z_addon': ce_z_results,
    }, indent=2))
    print(f"\nSaved: {target}")

    # Per-cell ΔF1 vs AS_m20 (floor=0.20) summary for the sweep
    print("\n=== ΔF1 vs AS_m20 (floor=0.20) — sweep ===")
    for cell in CELLS_BASE:
        cell_out = floor_results[cell]
        print(f"\n{cell}:")
        for t in FLOORS:
            key = f"floor={t:.2f}"
            d = cell_out[key]['vs_AS']
            print(f"  floor={t:.2f}  F1={cell_out[key]['f1']:5.2f}  calls={cell_out[key]['calls']:.2f}  "
                  f"ΔF1 vs AS={d['diff_f1_mean']:+5.2f} [{d['diff_f1_ci95'][0]:+5.2f},{d['diff_f1_ci95'][1]:+5.2f}]")

    print("\n=== ΔF1 vs AS_m20 for CE z-score add-on ===")
    for cell in CELLS_CE:
        cell_out = ce_z_results[cell]
        print(f"\n{cell}:")
        for t in Z_THRESHOLDS:
            key = f"AS_m20|CE_z>{t}"
            d = cell_out[key]['vs_AS_m20']
            print(f"  CE_z>{t}  F1={cell_out[key]['f1']:5.2f}  calls={cell_out[key]['calls']:.2f}  "
                  f"ΔF1 vs AS_m20={d['diff_f1_mean']:+5.2f} [{d['diff_f1_ci95'][0]:+5.2f},{d['diff_f1_ci95'][1]:+5.2f}]")


if __name__ == '__main__':
    main()
