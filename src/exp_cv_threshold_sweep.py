"""5-fold cross-validation threshold sweep on famA shape.

Splits each cell's tune set (n=100) into 5 folds, tunes (t_margin, t_overlap) on
4 folds, evaluates on the held-out fold AND on the 300-q eval set. Quantifies
overfitting (tune-set F1 vs eval-set F1 at the tuned thresholds) precisely.

Pure offline. No LLM calls.
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
    'qwen_hp':      ('results/signal_log_lp_enriched_tune.parquet',
                     'results/signal_log_lp_enriched.parquet'),
    'qwen_2w':      ('results/qwen_2wiki/signal_log_lp_enriched_tune.parquet',
                     'results/qwen_2wiki/signal_log_lp_enriched.parquet'),
    'devstral_hp':  ('results/devstral/signal_log_lp_enriched_tune.parquet',
                     'results/devstral/signal_log_lp_enriched.parquet'),
    'devstral_2w':  ('results/devstral_2wiki/signal_log_lp_enriched_tune.parquet',
                     'results/devstral_2wiki/signal_log_lp_enriched.parquet'),
    'gemma_hp':     ('results/gemma/signal_log_lp_enriched_tune.parquet',
                     'results/gemma/signal_log_lp_enriched.parquet'),
    'gemma_2w':     ('results/gemma_2wiki/signal_log_lp_enriched_tune.parquet',
                     'results/gemma_2wiki/signal_log_lp_enriched.parquet'),
}
LOCKED = (0.725, 0.150)
SEED = 42
N_FOLDS = 5
T_M_GRID = np.round(np.arange(0.4, 1.01, 0.025), 4)
T_O_GRID = np.round(np.arange(0.05, 0.41, 0.025), 4)
CALL_CAP = 3.5  # tune-set call budget


def _norm(s):
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def prep(p):
    df = pd.read_parquet(p).sort_values(['qid', 'round']).copy()
    df['norm_ans'] = df['current_answer'].apply(_norm)
    df['prev_ans'] = df.groupby('qid')['norm_ans'].shift(1)
    df['answer_stable'] = ((df['norm_ans'] == df['prev_ans']) & (df['round'] > 1)).astype(int)
    return df


def eval_famA_on(df, t_m, t_o, qids=None):
    if qids is not None:
        df = df[df['qid'].isin(qids)]
    out = []
    for qid, g in df.groupby('qid'):
        g = g.sort_values('round').reset_index(drop=True)
        chosen = None
        for _, r in g.iterrows():
            if (r['calibrated_logit_margin'] > t_m) or (r['overlap_signal'] > t_o):
                chosen = r
                break
        if chosen is None:
            chosen = g.iloc[-1]
        out.append({'f1': float(chosen['current_f1']), 'calls': int(chosen['round'])})
    rdf = pd.DataFrame(out)
    return float(rdf['f1'].mean() * 100), float(rdf['calls'].mean())


def grid_search(tune_df, qids):
    best = None
    best_f1 = -1.0
    for t_m in T_M_GRID:
        for t_o in T_O_GRID:
            f1, c = eval_famA_on(tune_df, t_m, t_o, qids)
            if c > CALL_CAP:
                continue
            if f1 > best_f1:
                best_f1 = f1
                best = (float(t_m), float(t_o), f1, c)
    return best


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    for cell, (tp, ep) in CELLS.items():
        tdf = prep(ROOT / tp)
        edf = prep(ROOT / ep)
        qids = sorted(tdf['qid'].unique())
        rng.shuffle(qids)
        folds = np.array_split(qids, N_FOLDS)
        fold_records = []
        for i in range(N_FOLDS):
            held = list(folds[i])
            kept = [q for j, fold in enumerate(folds) for q in fold if j != i]
            best = grid_search(tdf, kept)
            held_f1, held_c = eval_famA_on(tdf, best[0], best[1], held)
            eval_f1, eval_c = eval_famA_on(edf, best[0], best[1])
            fold_records.append({
                'fold': i,
                'best_t_m': best[0],
                'best_t_o': best[1],
                'train_f1': best[2],
                'train_calls': best[3],
                'held_fold_f1': held_f1,
                'held_fold_calls': held_c,
                'eval_set_f1': eval_f1,
                'eval_set_calls': eval_c,
            })
        # Also: full-tune grid search and eval
        full_best = grid_search(tdf, qids)
        full_eval_f1, full_eval_c = eval_famA_on(edf, full_best[0], full_best[1])
        locked_eval_f1, locked_eval_c = eval_famA_on(edf, *LOCKED)
        df_folds = pd.DataFrame(fold_records)
        results[cell] = {
            'folds': fold_records,
            'cv_train_f1_mean': float(df_folds['train_f1'].mean()),
            'cv_held_f1_mean': float(df_folds['held_fold_f1'].mean()),
            'cv_eval_f1_mean': float(df_folds['eval_set_f1'].mean()),
            'overfit_gap_train_minus_held': float(df_folds['train_f1'].mean() - df_folds['held_fold_f1'].mean()),
            'overfit_gap_train_minus_eval': float(df_folds['train_f1'].mean() - df_folds['eval_set_f1'].mean()),
            'full_tune_best': {'t_m': full_best[0], 't_o': full_best[1],
                               'tune_f1': full_best[2], 'eval_f1': full_eval_f1, 'eval_calls': full_eval_c},
            'locked': {'t_m': LOCKED[0], 't_o': LOCKED[1],
                       'eval_f1': locked_eval_f1, 'eval_calls': locked_eval_c},
            'delta_tuned_minus_locked_f1': float(full_eval_f1 - locked_eval_f1),
            'delta_tuned_minus_locked_calls': float(full_eval_c - locked_eval_c),
        }
        print(f"[{cell}] CV train F1 = {results[cell]['cv_train_f1_mean']:.2f}, held = {results[cell]['cv_held_f1_mean']:.2f}, "
              f"eval-set = {results[cell]['cv_eval_f1_mean']:.2f}, "
              f"overfit gap (train-held) = {results[cell]['overfit_gap_train_minus_held']:+.2f}")
    target = ROOT / 'results' / 'cv_threshold_sweep.json'
    target.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {target}")


if __name__ == '__main__':
    main()
