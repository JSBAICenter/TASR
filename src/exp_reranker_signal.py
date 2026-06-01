"""BGE-reranker-v2-m3 as a 7th signal, evaluated in C.2-of-5.

Pipeline (per dataset):
  1. Build BM25-ranked paragraph list per question (identical across models since
     BM25 is deterministic and only depends on question + corpus).
  2. Cross-encode (question, paragraph) for the top-5 BM25 paragraphs per qid
     with BAAI/bge-reranker-v2-m3.
  3. For each parquet row at round r, derive
        ce_top_score = max(CE(q, p_i) for i < r)
     plus the intra-qid z-score.
  4. Save enriched parquets next to the originals (suffix `_ce.parquet`),
     evaluate AS / famA / famB / C2of4 / C2of5 with bootstrap CIs.

`C2of5` adds CE>T (T = 0.0 — bge-reranker emits logit-style scores, ~0 is the
relevance threshold) as a fifth vote on top of C2of4's four votes.

Datasets: HotpotQA (data/dev_eval.json) and 2Wiki (data/2wiki_dev_eval.json).
Reused per model since the BM25 ordering is identical across models.
"""

from __future__ import annotations
import json
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from retrieval import build_paragraphs, retrieve  # noqa: E402

MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
TOP_K = 5
BATCH = 32
N_BOOT = 1000
SEED = 42
CE_THRESHOLD = 0.0  # bge-reranker-v2-m3 emits logit-style scores; > 0 ≈ relevant

DATASETS = {
    'hp': 'data/dev_eval.json',
    '2w': 'data/2wiki_dev_eval.json',
}
CELLS = {
    'qwen_hp':      ('hp', 'results/signal_log_lp_enriched.parquet'),
    'qwen_2w':      ('2w', 'results/qwen_2wiki/signal_log_lp_enriched.parquet'),
    'devstral_hp':  ('hp', 'results/devstral/signal_log_lp_enriched.parquet'),
    'devstral_2w':  ('2w', 'results/devstral_2wiki/signal_log_lp_enriched.parquet'),
    'gemma_hp':     ('hp', 'results/gemma/signal_log_lp_enriched.parquet'),
    'gemma_2w':     ('2w', 'results/gemma_2wiki/signal_log_lp_enriched.parquet'),
}


def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def build_pair_scores(data_path: Path, model) -> dict[tuple[str, int], float]:
    with open(data_path) as f:
        items = json.load(f)
    pairs, keys = [], []
    for it in items:
        paragraphs = build_paragraphs(it)
        ranked = retrieve(it['question'], paragraphs, k=TOP_K)
        for r_idx, p in enumerate(ranked):
            pairs.append((it['question'], p['text']))
            keys.append((it['id'], r_idx))
    t0 = time.time()
    print(f"  scoring {len(pairs)} pairs with {MODEL_NAME}...", flush=True)
    scores = model.predict(pairs, batch_size=BATCH, show_progress_bar=True)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({len(pairs)/elapsed:.1f} pairs/s)", flush=True)
    return {k: float(s) for k, s in zip(keys, scores)}


def add_ce_signals(df: pd.DataFrame, pair_scores: dict[tuple[str, int], float]) -> pd.DataFrame:
    df = df.sort_values(['qid', 'round']).copy().reset_index(drop=True)
    ce_top = np.zeros(len(df))
    for i, row in df.iterrows():
        qid = row['qid']
        r = int(row['round'])
        scores_r = [pair_scores.get((qid, j), -1e9) for j in range(min(r, TOP_K))]
        ce_top[i] = max(scores_r) if scores_r else -1e9
    df['ce_top_score'] = ce_top
    df['ce_top_score_z_intra'] = df.groupby('qid')['ce_top_score'].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9)
    )
    return df


def prep(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p).sort_values(['qid', 'round']).copy()
    df['norm_ans'] = df['current_answer'].apply(_norm)
    df['prev_ans'] = df.groupby('qid')['norm_ans'].shift(1)
    df['answer_stable'] = ((df['norm_ans'] == df['prev_ans']) & (df['round'] > 1)).astype(int)
    return df


def AS(r): return r['answer_stable'] == 1
def famA(r): return (r['calibrated_logit_margin'] > 0.725) or (r['overlap_signal'] > 0.15)
def famB(r): return (r['calibrated_conf'] > 0.60) or (r['calibrated_logit_margin'] > 0.675) or (r['overlap_signal'] > 0.15)
def C2of4(r):
    v = int(r['answer_stable'] == 1) + int(r['calibrated_logit_margin'] > 0.6) + int(r['overlap_signal'] > 0.15) + int(r['calibrated_conf'] > 0.7)
    return v >= 2
def C2of5(r):
    v = (int(r['answer_stable'] == 1) + int(r['calibrated_logit_margin'] > 0.6)
         + int(r['overlap_signal'] > 0.15) + int(r['calibrated_conf'] > 0.7)
         + int(r['ce_top_score'] > CE_THRESHOLD))
    return v >= 2
def C3of5(r):
    v = (int(r['answer_stable'] == 1) + int(r['calibrated_logit_margin'] > 0.6)
         + int(r['overlap_signal'] > 0.15) + int(r['calibrated_conf'] > 0.7)
         + int(r['ce_top_score'] > CE_THRESHOLD))
    return v >= 3

RULES = {'AS': AS, 'famA': famA, 'famB': famB, 'C2of4': C2of4, 'C2of5': C2of5, 'C3of5': C3of5}


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
        'mean_f1_a': float(af1.mean()), 'mean_f1_b': float(bf1.mean()),
        'mean_calls_a': float(ac.mean()), 'mean_calls_b': float(bc.mean()),
        'diff_f1_mean': float(df1.mean()),
        'diff_f1_ci95': [float(np.percentile(df1, 2.5)), float(np.percentile(df1, 97.5))],
        'diff_calls_mean': float(dc.mean()),
        'diff_calls_ci95': [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
        'p_a_better_f1': float((df1 > 0).mean()),
    }


def main():
    from sentence_transformers import CrossEncoder
    print(f"Loading reranker {MODEL_NAME}...", flush=True)
    model = CrossEncoder(MODEL_NAME, max_length=512)
    print("  loaded.")

    pair_scores_cache: dict[str, dict] = {}
    for ds, rel in DATASETS.items():
        print(f"\n=== dataset {ds}: scoring once and reusing across models ===")
        pair_scores_cache[ds] = build_pair_scores(ROOT / rel, model)

    enriched = {}
    for cell, (ds, rel) in CELLS.items():
        print(f"\n--- enriching {cell} ---", flush=True)
        df = prep(ROOT / rel)
        df = add_ce_signals(df, pair_scores_cache[ds])
        out_p = (ROOT / rel).with_name((ROOT / rel).stem + '_ce.parquet')
        df.to_parquet(out_p, index=False)
        print(f"  saved {out_p}")
        enriched[cell] = df

    print("\n=== evaluating rules with bootstrap CIs ===", flush=True)
    out = {}
    for cell, df in enriched.items():
        per_rule = {name: per_q_stop(df, fn) for name, fn in RULES.items()}
        cell_out = {
            'point_estimates': {name: {
                'f1_mean': float(per_rule[name]['f1'].mean() * 100),
                'calls_mean': float(per_rule[name]['calls'].mean()),
                'n': int(len(per_rule[name])),
            } for name in RULES},
            'comparisons': {},
        }
        for a, b in [('C2of5', 'C2of4'), ('C2of5', 'famA'), ('C2of5', 'famB'),
                     ('C3of5', 'C2of4'), ('C3of5', 'famA'), ('C3of5', 'famB')]:
            cell_out['comparisons'][f'{a}_vs_{b}'] = paired_bootstrap(per_rule[a], per_rule[b])
        out[cell] = cell_out
        c25 = cell_out['point_estimates']['C2of5']
        c24 = cell_out['point_estimates']['C2of4']
        d = cell_out['comparisons']['C2of5_vs_C2of4']
        print(f"[{cell}] C2of4 F1={c24['f1_mean']:.2f}/c={c24['calls_mean']:.2f}  "
              f"C2of5 F1={c25['f1_mean']:.2f}/c={c25['calls_mean']:.2f}  "
              f"ΔF1={d['diff_f1_mean']:+.2f} CI=[{d['diff_f1_ci95'][0]:+.2f},{d['diff_f1_ci95'][1]:+.2f}]")
    target = ROOT / 'results' / 'reranker_C2of5.json'
    target.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {target}")


if __name__ == '__main__':
    main()
