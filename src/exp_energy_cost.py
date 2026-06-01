"""Energy and CO2 per question, with bracketed assumptions.

Defensible back-of-envelope. We do NOT claim a measured GPU power on our
cluster; we bracket per-token energy using published measurements for
comparable-class models, and report kWh / CO2 ranges, not point estimates.

The ratio between two stopping methods (calls saved -> tokens saved) is robust
to which J/token point you pick; the absolute kWh is not. We therefore report:
  (a) bracketed absolute energy / CO2 per question (LOW / MID / HIGH),
  (b) ratio AS_m25 / fixed_k=5 (energy savings, robust to bracket),
  (c) Stop-RAG training-amortization break-even, in queries.

Cited constants
---------------
- J/token bracket (covers prefill + decode aggregate, total tokens):
    LOW  = 0.3  J/token  -- well-batched 7-8B class on A100, lower bound
                            (Samsi et al. 2023, "From Words to Watts",
                            arXiv:2310.03003).
    MID  = 1.0  J/token  -- 8B-class median. Derived from Luccioni et al.
                            2024 ("Power Hungry Processing", FAccT'24,
                            arXiv:2311.16863): BLOOMz-7B reports 1.0e-4
                            kWh / inference; text gen median 0.042 kWh /
                            1000 inferences. At ~150 tokens / inference
                            that's ~1.0 J/token.
    HIGH = 3.5  J/token  -- LLaMA-65B on A100/V100 upper anchor
                            (Samsi et al. 2023).
  The bracket spans 7.5x, which matches the published spread for 7-65B
  models across hardware / precision / batch settings. Our models are
  24-31B; their per-token energy is expected somewhere between MID and
  HIGH, but we report the full bracket so reviewers can pick.

- PUE = 1.1  (Patterson et al. 2021 hyperscale convention,
  arXiv:2104.10350). We do NOT claim our cluster's PUE.

- Grid intensity (gCO2eq / kWh):
    iea_2024_global =  445  -- IEA Electricity 2025 baseline
    clean_norway    =   50  -- low-bound sensitivity (mostly hydro)
    us_dc_weighted  =  500  -- upper-bound sensitivity (US data-center
                              weighted intensity, ~48% above the US grid
                              average per arXiv:2411.09786)

- Input tokens per call: 1000 (question + 5 BM25 paragraphs, derived
  from the prompt template in src/agent.py). Generated tokens come from
  the parquet's `n_tokens` column.

Operational scope only. We do not attribute embodied / hardware
manufacturing carbon (different methodology; see Gupta et al. 2020).

Stop-RAG (Park et al. 2025, arXiv:2510.14337) training-cost envelope
-------------------------------------------------------------------
Their Appendix B specifies AdamW + DeBERTa-v3-large + N=8 reward samples
per state across MuSiQue + HotpotQA + 2WikiMultihopQA. Trajectory
generation dominates training; we estimate using their reported
hyperparameters:
  ~100k training questions (full train splits, three datasets) x
  max iter horizon T = 10 x
  N = 8 reward samples per state x
  ~3200 tokens per call (their prompt template; prefill+decode)
DeBERTa-v3-large fine-tune is <0.1% of this and ignored.
"""

from __future__ import annotations
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# --- Cited constants ---------------------------------------------------------
J_PER_TOKEN_BRACKET = {
    'low':  0.3,   # Samsi 2023, well-batched 7-8B A100 lower bound
    'mid':  1.0,   # Luccioni 2024, 8B-class median
    'high': 3.5,   # Samsi 2023, LLaMA-65B on A100/V100 upper anchor
}
PUE = 1.1
GRID_GCO2_PER_KWH = {
    'iea_2024_global': 445.0,
    'clean_norway':     50.0,
    'us_dc_weighted':  500.0,
}
DEFAULT_BRACKET = 'mid'
DEFAULT_GRID = 'iea_2024_global'

INPUT_TOKS_PER_CALL = 1000

# Stop-RAG training envelope (their hyperparameters)
STOP_RAG = {
    'n_train_questions_total': 100_000,
    'horizon_T':               10,
    'n_reward_samples':        8,
    'tokens_per_call':         3200,
}

CELLS = {
    'qwen_hp':      'results/signal_log_lp_enriched.parquet',
    'qwen_2w':      'results/qwen_2wiki/signal_log_lp_enriched.parquet',
    'devstral_hp':  'results/devstral/signal_log_lp_enriched.parquet',
    'devstral_2w':  'results/devstral_2wiki/signal_log_lp_enriched.parquet',
    'gemma_hp':     'results/gemma/signal_log_lp_enriched.parquet',
    'gemma_2w':     'results/gemma_2wiki/signal_log_lp_enriched.parquet',
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
def AS_m20(r): return r['answer_stable'] == 1 and r['calibrated_logit_margin'] > 0.20
def AS_m25(r): return r['answer_stable'] == 1 and r['calibrated_logit_margin'] > 0.25
def famA(r): return (r['calibrated_logit_margin'] > 0.725) or (r['overlap_signal'] > 0.15)
def famB(r): return (r['calibrated_conf'] > 0.60) or (r['calibrated_logit_margin'] > 0.675) or (r['overlap_signal'] > 0.15)
def fixed_k(k): return lambda r: r['round'] >= k

RULES = {
    'AS': AS, 'AS_m20': AS_m20, 'AS_m25': AS_m25,
    'famA': famA, 'famB': famB,
    'fixed_k=3': fixed_k(3), 'fixed_k=5': fixed_k(5),
}


def per_q_stop(df: pd.DataFrame, fn) -> pd.DataFrame:
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
        gen_tok = int(g.iloc[: chosen_idx + 1]['n_tokens'].sum())
        out.append({
            'qid': qid,
            'f1': float(chosen['current_f1']),
            'calls': int(chosen['round']),
            'gen_tokens': gen_tok,
        })
    return pd.DataFrame(out)


def energy_J(total_tokens: float, j_per_tok: float) -> float:
    return PUE * j_per_tok * total_tokens


def to_co2_kg(Wh: float, grid_g_per_kwh: float) -> float:
    return (Wh / 1000.0) * grid_g_per_kwh / 1000.0


def summarize_cell(per_rule_df: dict) -> dict:
    out = {}
    for name, rdf in per_rule_df.items():
        calls = float(rdf['calls'].mean())
        gen = float(rdf['gen_tokens'].mean())
        f1 = float(rdf['f1'].mean() * 100)
        total_tok = gen + calls * INPUT_TOKS_PER_CALL
        bracket_J = {b: energy_J(total_tok, j) for b, j in J_PER_TOKEN_BRACKET.items()}
        bracket_Wh = {b: bracket_J[b] / 3600.0 for b in bracket_J}
        co2 = {b: {g: to_co2_kg(bracket_Wh[b], gi) for g, gi in GRID_GCO2_PER_KWH.items()}
               for b in bracket_J}
        out[name] = {
            'f1_mean': f1,
            'calls_mean': calls,
            'gen_tokens_mean': gen,
            'total_tokens_mean': total_tok,
            'energy_J_per_q':  bracket_J,
            'energy_Wh_per_q': bracket_Wh,
            'co2_g_per_q':     {b: {g: co2[b][g] * 1000.0 for g in co2[b]} for b in co2},
        }
    return out


def print_cell_table(cell: str, summary: dict):
    print(f"\n=== {cell} (MID J/tok = {J_PER_TOKEN_BRACKET[DEFAULT_BRACKET]}, "
          f"grid = {DEFAULT_GRID} {GRID_GCO2_PER_KWH[DEFAULT_GRID]:.0f} gCO2/kWh) ===")
    print(f"{'rule':12s} | {'F1':>5s} | {'calls':>5s} | "
          f"{'kJ/Q low':>9s} | {'kJ/Q mid':>9s} | {'kJ/Q high':>9s} | "
          f"{'gCO2/Q mid':>11s}")
    print('-' * 90)
    for name in RULES:
        s = summary[name]
        kJ = {b: s['energy_J_per_q'][b] / 1000.0 for b in J_PER_TOKEN_BRACKET}
        gco2_mid = s['co2_g_per_q'][DEFAULT_BRACKET][DEFAULT_GRID]
        print(f"{name:12s} | {s['f1_mean']:5.2f} | {s['calls_mean']:5.2f} | "
              f"{kJ['low']:9.3f} | {kJ['mid']:9.3f} | {kJ['high']:9.3f} | "
              f"{gco2_mid:11.3f}")


def headline_savings_macro(macro: dict, headline='AS_m25', baseline='fixed_k=5') -> dict:
    h, b = macro[headline], macro[baseline]
    return {
        'delta_f1':         h['f1_mean'] - b['f1_mean'],
        'delta_calls':      h['calls_mean'] - b['calls_mean'],
        'energy_ratio':     h['energy_Wh_per_q'][DEFAULT_BRACKET] / b['energy_Wh_per_q'][DEFAULT_BRACKET],
        'savings_per_q':    {b_k: (b['energy_Wh_per_q'][b_k] - h['energy_Wh_per_q'][b_k]) for b_k in J_PER_TOKEN_BRACKET},
        'co2_saved_kg_per_Mq': {
            b_k: {g: (b['co2_g_per_q'][b_k][g] - h['co2_g_per_q'][b_k][g]) * 1e6 / 1e3  # g/Q * 1M = g/Mq; /1000 -> kg/Mq
                  for g in GRID_GCO2_PER_KWH}
            for b_k in J_PER_TOKEN_BRACKET},
    }


def stop_rag_training_envelope() -> dict:
    n_train = STOP_RAG['n_train_questions_total']
    T = STOP_RAG['horizon_T']
    N = STOP_RAG['n_reward_samples']
    tok = STOP_RAG['tokens_per_call']
    total_calls = n_train * T * N
    total_tok = total_calls * tok
    out = {
        'assumptions': dict(STOP_RAG),
        'total_calls': total_calls,
        'total_tokens': total_tok,
    }
    for b, j in J_PER_TOKEN_BRACKET.items():
        E_J = PUE * j * total_tok
        kWh = E_J / 3.6e6
        out[f'energy_kWh_{b}'] = kWh
        for g, gi in GRID_GCO2_PER_KWH.items():
            out[f'co2_kg_{b}_{g}'] = to_co2_kg(kWh * 1000.0, gi)
    return out


def main():
    out = {
        'constants': {
            'j_per_token_bracket': J_PER_TOKEN_BRACKET,
            'pue': PUE,
            'grid_gco2_per_kwh': GRID_GCO2_PER_KWH,
            'input_toks_per_call': INPUT_TOKS_PER_CALL,
            'stop_rag_training_params': STOP_RAG,
        },
        'per_cell': {},
    }

    macro_acc = {name: {
        'f1': [], 'calls': [], 'total_tok': [],
        'Wh': {b: [] for b in J_PER_TOKEN_BRACKET},
        'co2_g': {b: {g: [] for g in GRID_GCO2_PER_KWH} for b in J_PER_TOKEN_BRACKET},
    } for name in RULES}

    for cell, rel in CELLS.items():
        df = prep(ROOT / rel)
        per_rule_df = {name: per_q_stop(df, fn) for name, fn in RULES.items()}
        summary = summarize_cell(per_rule_df)
        out['per_cell'][cell] = summary
        print_cell_table(cell, summary)
        for name, s in summary.items():
            macro_acc[name]['f1'].append(s['f1_mean'])
            macro_acc[name]['calls'].append(s['calls_mean'])
            macro_acc[name]['total_tok'].append(s['total_tokens_mean'])
            for b in J_PER_TOKEN_BRACKET:
                macro_acc[name]['Wh'][b].append(s['energy_Wh_per_q'][b])
                for g in GRID_GCO2_PER_KWH:
                    macro_acc[name]['co2_g'][b][g].append(s['co2_g_per_q'][b][g])

    macro = {}
    for name in RULES:
        macro[name] = {
            'f1_mean':           float(np.mean(macro_acc[name]['f1'])),
            'calls_mean':        float(np.mean(macro_acc[name]['calls'])),
            'total_tokens_mean': float(np.mean(macro_acc[name]['total_tok'])),
            'energy_Wh_per_q':   {b: float(np.mean(macro_acc[name]['Wh'][b])) for b in J_PER_TOKEN_BRACKET},
            'co2_g_per_q':       {b: {g: float(np.mean(macro_acc[name]['co2_g'][b][g]))
                                      for g in GRID_GCO2_PER_KWH}
                                  for b in J_PER_TOKEN_BRACKET},
        }
    out['macro'] = macro

    print(f"\n=== MACRO (across 6 cells) ===")
    print(f"J/token bracket: {J_PER_TOKEN_BRACKET}, PUE = {PUE}, "
          f"grid (default): {DEFAULT_GRID} = {GRID_GCO2_PER_KWH[DEFAULT_GRID]:.0f} gCO2/kWh")
    print(f"{'rule':12s} | {'F1':>5s} | {'calls':>5s} | "
          f"{'kJ low':>7s} | {'kJ mid':>7s} | {'kJ high':>8s} | {'gCO2 mid':>9s}")
    print('-' * 75)
    for name in RULES:
        m = macro[name]
        print(f"{name:12s} | {m['f1_mean']:5.2f} | {m['calls_mean']:5.2f} | "
              f"{m['energy_Wh_per_q']['low']*3.6:7.3f} | "
              f"{m['energy_Wh_per_q']['mid']*3.6:7.3f} | "
              f"{m['energy_Wh_per_q']['high']*3.6:8.3f} | "
              f"{m['co2_g_per_q']['mid'][DEFAULT_GRID]:9.3f}")

    print(f"\n=== Headline savings (macro): AS_m25 vs fixed_k=5 ===")
    sav5 = headline_savings_macro(macro, 'AS_m25', 'fixed_k=5')
    out['headline_AS_m25_vs_fixed5'] = sav5
    print(f"  ΔF1 = {sav5['delta_f1']:+.2f}, Δcalls = {sav5['delta_calls']:+.2f}")
    print(f"  energy ratio (MID J/tok): {sav5['energy_ratio']:.1%}")
    for b in J_PER_TOKEN_BRACKET:
        print(f"  saves {sav5['savings_per_q'][b]*1000:.2f} mWh / Q at {b.upper()} J/tok "
              f"({sav5['co2_saved_kg_per_Mq'][b][DEFAULT_GRID]:+.2f} kg CO2 / 1M-q at default grid)")

    print(f"\n=== Headline savings (macro): AS_m25 vs fixed_k=3 ===")
    sav3 = headline_savings_macro(macro, 'AS_m25', 'fixed_k=3')
    out['headline_AS_m25_vs_fixed3'] = sav3
    print(f"  ΔF1 = {sav3['delta_f1']:+.2f}, Δcalls = {sav3['delta_calls']:+.2f}")
    print(f"  energy ratio (MID J/tok): {sav3['energy_ratio']:.1%}")

    # CO2 sensitivity for AS_m25 across grids
    print(f"\n=== AS_m25 CO2 sensitivity (gCO2 / Q, MID J/tok) ===")
    grid_breakdown = {}
    for g, gi in GRID_GCO2_PER_KWH.items():
        v = macro['AS_m25']['co2_g_per_q'][DEFAULT_BRACKET][g]
        grid_breakdown[g] = v
        print(f"  {g:20s} ({gi:.0f} gCO2/kWh): {v:.3f} gCO2/Q  "
              f"= {v * 1e3:.1f} kg CO2 / 1M-q")
    out['as_m25_co2_per_q_by_grid_mid_bracket'] = grid_breakdown

    # Stop-RAG training envelope
    sr = stop_rag_training_envelope()
    out['stop_rag_training_envelope'] = sr
    print(f"\n=== Stop-RAG training envelope (their hyperparameters) ===")
    print(f"  Total calls: {sr['total_calls']:,}")
    print(f"  Total tokens: {sr['total_tokens']:,}")
    for b in J_PER_TOKEN_BRACKET:
        print(f"  Energy ({b.upper()}): {sr[f'energy_kWh_{b}']:.1f} kWh, "
              f"CO2 (default grid): {sr[f'co2_kg_{b}_{DEFAULT_GRID}']:.1f} kg")

    # Break-even: queries that AS_m25 must serve so the per-query savings vs
    # fixed_k=5 sum to Stop-RAG's training energy. Robust across brackets (the
    # bracket cancels).
    print(f"\n=== Break-even: queries until AS_m25 savings vs fixed_k=5 cover "
          f"Stop-RAG training (MID J/tok) ===")
    Wh_saved_per_q = sav5['savings_per_q'][DEFAULT_BRACKET] * 1000.0  # Wh
    kWh_saved_per_q = Wh_saved_per_q / 1000.0
    sr_train_kWh = sr[f'energy_kWh_{DEFAULT_BRACKET}']
    if kWh_saved_per_q > 0:
        breakeven_q = sr_train_kWh / kWh_saved_per_q
    else:
        breakeven_q = float('inf')
    out['breakeven_q_AS_m25_vs_fixed5_covers_stop_rag_training'] = breakeven_q
    print(f"  Stop-RAG training: {sr_train_kWh:.1f} kWh")
    print(f"  AS_m25 savings vs fixed_k=5: {Wh_saved_per_q*1000:.2f} mWh / Q")
    print(f"  Break-even: {breakeven_q:,.0f} queries")
    print(f"  (Ratio robust to J/tok choice; the bracket cancels in num and denom.)")

    print(f"\n=== Two robust ratios (independent of J/tok choice) ===")
    # (a) per-query energy ratio AS_m25 / fixed_k=5 = tokens ratio (since
    # energy = const * tokens for both methods)
    tok_ratio_h_b = macro['AS_m25']['total_tokens_mean'] / macro['fixed_k=5']['total_tokens_mean']
    print(f"  (a) AS_m25 / fixed_k=5 energy ratio per query = "
          f"{tok_ratio_h_b:.4f}  (= total-tokens ratio, robust)")
    # (b) break-even queries already computed; also independent of J/tok
    print(f"  (b) Training-amortization break-even: {breakeven_q:,.0f} queries "
          f"(robust)")

    target = ROOT / 'results' / 'energy_cost_table.json'
    target.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {target}")


if __name__ == '__main__':
    main()
