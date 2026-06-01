"""Answer-based signals from existing per-round parquets. No LLM calls.

Adds four columns to the enriched parquet, all deterministic:
- answer_stable:        1 iff normalized current_answer == normalized previous
                        round's answer (same qid). Round 1 is 0 by construction.
- answer_stable_f1:     HotpotQA-style token F1 in [0, 1] between this round's
                        normalized answer and the previous round's. Continuous
                        relaxation of answer_stable. Round 1 is 0 by construction.
- answer_stable_seq:    difflib SequenceMatcher.ratio() in [0, 1] between this
                        round's normalized answer and the previous round's.
                        Character-level similarity. Round 1 is 0 by construction.
- answer_in_evidence:   1 iff the normalized answer appears as a substring of
                        the concatenated top-`round` BM25 paragraphs.

Reports Cohen's d for each new signal (correct vs incorrect on EM) and a
per-round breakdown. Overwrites the enriched parquet in place.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval import build_paragraphs, retrieve  # noqa: E402


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (a.std() + b.std()) / 2 + 1e-9
    return abs(a.mean() - b.mean()) / pooled


def _token_f1(pred_norm: str, gold_norm: str) -> float:
    """HotpotQA-style token F1 between two already-normalized strings.
    Empty in either side returns 0 (matches the official metric)."""
    p_toks = pred_norm.split()
    g_toks = gold_norm.split()
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _seq_ratio(a: str, b: str) -> float:
    """difflib SequenceMatcher.ratio() between two normalized strings.
    Empty in either side returns 0."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def load_question_paragraphs(data_path: str) -> dict[str, list[dict]]:
    with open(data_path) as f:
        items = json.load(f)
    out: dict[str, list[dict]] = {}
    for item in items:
        paragraphs = build_paragraphs(item)
        ranked = retrieve(item["question"], paragraphs, k=len(paragraphs))
        out[item["id"]] = ranked
    return out


def add_signals(df: pd.DataFrame, paragraphs_by_qid: dict) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy().reset_index(drop=True)

    norm_ans = df["current_answer"].apply(normalize)
    prev_qid = df["qid"].shift(1)
    prev_ans = norm_ans.shift(1).fillna("")
    same_qid = (df["qid"] == prev_qid).values
    df["answer_stable"] = ((norm_ans == prev_ans) & same_qid).astype(int)

    f1_arr = np.zeros(len(df), dtype=float)
    seq_arr = np.zeros(len(df), dtype=float)
    norm_ans_vals = norm_ans.values
    prev_ans_vals = prev_ans.values
    for i in range(len(df)):
        if not same_qid[i]:
            continue
        f1_arr[i] = _token_f1(norm_ans_vals[i], prev_ans_vals[i])
        seq_arr[i] = _seq_ratio(norm_ans_vals[i], prev_ans_vals[i])
    df["answer_stable_f1"] = f1_arr
    df["answer_stable_seq"] = seq_arr

    # Pre-normalize evidence text per (qid, round). We reuse: for a given qid,
    # evidence_text[r] = paragraphs[0..r-1] concatenated and normalized.
    grounded = np.zeros(len(df), dtype=int)
    norm_evidence_cache: dict[tuple[str, int], str] = {}
    for qid, paragraphs in paragraphs_by_qid.items():
        running = ""
        for r in range(1, len(paragraphs) + 1):
            running = (running + " " + paragraphs[r - 1]["text"]) if running else paragraphs[r - 1]["text"]
            norm_evidence_cache[(qid, r)] = normalize(running)

    for i, row in df.iterrows():
        ans = norm_ans.iloc[i]
        if not ans:
            continue
        ev = norm_evidence_cache.get((row["qid"], int(row["round"])))
        if ev and ans in ev:
            grounded[i] = 1
    df["answer_in_evidence"] = grounded
    return df


def report_separation(df: pd.DataFrame, signals: list[str], label: str) -> None:
    print(f"\n[{label}] Cohen's d for new signals (correct vs incorrect on EM):")
    print(f"  {'signal':22s} {'d':>6s}  {'mean(correct)':>14s}  {'mean(incorrect)':>16s}  {'pct=1':>7s}")
    for s in signals:
        cor = df[df["current_em"] == 1][s].astype(float)
        inc = df[df["current_em"] == 0][s].astype(float)
        d = cohens_d(cor, inc)
        pct = float(df[s].mean()) * 100
        print(f"  {s:22s} {d:6.3f}  {cor.mean():>14.4f}  {inc.mean():>16.4f}  {pct:6.1f}%")
    print(f"\n[{label}] Per-round means:")
    print(df.groupby("round")[signals].mean().round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-parquet", default="results/signal_log_enriched.parquet")
    ap.add_argument("--tune-parquet", default="results/signal_log_enriched_tune.parquet")
    ap.add_argument("--data-eval", default="data/dev_eval.json")
    ap.add_argument("--data-tune", default="data/dev_tune.json")
    args = ap.parse_args()

    print("Building BM25 rankings per qid (cached eval + tune)...")
    eval_pars = load_question_paragraphs(args.data_eval)
    tune_pars = load_question_paragraphs(args.data_tune)
    print(f"  eval: {len(eval_pars)} qids   tune: {len(tune_pars)} qids")

    for label, path, pars in [
        ("eval", args.eval_parquet, eval_pars),
        ("tune", args.tune_parquet, tune_pars),
    ]:
        print(f"\n{'=' * 70}\nProcessing {label}: {path}\n{'=' * 70}")
        df = pd.read_parquet(path)
        df = add_signals(df, pars)
        report_separation(
            df,
            ["answer_stable", "answer_stable_f1", "answer_stable_seq", "answer_in_evidence"],
            label,
        )
        df.to_parquet(path, index=False)
        print(f"\nWrote {path} ({df.shape})")


if __name__ == "__main__":
    main()
