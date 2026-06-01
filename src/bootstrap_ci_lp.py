"""Paired bootstrap CIs for LP-parquet stopping rules vs fixed-k baselines.

Supports either:
    family A JSON: {family: "A", t_margin, t_overlap}
    family B JSON: {family: "B", t_conf, t_margin, t_overlap}
    plain-text expressions: answer_stable == 1

If the rule expression references ``answer_stable`` and the parquet does not
already contain that column, it is computed inline from ``current_answer``.

Same paired-resampling approach as bootstrap_ci.py.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_ci import paired_bootstrap, per_question_fixed_k  # noqa: E402


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def add_answer_stable(df: pd.DataFrame) -> pd.DataFrame:
    if "answer_stable" in df.columns:
        return df
    if "current_answer" not in df.columns:
        raise KeyError("Rule references answer_stable but current_answer is missing from the parquet")

    df = df.sort_values(["qid", "round"]).copy()
    df["_norm"] = df["current_answer"].map(normalize)
    df["_prev"] = df.groupby("qid")["_norm"].shift(1)
    df["answer_stable"] = ((df["_prev"].notna()) & (df["_prev"] == df["_norm"])).astype(int)
    return df


def per_question_rule_A(df: pd.DataFrame, t_margin: float, t_overlap: float) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = (df["calibrated_logit_margin"] > t_margin) | (df["overlap_signal"] > t_overlap)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def per_question_rule_B(df: pd.DataFrame, t_conf: float, t_margin: float, t_overlap: float) -> pd.DataFrame:
    df = df.sort_values(["qid", "round"]).copy()
    df["_stop"] = (
        (df["calibrated_conf"] > t_conf)
        | (df["calibrated_logit_margin"] > t_margin)
        | (df["overlap_signal"] > t_overlap)
    )
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def per_question_rule_expr(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if re.search(r"\banswer_stable\b", rule) and "answer_stable" not in df.columns:
        df = add_answer_stable(df)
    else:
        df = df.sort_values(["qid", "round"]).copy()

    df["_stop"] = df.eval(rule)
    stops = df[df["_stop"]].groupby("qid", sort=False).head(1)
    stopped_qids = set(stops["qid"])
    fallback = df[~df["qid"].isin(stopped_qids)].groupby("qid", sort=False).tail(1)
    chosen = pd.concat([stops, fallback], axis=0)
    return chosen[["qid", "current_em", "current_f1", "round"]].sort_values("qid").reset_index(drop=True)


def load_locked_rule(path: Path) -> dict:
    raw = path.read_text().strip()
    if not raw:
        raise ValueError(f"Locked rule file is empty: {path}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"family": "expr", "rule": raw}

    if isinstance(parsed, str):
        return {"family": "expr", "rule": parsed.strip()}
    if not isinstance(parsed, dict):
        raise ValueError(f"Locked rule file must contain a JSON object or plain-text rule: {path}")
    return parsed


def describe_rule(locked: dict) -> str:
    family = locked.get("family")
    if locked.get("rule"):
        return str(locked["rule"])
    if family == "A":
        return (
            f"(calibrated_logit_margin > {locked['t_margin']}) or "
            f"(overlap_signal > {locked['t_overlap']})"
        )
    if family == "B":
        return (
            f"(calibrated_conf > {locked['t_conf']}) or "
            f"(calibrated_logit_margin > {locked['t_margin']}) or "
            f"(overlap_signal > {locked['t_overlap']})"
        )
    raise ValueError(f"Locked rule is missing a rule string: {locked}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/signal_log_lp_enriched.parquet")
    ap.add_argument("--locked-rule", default="results/locked_rule_lp.txt")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/bootstrap_ci_lp.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.eval)
    locked = load_locked_rule(Path(args.locked_rule))
    family = locked.get("family", "expr")
    rule = describe_rule(locked)
    print("=" * 72)
    print(f"Paired bootstrap CI on LP variant   ({args.n_boot} resamples, seed={args.seed})")
    print(f"  family: {family}")
    print(f"  rule:   {rule}")
    print(f"  eval:   {args.eval}  n_questions={df['qid'].nunique()}")
    print("=" * 72)

    if family == "A":
        ours = per_question_rule_A(df, locked["t_margin"], locked["t_overlap"])
    elif family == "B":
        ours = per_question_rule_B(df, locked["t_conf"], locked["t_margin"], locked["t_overlap"])
    elif family == "expr":
        ours = per_question_rule_expr(df, rule)
    else:
        raise ValueError(f"Unknown rule family in locked_rule_lp.txt: {family}")

    out = {
        "family": family,
        "rule": rule,
        "n_questions": int(df["qid"].nunique()),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "ours_em_mean": round(float(ours["current_em"].mean() * 100), 3),
        "ours_f1_mean": round(float(ours["current_f1"].mean() * 100), 3),
        "ours_avg_calls": round(float(ours["round"].mean()), 3),
        "comparisons": {},
    }

    for k in [1, 2, 3, 5]:
        base = per_question_fixed_k(df, k)
        res = paired_bootstrap(ours, base, args.n_boot, args.seed)
        out["comparisons"][f"fixed_k={k}"] = res
        print(f"\nvs fixed-k={k}:")
        print(f"  F1 (ours)         {res['ours_f1_mean']:6.2f}  CI95 {res['ours_f1_ci95']}")
        print(f"  F1 (k={k})          {res['base_f1_mean']:6.2f}  CI95 {res['base_f1_ci95']}")
        print(f"  diff_F1 (ours-k)  {res['diff_f1_mean']:+6.2f}  CI95 {res['diff_f1_ci95']}")
        print(f"  diff_EM (ours-k)  {res['diff_em_mean']:+6.2f}  CI95 {res['diff_em_ci95']}")
        print(f"  diff_calls        {res['diff_calls_mean']:+6.2f}  CI95 {res['diff_calls_ci95']}")
        lo, hi = res["diff_f1_ci95"]
        if lo > 0:
            verdict = f"SIGNIFICANT WIN on F1 (lower bound +{lo:.2f})"
        elif hi < 0:
            verdict = f"SIGNIFICANT LOSS on F1 (upper bound {hi:.2f})"
        else:
            verdict = "NOT SIGNIFICANT on F1 (CI spans 0)"
        print(f"  F1 verdict:       {verdict}")
        print(f"  P(ours worse on F1) = {res['p_ours_worse_f1']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
