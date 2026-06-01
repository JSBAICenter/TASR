"""Qualitative dilution analysis on the LP parquet.

For each evaluation question, computes:
  - the round at which the answer-stable rule first fires (or 5 if it never does)
  - the per-round (EM, F1, answer) trajectory
  - whether retrieval dilution occurs: the model was correct at some round r<5
    but wrong at r=5 (`em_dilution`)
  - the F1 saved by stopping at AS-fire instead of running to k=5 (`as_saves_f1`)

Writes:
  - results/qualitative/dilution_per_qid.csv      (every qid, full per-round trace)
  - results/qualitative/dilution_examples.json    (top examples with question + gold)
  - results/qualitative/dilution_summary.json     (population counts)

No fresh LLM calls. Pure post-hoc analysis on the cached lp parquet.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import pandas as pd


DEFAULT_PARQUET = "results/signal_log_lp_enriched.parquet"
DEFAULT_EVAL_JSON = "data/dev_eval.json"
DEFAULT_OUT_DIR = "results/qualitative"


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--eval-json", default=DEFAULT_EVAL_JSON)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    df = pd.read_parquet(args.parquet).sort_values(["qid", "round"]).reset_index(drop=True)
    df["_norm"] = df["current_answer"].map(normalize)
    df["_prev"] = df.groupby("qid")["_norm"].shift(1)
    df["answer_stable"] = ((df["_prev"].notna()) & (df["_prev"] == df["_norm"])).astype(int)

    eval_items = {x["id"]: x for x in json.load(open(args.eval_json))}

    rows = []
    for qid, g in df.groupby("qid"):
        g = g.sort_values("round")
        em_seq = [float(g[g["round"] == r]["current_em"].iloc[0]) for r in [1, 2, 3, 4, 5]]
        f1_seq = [float(g[g["round"] == r]["current_f1"].iloc[0]) for r in [1, 2, 3, 4, 5]]
        ans_seq = [g[g["round"] == r]["current_answer"].iloc[0] for r in [1, 2, 3, 4, 5]]
        as_fires = g[g["answer_stable"] == 1]["round"].tolist()
        as_stop = int(as_fires[0]) if as_fires else 5
        as_em = em_seq[as_stop - 1]
        as_f1 = f1_seq[as_stop - 1]
        as_ans = ans_seq[as_stop - 1]
        final_em, final_f1, final_ans = em_seq[4], f1_seq[4], ans_seq[4]
        em_dilution = any(em_seq[r] == 1.0 for r in range(4)) and em_seq[4] == 0.0
        as_saves_em = int(as_em == 1.0 and final_em == 0.0)
        as_saves_f1 = round(as_f1 - final_f1, 4)
        item = eval_items.get(qid, {})
        rows.append({
            "qid": qid,
            "question_type": item.get("type", ""),
            "level": item.get("level", ""),
            "question": item.get("question", ""),
            "gold_answer": item.get("answer", ""),
            "as_stop": as_stop,
            "as_answer": as_ans,
            "as_em": as_em,
            "as_f1": round(as_f1, 4),
            "final_answer": final_ans,
            "final_em": final_em,
            "final_f1": round(final_f1, 4),
            "em_seq": em_seq,
            "f1_seq": [round(x, 4) for x in f1_seq],
            "ans_seq": ans_seq,
            "em_dilution": int(em_dilution),
            "as_saves_em": as_saves_em,
            "as_saves_f1": as_saves_f1,
        })
    out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "dilution_per_qid.csv", index=False)

    n_total = len(out)
    n_em_dilution = int(out["em_dilution"].sum())
    n_as_saves_em = int(out["as_saves_em"].sum())
    n_f1_drop = int((out["as_saves_f1"] >= 0.5).sum())
    f1_savings_sum = float(out["as_saves_f1"].sum())
    f1_savings_mean = float(out["as_saves_f1"].mean())

    # Among questions where AS fires before r=5, what is the dilution profile?
    fires_early = out[out["as_stop"] < 5]
    pre5_em_dilution = int(((fires_early["as_em"] == 1.0) & (fires_early["final_em"] == 0.0)).sum())
    pre5_em_loss = int(((fires_early["as_em"] == 0.0) & (fires_early["final_em"] == 1.0)).sum())

    summary = {
        "n_total": n_total,
        "n_em_dilution_overall": n_em_dilution,
        "n_as_saves_em": n_as_saves_em,
        "n_f1_drop_ge_0p5": n_f1_drop,
        "f1_savings_sum_300q": round(f1_savings_sum, 4),
        "f1_savings_mean_per_q": round(f1_savings_mean, 4),
        "n_as_fires_early_lt_r5": int(len(fires_early)),
        "pre5_em_dilution_avoided": pre5_em_dilution,
        "pre5_em_loss_caused": pre5_em_loss,
    }
    (out_dir / "dilution_summary.json").write_text(json.dumps(summary, indent=2))

    # Top examples for the paper: keep dilution-avoidance wins and dilution-cause losses.
    wins = out[out["as_saves_em"] == 1].sort_values(["as_stop", "qid"]).to_dict(orient="records")
    losses = out[(out["as_em"] == 0.0) & (out["final_em"] == 1.0) & (out["as_stop"] < 5)] \
        .sort_values(["as_stop", "qid"]).to_dict(orient="records")
    ex = {
        "as_stops_before_dilution": wins,
        "as_stops_too_early_correct_lost": losses,
    }
    (out_dir / "dilution_examples.json").write_text(json.dumps(ex, indent=2, default=str))

    print(json.dumps(summary, indent=2))
    print(f"\nWins (AS stops before dilution): {len(wins)}")
    for r in wins[:6]:
        print(
            f"  {r['qid']}  AS@r{r['as_stop']}='{r['as_answer']}'  "
            f"vs k=5='{r['final_answer']}'  gold='{r['gold_answer']}'"
        )
    print(f"\nLosses (AS stops too early on a correct-later question): {len(losses)}")
    for r in losses[:6]:
        print(
            f"  {r['qid']}  AS@r{r['as_stop']}='{r['as_answer']}'  "
            f"vs k=5='{r['final_answer']}'  gold='{r['gold_answer']}'"
        )


if __name__ == "__main__":
    main()
