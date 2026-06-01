"""Accuracy-cost frontier plots: distractor cells and open-domain cells.

Two output figures:
  - guardrail_frontier_distractor.{pdf,png}: macro across 6 distractor cells
    (3 models x {HotpotQA, 2Wiki}), from results/headline_table.json
  - guardrail_frontier_open.{pdf,png}: macro across 7 open-domain cells
    (qwen x {fullwiki, NQ, trivia}, devstral/gemma x {fullwiki, NQ}),
    from results/wiki_replay_table.json

Rules plotted: fixed-k anchors, AS, AS + margin>0.20, AS_m25 (headline), oracle.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DISTRACTOR_TABLE = ROOT / "results" / "headline_table.json"
OPEN_TABLE = ROOT / "results" / "wiki_replay_table.json"
FIG_DIR = ROOT / "figures"

RULE_STYLE = {
    "fixed_k=1":  dict(marker="o", color="#888", label="fixed-k"),
    "fixed_k=3":  dict(marker="o", color="#888"),
    "fixed_k=5":  dict(marker="o", color="#888"),
    "AS":         dict(marker="^", color="#1f77b4", label="AS (stability only)"),
    "AS_m25":     dict(marker="*", color="#d62728", label="AS_m25 (headline)", s=180),
    "oracle":     dict(marker="x", color="#444", label="oracle"),
}


def plot_frontier(point: dict, title: str, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ks = [k for k in ["fixed_k=1", "fixed_k=3", "fixed_k=5"] if k in point]
    xs = [point[k]["calls"] for k in ks]
    ys = [point[k]["f1"] for k in ks]
    ax.plot(xs, ys, color="#888", linestyle=":", linewidth=1, zorder=1)

    all_xs, all_ys = [], []
    for rule, style in RULE_STYLE.items():
        if rule not in point:
            continue
        x, y = point[rule]["calls"], point[rule]["f1"]
        all_xs.append(x)
        all_ys.append(y)
        edge_kwargs = {"edgecolor": "black", "linewidth": 0.5} if style["marker"] != "x" else {}
        ax.scatter([x], [y], s=style.get("s", 80), marker=style["marker"],
                   color=style["color"], label=style.get("label"), zorder=3, **edge_kwargs)
        ax.annotate(rule, (x, y), xytext=(x + 0.05, y + 0.3), fontsize=8)

    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    x_pad = max(0.6, (x_max - x_min) * 0.12)
    y_pad = max(3.0, (y_max - y_min) * 0.12)
    ax.set_xlim(x_min - x_pad * 0.4, x_max + x_pad)
    ax.set_ylim(y_min - y_pad * 0.4, y_max + y_pad)

    ax.set_xlabel("Average calls per question")
    ax.set_ylabel("F1 (per-question mean x 100)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    out_pdf = out_stem.with_suffix(".pdf")
    out_png = out_stem.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def macro_from_open(table: dict) -> dict:
    """Average F1 and calls across every (model, corpus) cell in wiki_replay_table.json."""
    cells = [(m, c, rules["rules"]) for m, corpora in table.items() for c, rules in corpora.items()]
    rule_names = set()
    for _, _, rules in cells:
        rule_names.update(rules.keys())

    point: dict[str, dict] = {}
    for name in rule_names:
        f1s, calls = [], []
        for _, _, rules in cells:
            if name in rules:
                f1s.append(rules[name]["f1"])
                calls.append(rules[name]["calls"])
        if f1s:
            point[name] = {"f1": sum(f1s) / len(f1s), "calls": sum(calls) / len(calls)}
    return point


def main() -> None:
    distractor = json.loads(DISTRACTOR_TABLE.read_text())
    if "macro" not in distractor:
        raise SystemExit("headline_table.json missing 'macro' key")
    plot_frontier(
        distractor["macro"]["point"],
        "Distractor cells: macro across 3 models x {HotpotQA, 2Wiki}",
        FIG_DIR / "guardrail_frontier_distractor",
    )

    open_table = json.loads(OPEN_TABLE.read_text())
    open_point = macro_from_open(open_table)
    n_cells = sum(len(corpora) for corpora in open_table.values())
    plot_frontier(
        open_point,
        f"Open-domain cells: macro across {n_cells} (model x corpus) cells",
        FIG_DIR / "guardrail_frontier_open",
    )


if __name__ == "__main__":
    main()
