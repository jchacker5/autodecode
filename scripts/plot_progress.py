#!/usr/bin/env python3
"""Generate progress chart from results/experiments.tsv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "results" / "experiments.tsv"
OUT = ROOT / "docs" / "progress.png"


def main() -> None:
    rows: list[dict[str, str]] = []
    with TSV.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    keeps = [r for r in rows if r["status"] == "keep"]
    exp_nums = list(range(1, len(rows) + 1))
    all_scores = [float(r["decode_tok_s"]) for r in rows]
    statuses = [r["status"] for r in rows]

    best_so_far: list[float] = []
    running = 0.0
    for r in rows:
        score = float(r["decode_tok_s"])
        if r["status"] == "keep" and score > running:
            running = score
        best_so_far.append(running)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#0d1117")

    # Milestones bar
    milestones = [
        ("Ollama Q8", 10.0, "#6e7681"),
        ("MLX baseline", 17.5, "#58a6ff"),
        ("autodecode", 21.395, "#3fb950"),
    ]
    names, vals, colors = zip(*milestones)
    bars = ax1.barh(names, vals, color=colors, height=0.55, edgecolor="#30363d")
    ax1.set_xlim(0, 24)
    ax1.set_xlabel("decode tok/s (higher = faster)", color="#c9d1d9")
    ax1.set_title("Ornith-9B on M4 MacBook Pro · 32 GB", color="#f0f6fc", fontsize=14, pad=12)
    ax1.tick_params(colors="#8b949e")
    ax1.set_facecolor("#161b22")
    for spine in ax1.spines.values():
        spine.set_color("#30363d")
    for bar, val in zip(bars, vals):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", color="#f0f6fc", fontsize=11)

    # Experiment scatter + best-so-far line
    discard_x = [i for i, s in enumerate(exp_nums) if statuses[i - 1] == "discard"]
    discard_y = [all_scores[i - 1] for i in discard_x]
    keep_x = [i for i, s in enumerate(exp_nums) if statuses[i - 1] == "keep"]
    keep_y = [all_scores[i - 1] for i in keep_x]

    ax2.scatter(discard_x, discard_y, c="#f85149", s=36, alpha=0.7, label="discard", zorder=3)
    ax2.scatter(keep_x, keep_y, c="#3fb950", s=52, alpha=0.9, label="keep", zorder=4)
    ax2.plot(exp_nums, best_so_far, color="#d2a8ff", linewidth=2.2, label="best so far", zorder=2)
    ax2.axhline(17.453, color="#58a6ff", linestyle="--", alpha=0.5, linewidth=1, label="stable MLX baseline")
    ax2.set_xlabel("experiment #", color="#c9d1d9")
    ax2.set_ylabel("decode tok/s", color="#c9d1d9")
    ax2.set_title("22 agent experiments · 1 session", color="#f0f6fc", fontsize=12)
    ax2.set_facecolor("#161b22")
    ax2.tick_params(colors="#8b949e")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#30363d")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, facecolor=fig.get_facecolor())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()