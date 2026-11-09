#!/usr/bin/env python3
"""Turns experiments/results/*.csv (from bench.sh) into PNGs in experiments/figures/.

  speedup.png        delta-stepping speedup over sequential Dijkstra vs ranks
  delta_sweep.png    runtime vs delta at 1 and 6 ranks
  communication.png  bytes sent between ranks vs ranks
"""
import csv
import os
from collections import defaultdict
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

# graph name -> (label, color); fixed so a graph keeps its color in every chart
GRAPHS = {
    "rmat20":    ("R-MAT scale 20",            "#2a78d6"),
    "grid":      ("Grid 1000x1000",            "#eb6834"),
    "grid_shuf": ("Grid 1000x1000, shuffled",  "#1baf7a"),
}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def load(name):
    path = os.path.join(RESULTS, name + ".csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def seq_times(rows):
    """median sequential time per source"""
    by_src = defaultdict(list)
    for r in rows:
        if r["algo"] == "seq":
            by_src[r["source"]].append(float(r["seconds"]))
    return {s: median(t) for s, t in by_src.items()}


def delta_rows(rows, **match):
    return [r for r in rows if r["algo"] == "delta"
            and all(r[k] == str(v) for k, v in match.items())]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)
    ax.plot([x], [y], "o", color=color, markersize=6, markeredgecolor="white")


def plot_speedup():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (label, color) in GRAPHS.items():
        rows = load(name)
        seq = seq_times(rows)
        if not rows or not seq:
            continue
        ranks = sorted({int(r["ranks"]) for r in delta_rows(rows)})
        deltas = {r["delta"] for r in delta_rows(rows)}
        if len(deltas) != 1:
            raise SystemExit(f"{name}.csv mixes deltas {sorted(deltas)}; use one per scaling file")
        meds, lo, hi = [], [], []
        for p in ranks:
            # speedup per source (same source for both), then the spread
            s = [seq[r["source"]] / float(r["seconds"]) for r in delta_rows(rows, ranks=p)]
            meds.append(median(s)); lo.append(min(s)); hi.append(max(s))
        ax.fill_between(ranks, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(ranks, meds, "-o", color=color, linewidth=2, markersize=6,
                markeredgecolor="white", label=f"{label} (delta = {deltas.pop()})")
        label_end(ax, ranks[-1], meds[-1], f"{meds[-1]:.2f}x", color)
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("sequential Dijkstra", (6, 1.0), xytext=(0, -12), textcoords="offset points",
                ha="right", color=MUTED, fontsize=9)
    ax.set_xticks(ranks)
    style(ax, "Delta-stepping speedup over sequential Dijkstra",
          "MPI ranks (one per core)", "speedup (median, band = min-max)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "speedup.png")


def plot_delta_sweep():
    panels = [n for n in ("rmat20", "grid") if load("sweep_" + n)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, name in zip(axes[0], panels):
        rows = load("sweep_" + name)
        label, color = GRAPHS[name]
        for p, ls in ((1, ":"), (6, "-")):
            deltas = sorted({int(r["delta"]) for r in delta_rows(rows, ranks=p)})
            if not deltas:
                continue
            t = [median(float(r["seconds"]) for r in delta_rows(rows, ranks=p, delta=d))
                 for d in deltas]
            ax.plot(deltas, t, ls, marker="o", color=color, linewidth=2, markersize=6,
                    markeredgecolor="white", label=f"{p} rank{'s' if p > 1 else ''}")
        seq = median(seq_times(rows).values())
        ax.axhline(seq, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate("sequential", (deltas[0], seq), xytext=(0, 4),
                    textcoords="offset points", color=MUTED, fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(deltas)
        ax.set_xticklabels([str(d) for d in deltas])
        ax.minorticks_off()
        style(ax, label, "delta (bucket width, log scale)", "seconds (median)")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
    fig.suptitle("Small delta = more rounds; large delta = more wasted relaxations",
                 color=INK, fontsize=11, x=0.01, ha="left")
    save(fig, "delta_sweep.png")


def plot_communication():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (label, color) in GRAPHS.items():
        rows = [r for r in delta_rows(load(name)) if int(r["ranks"]) > 1]
        if not rows:
            continue
        ranks = sorted({int(r["ranks"]) for r in rows})
        mb = [median(int(r["bytes_sent"]) for r in rows if int(r["ranks"]) == p) / 1e6
              for p in ranks]
        ax.plot(ranks, mb, "-o", color=color, linewidth=2, markersize=6,
                markeredgecolor="white", label=label)
        label_end(ax, ranks[-1], mb[-1], f"{mb[-1]:.1f} MB", color)
        ax.set_xticks(ranks)
    style(ax, "Data sent between ranks per run",
          "MPI ranks", "MB sent (median)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "communication.png")


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, name), dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join("experiments", "figures", name))


if __name__ == "__main__":
    plot_speedup()
    plot_delta_sweep()
    plot_communication()
