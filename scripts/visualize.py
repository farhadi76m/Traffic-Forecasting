#!/usr/bin/env python3
"""Visualize model predictions: network heatmaps, edge profiles, accuracy.

Produces four PNGs in --out-dir:
  network_traffic.png   map colored by predicted vehicles/hour at key hours
  edge_profiles.png     24h predicted vs simulated profiles, busiest edges
  pred_vs_actual.png    prediction vs typical-day simulated traffic
  daily_profile.png     network-wide traffic volume over the day

Example:
    python scripts/visualize.py --net sumo/tehran_2026_area.net.xml \
        --data-dir output/dataset --model-dir output/model --out-dir output/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sumolib
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from predict import load_predictor

# palette (light surface)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, AQUA = "#2a78d6", "#1baf7a"  # predicted, simulated
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hours", type=int, nargs="+", default=[7, 13, 18, 23])
    p.add_argument("--top-edges", type=int, default=6)
    return p.parse_args()


def fig_network(net, meta, preds, hours, out):
    shapes = [net.getEdge(e).getShape() for e in meta["edges"]]
    vmax = np.percentile(preds[:, hours], 99.5)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=max(vmax, 1))
    # frame the region that actually carries traffic, not the full bbox
    act = np.flatnonzero(preds.max(1) >= 0.5)
    xs = [p[0] for i in act for p in shapes[i]]
    ys = [p[1] for i in act for p in shapes[i]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mx, my = 0.03 * (x1 - x0), 0.03 * (y1 - y0)
    aspect = (y1 - y0 + 2 * my) / (x1 - x0 + 2 * mx)
    ncol = 2
    nrow = int(np.ceil(len(hours) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 5.2 * aspect * nrow + 1.5))
    for ax, h in zip(np.ravel(axes), hours):
        v = preds[:, h]
        quiet = v < 0.5
        ax.add_collection(LineCollection(
            [shapes[i] for i in np.flatnonzero(quiet)],
            colors=GRID, linewidths=0.5))
        busy = np.flatnonzero(~quiet)
        busy = busy[np.argsort(v[busy])]  # draw heaviest last, on top
        ax.add_collection(LineCollection(
            [shapes[i] for i in busy],
            colors=CMAP(norm(v[busy])), linewidths=0.6 + 1.9 * norm(v[busy])))
        ax.set_xlim(x0 - mx, x1 + mx)
        ax.set_ylim(y0 - my, y1 + my)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{h:02d}:00", loc="left", color=INK, fontweight="bold")
    for ax in np.ravel(axes)[len(hours):]:
        ax.set_visible(False)
    fig.suptitle("Estimated traffic across the network", x=0.06, y=0.99,
                 ha="left", fontsize=13, fontweight="bold", color=INK)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=np.ravel(axes).tolist(), orientation="horizontal",
                      fraction=0.03, pad=0.03, aspect=45)
    cb.set_label("vehicles / hour", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_profiles(data, meta, preds, k, out):
    idx = {e: i for i, e in enumerate(meta["edges"])}
    top = (data.groupby("edge")["count"].sum().sort_values(ascending=False)
           .head(k).index.tolist())
    ncol, nrow = 3, int(np.ceil(k / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.1 * nrow),
                             sharex=True, sharey=True)
    hours = np.arange(24)
    for n, (ax, e) in enumerate(zip(np.ravel(axes), top)):
        piv = (data[data["edge"] == e]
               .pivot(index="scenario", columns="hour", values="count")
               .to_numpy())
        p10, p50, p90 = np.percentile(piv, [10, 50, 90], axis=0)
        ax.fill_between(hours, p10, p90, color=AQUA, alpha=0.16, linewidth=0)
        ax.plot(hours, p50, color=AQUA, lw=2, label="simulated (median, 10-90%)")
        ax.plot(hours, preds[idx[e]], color=BLUE, lw=2, label="predicted")
        ax.set_title(e, loc="left", fontsize=9, color=INK2)
        ax.set_xticks([0, 6, 12, 18, 23])
        if n == 0:
            ax.text(0.03, 0.9, "predicted", color=BLUE, fontsize=9,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(0.03, 0.78, "simulated", color=AQUA, fontsize=9,
                    fontweight="bold", transform=ax.transAxes)
    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False,
               labelcolor=INK2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Busiest edges: 24h traffic profile", x=0.06, y=1.0,
                 ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.supxlabel("hour of day", color=INK2, fontsize=10)
    fig.supylabel("vehicles / hour", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_scatter(data, meta, preds, out):
    idx = {e: i for i, e in enumerate(meta["edges"])}
    typ = data.groupby(["edge", "hour"])["count"].mean().reset_index()
    y = typ["count"].to_numpy()
    x = preds[typ["edge"].map(idx).to_numpy(), typ["hour"].to_numpy()]
    if len(x) > 25000:
        sel = np.random.default_rng(0).choice(len(x), 25000, replace=False)
        x, y = x[sel], y[sel]
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    lim = max(x.max(), y.max()) * 1.2 + 1
    ax.plot([1, lim], [1, lim], ls="--", lw=1, color=BASELINE, zorder=1)
    ax.scatter(x + 1, y + 1, s=6, color=BLUE, alpha=0.2, linewidths=0, zorder=2)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.9, lim), ax.set_ylim(0.9, lim)
    ax.set_xlabel("predicted vehicles/hour (+1, log)")
    ax.set_ylabel("simulated typical day (+1, log)")
    ax.set_title("Predicted vs typical-day traffic", loc="left",
                 fontsize=13, fontweight="bold", color=INK)
    m = meta.get("metrics", {}).get("test_typical")
    if m:
        ax.text(0.03, 0.97,
                f"held-out days:  accuracy {m['accuracy']*100:.1f}%   "
                f"R² {m['r2']:.3f}   MAE {m['mae']:.2f}",
                transform=ax.transAxes, va="top", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_daily(data, out):
    tot = data.groupby(["scenario", "hour"])["count"].sum().unstack().to_numpy()
    hours = np.arange(24)
    p10, p90 = np.percentile(tot, [10, 90], axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(hours, p10, p90, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(hours, tot.mean(0), color=BLUE, lw=2)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 23])
    ax.set_xlabel("hour of day")
    ax.set_ylabel("vehicle entries / hour (all edges)")
    ax.set_title("Network-wide daily traffic (mean, 10-90% over days)",
                 loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    net = sumolib.net.readNet(args.net)
    data = pd.read_csv(os.path.join(args.data_dir, "traffic.csv.gz"))
    meta, predict = load_predictor(args.model_dir)
    preds = predict(np.arange(len(meta["edges"])))

    fig_network(net, meta, preds, args.hours,
                os.path.join(args.out_dir, "network_traffic.png"))
    fig_profiles(data, meta, preds, args.top_edges,
                 os.path.join(args.out_dir, "edge_profiles.png"))
    fig_scatter(data, meta, preds,
                os.path.join(args.out_dir, "pred_vs_actual.png"))
    fig_daily(data, os.path.join(args.out_dir, "daily_profile.png"))
    print(f"wrote 4 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
