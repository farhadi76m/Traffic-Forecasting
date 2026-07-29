#!/usr/bin/env python3
"""Snapshot a TAZ partition: zone polygons over the road network, in lon/lat.

Reads the *_districts.geojson + *_network.geojson pair written by
build_taz_districts.py for one output folder and renders a single map image,
used as the per-level slide picture in the TAZ report (make_taz_report.py).

    python scripts/plot_taz_snapshot.py neshan/b_tehran_l9 --out out.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
ROAD_MAJOR, ROAD_MINOR = "#5b5a56", "#c3c2b7"
PALETTE = [
    "#e76c5a", "#2e86c1", "#f1c40f", "#27ae60", "#9b59b6", "#e67e22",
    "#16a085", "#34495e", "#c0392b", "#1a5276", "#d35400", "#7f8c8d",
    "#48c9b0", "#af7ac5", "#f39c12", "#5dade2",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", help="folder with *_districts.geojson / *_network.geojson")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--max-edges", type=int, default=120_000,
                    help="subsample road edges above this count, to keep huge "
                         "networks (e.g. the 342-TAZ / 247k-edge set) renderable")
    return p.parse_args()


def find(folder, suffix):
    hits = glob.glob(os.path.join(folder, f"*{suffix}"))
    if not hits:
        raise SystemExit(f"no *{suffix} in {folder}")
    return hits[0]


def render(folder, out, title=None, max_edges=120_000):
    districts = json.load(open(find(folder, "_districts.geojson"), encoding="utf-8"))
    network = json.load(open(find(folder, "_network.geojson"), encoding="utf-8"))

    feats = network["features"]
    if len(feats) > max_edges:
        step = len(feats) // max_edges + 1
        feats = feats[::step]
    major, minor = [], []
    for f in feats:
        pts = np.asarray(f["geometry"]["coordinates"])
        (major if f["properties"].get("speed_kmh", 0) >= 50 else minor).append(pts)

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    ax.add_collection(LineCollection(minor, colors=ROAD_MINOR, linewidths=0.25,
                                      alpha=0.6, zorder=1))
    ax.add_collection(LineCollection(major, colors=ROAD_MAJOR, linewidths=0.7,
                                      alpha=0.85, zorder=2))

    xs, ys = [], []
    for i, feat in enumerate(districts["features"]):
        geom = feat["geometry"]
        rings = geom["coordinates"] if geom["type"] == "Polygon" else [r[0] for r in geom["coordinates"]]
        col = PALETTE[i % len(PALETTE)]
        for ring in rings:
            pts = np.asarray(ring)
            xs.extend(pts[:, 0]); ys.extend(pts[:, 1])
            ax.add_patch(Polygon(pts, closed=True, facecolor=col, alpha=0.45,
                                  edgecolor="white", linewidth=0.8, zorder=3))

    lat0 = np.mean(ys)
    ax.set_aspect(1.0 / np.cos(np.radians(lat0)))
    pad = 0.04 * max(max(xs) - min(xs), max(ys) - min(ys))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    n = len(districts["features"])
    title = title or f"{os.path.basename(folder)} — {n} TAZ zones"
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left", pad=10)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n} zones, {len(feats)} edges drawn)")
    return n, len(feats)


def main():
    args = parse_args()
    render(args.folder, args.out, args.title, args.max_edges)


if __name__ == "__main__":
    main()
