#!/usr/bin/env python3
"""Map the TAZ zones and the OD demand flows, with an honest provenance banner.

Left  : zones, colored by land-use balance (home-heavy <-> job-heavy). Diverging,
        because the quantity has a natural neutral midpoint (jobs == homes).
Right : morning-peak OD flows. Line width = trips, sequential blue = magnitude.

    python scripts/plot_od_map.py --taz sumo/taz_od.xml \
        --weights output/osm_weights.npz --od output/od_gravity/od_00.xml \
        --out output/figures/od_map.png
"""
import argparse
import os
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list(  # blue <-> gray <-> red
    "homejob", ["#184f95", "#86b6ef", "#f0efec", "#e88b8a", "#c22b2a"])
SEQ = LinearSegmentedColormap.from_list(
    "demand", ["#cde2fb", "#5598e7", "#256abf", "#0d366b"])
ISLAND = "#eb6834"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taz", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--od", required=True)
    p.add_argument("--net", default=None,
                   help="draw the road network under the zones -- the same net "
                        "the simulation runs on, so zones sit on a real map")
    p.add_argument("--cost", default=None,
                   help="cost matrix CSV; zones that cannot be routed to are "
                        "flagged as islands. Without it nothing is flagged -- do "
                        "NOT hardcode zone ids, they differ per TAZ file")
    p.add_argument("--hour", type=int, default=7, help="peak hour to draw")
    p.add_argument("--out", default="output/figures/od_map.png")
    p.add_argument("--title",
                   default="Tehran District 5 — traffic analysis zones and OD demand")
    p.add_argument("--subtitle",
                   default="Zone shapes and land use are real. Trip COUNTS are "
                           "modelled, not measured — see the reliability note.")
    p.add_argument("--provenance", default="SYNTHETIC demand - not measured",
                   help="banner text; say plainly where the numbers came from")
    return p.parse_args()


def find_islands(cost_csv):
    """Zones whose costs are the sentinel fill -- unreachable from the rest.

    cost_matrix.py fills unroutable pairs with 3x the max real cost, so an island
    shows up as a row of identical maximal values.
    """
    if not cost_csv or not os.path.exists(cost_csv):
        return set()
    rows = [l.strip().split(",") for l in open(cost_csv) if l.strip()]
    zones = rows[0][1:]
    m = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    off = m[~np.eye(len(zones), dtype=bool)]
    # The sentinel is exactly 3x the largest REAL cost, so it sits a factor of
    # three above the next distinct value. Comparing against the median instead
    # flags the slowest genuine pair on any net with a wide cost spread.
    distinct = np.unique(off)
    if len(distinct) < 2 or distinct[-1] < 2.5 * distinct[-2]:
        return set()                      # no sentinel -> fully connected net
    bad = (m >= distinct[-1] - 1e-6).sum(axis=1)
    return {z for z, n in zip(zones, bad) if n > 0}


def short(zid, n=14):
    """District names are long; keep labels legible."""
    return zid if len(zid) <= n else zid[:n - 1] + "…"


def road_layers(net_file):
    """Edge polylines in km, split into arterial (fast) and local roads."""
    import sumolib
    net = sumolib.net.readNet(net_file, withInternal=False)
    major, minor = [], []
    for e in net.getEdges():
        pts = np.asarray(e.getShape()) / 1000.0
        (major if e.getSpeed() >= 15 else minor).append(pts)
    return major, minor


def main():
    args = parse_args()
    islands = find_islands(args.cost)
    polys, cents = {}, {}
    for taz in ET.parse(args.taz).getroot().iter("taz"):
        pts = np.array([tuple(map(float, p.split(",")))
                        for p in taz.get("shape").split()])
        polys[taz.get("id")] = pts / 1000.0          # metres -> km
        cents[taz.get("id")] = pts.mean(axis=0) / 1000.0

    z = np.load(args.weights, allow_pickle=True)
    zones = list(z["zones"])
    prod = dict(zip(zones, z["prod"]))
    attr = dict(zip(zones, z["attr"]))

    flows = {}
    for iv in ET.parse(args.od).getroot().iter("interval"):
        if int(float(iv.get("begin"))) // 3600 != args.hour:
            continue
        for rel in iv.iter("tazRelation"):
            a, b = rel.get("from"), rel.get("to")
            if a != b:
                flows[(a, b)] = flows.get((a, b), 0) + int(rel.get("count"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.2), facecolor=SURFACE)
    fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.12)

    if args.net:
        major, minor = road_layers(args.net)
        # left: white roads etched into the choropleth (between fills and labels)
        ax1.add_collection(LineCollection(minor, colors=SURFACE, linewidths=0.3,
                                          alpha=0.4, zorder=2))
        ax1.add_collection(LineCollection(major, colors=SURFACE, linewidths=1.0,
                                          alpha=0.7, zorder=2))
        # right: gray basemap under the flow arrows
        ax2.add_collection(LineCollection(minor, colors=GRID, linewidths=0.3,
                                          zorder=0.4))
        ax2.add_collection(LineCollection(major, colors=BASELINE, linewidths=0.8,
                                          alpha=0.8, zorder=0.5))

    # ---- left: land-use balance (measured from OSM) ----
    ratios = {k: np.log2(attr[k] / prod[k]) for k in zones}
    lim = max(abs(v) for v in ratios.values())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    for zid, pts in polys.items():
        if zid not in ratios:
            continue
        ax1.add_patch(Polygon(pts, closed=True, facecolor=DIVERGING(norm(ratios[zid])),
                              edgecolor=SURFACE, linewidth=2))
        cx, cy = cents[zid]
        ax1.text(cx, cy + 0.18, short(zid), ha="center", va="center", fontsize=9,
                 color=INK, fontweight="bold")
        ax1.text(cx, cy - 0.24, f"{int(prod[zid])}h / {int(attr[zid])}j",
                 ha="center", va="center", fontsize=8, color=INK2)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=norm)
    cb = fig.colorbar(sm, ax=ax1, orientation="horizontal", pad=0.07, shrink=0.75)
    cb.set_label("← more homes      jobs / homes (log₂)      more jobs →",
                 fontsize=9, color=INK2)
    cb.outline.set_edgecolor(BASELINE)
    ax1.set_title("Land use — MEASURED (OpenStreetMap)", fontsize=12,
                  color=INK, pad=10, loc="left", fontweight="bold")

    # ---- right: OD flows ----
    mx = max(flows.values()) if flows else 1
    for (a, b), n in sorted(flows.items(), key=lambda kv: kv[1]):
        if a not in cents or b not in cents:
            continue
        (x1, y1), (x2, y2) = cents[a], cents[b]
        ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=SEQ(n / mx),
                                     linewidth=0.4 + 4.6 * (n / mx),
                                     alpha=0.85, shrinkA=9, shrinkB=11,
                                     connectionstyle="arc3,rad=0.12"))
    for zid, pts in polys.items():
        island = zid in islands
        ax2.add_patch(Polygon(pts, closed=True, facecolor="none",
                              edgecolor=ISLAND if island else GRID,
                              linewidth=1.6 if island else 1.0,
                              linestyle=(0, (3, 2)) if island else "-", zorder=0))
        cx, cy = cents[zid]
        ax2.scatter([cx], [cy], s=90, facecolor=SURFACE,
                    edgecolor=ISLAND if island else BASELINE, linewidth=1.4, zorder=3)
        ax2.text(cx, cy + 0.3, short(zid), ha="center", va="bottom", fontsize=8,
                 color=ISLAND if island else INK, zorder=4,
                 fontweight="bold" if island else "normal")
    sm2 = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, mx))
    cb2 = fig.colorbar(sm2, ax=ax2, orientation="horizontal", pad=0.07, shrink=0.75)
    cb2.set_label(f"trips in hour {args.hour:02d}:00–{args.hour + 1:02d}:00",
                  fontsize=9, color=INK2)
    cb2.outline.set_edgecolor(BASELINE)
    ax2.set_title(f"Demand flows, {args.hour:02d}:00 peak — {args.provenance}",
                  fontsize=12, color=INK, pad=10, loc="left", fontweight="bold")
    if islands:
        ax2.plot([], [], color=ISLAND, linestyle=(0, (3, 2)), linewidth=1.6,
                 label=f"unroutable island ({len(islands)})")
        ax2.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)

    # add_patch does not grow the data limits, so autoscale would silently crop
    # zones out of the frame. Set the extent from the polygons themselves.
    allpts = np.vstack(list(polys.values()))
    (x0, y0), (x1, y1) = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.06 * max(x1 - x0, y1 - y0)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + 2 * pad)   # headroom for the legend
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("km east of net origin", fontsize=9, color=MUTED)
    ax1.set_ylabel("km north", fontsize=9, color=MUTED)

    fig.suptitle(args.title, x=0.045, y=0.955, ha="left", fontsize=16,
                 color=INK, fontweight="bold")
    fig.text(0.045, 0.885, args.subtitle, ha="left", fontsize=10.5, color=INK2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"{args.out}  ({len(polys)} zones, {len(flows)} flows, hour {args.hour})")


if __name__ == "__main__":
    main()
