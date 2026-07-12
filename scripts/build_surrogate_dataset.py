#!/usr/bin/env python3
"""Pair each sampled OD matrix with what SUMO did to it -> surrogate training set.

`build_dataset.py` keeps only vehicle counts and throws the OD away, because the
forecasting model never sees OD. The surrogate needs both sides of the map:

    OD (24, n_zones^2)  ->  SUMO  ->  counts + travel times (n_edges, 24)

Travel time is already in edgedata.xml (the `traveltime` attribute); the old
builder simply never read it.

Edges absent from an hour's interval had no vehicle on them (excludeEmpty), so
their count is 0 and their travel time is the free-flow time length/speed --
which is the physically right fill, not a missing value.

Writes <out-dir>/data.npz with od, counts, traveltime, free-flow time and the
static edge features.

Example:
    python scripts/build_surrogate_dataset.py --net sumo/tehran_2026_area.net.xml \
        --od-dir output_sur/cong/od --sim-dir output_sur/cong/sim \
        --out-dir output_sur/cong/dataset
"""
import argparse
import gzip
import os
import xml.etree.ElementTree as ET

import numpy as np
import sumolib
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net", required=True)
    p.add_argument("--od-dir", required=True, help="dir with od.npy + od_*.xml")
    p.add_argument("--sim-dir", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def read_edgedata(path, edge_idx, n_edges):
    """-> counts (n_edges, 24), traveltime (n_edges, 24); NaN where unobserved."""
    counts = np.zeros((n_edges, 24), np.float32)
    tt = np.full((n_edges, 24), np.nan, np.float32)
    opener = gzip.open if path.endswith(".gz") else open
    hour = None
    with opener(path, "rb") as f:
        for _, el in ET.iterparse(f, events=("start",)):
            if el.tag == "interval":
                hour = int(float(el.get("begin")) // 3600)
            elif el.tag == "edge" and hour is not None and hour < 24:
                i = edge_idx.get(el.get("id"))
                if i is None:
                    continue
                counts[i, hour] = float(el.get("entered") or 0)
                if el.get("traveltime") is not None:
                    tt[i, hour] = float(el.get("traveltime"))
            el.clear()
    return counts, tt


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    net = sumolib.net.readNet(args.net)
    edges = sorted(e.getID() for e in net.getEdges() if e.allows("passenger"))
    edge_idx = {e: i for i, e in enumerate(edges)}
    n_edges = len(edges)

    static = np.array([[net.getEdge(e).getLength(), net.getEdge(e).getSpeed(),
                        net.getEdge(e).getLaneNumber(), net.getEdge(e).getPriority()]
                       for e in edges], np.float32)
    free_flow = static[:, 0] / np.maximum(static[:, 1], 0.1)  # length / speed limit

    od_all = np.load(os.path.join(args.od_dir, "od.npy"))  # (n_scen, 24, n_od)

    keep, counts, tts = [], [], []
    for s in tqdm(range(len(od_all)), desc="scenarios"):
        name = f"od_{s:03d}"
        path = os.path.join(args.sim_dir, name, "edgedata.xml.gz")
        if not os.path.exists(path):
            path = os.path.join(args.sim_dir, name, "edgedata.xml")
        if not os.path.exists(path):
            continue  # scenario failed in SUMO; drop it from both sides
        c, t = read_edgedata(path, edge_idx, n_edges)
        # unobserved edge-hour = empty road = free flow
        t = np.where(np.isnan(t), free_flow[:, None], t)
        keep.append(s)
        counts.append(c)
        tts.append(t)

    od = od_all[keep]
    counts = np.stack(counts)
    tts = np.stack(tts)

    out = os.path.join(args.out_dir, "data.npz")
    np.savez_compressed(out, od=od, counts=counts, traveltime=tts,
                        free_flow=free_flow, static=static,
                        edges=np.array(edges), scenarios=np.array(keep))

    ratio = tts / free_flow[None, :, None]
    peak = ratio[:, :, 7:10]
    print(f"\nwrote {out}")
    print(f"  {len(keep)}/{len(od_all)} scenarios, {n_edges} edges, OD dim {od.shape[-1]} x 24")
    print(f"  counts     : mean {counts.mean():.1f}  max {counts.max():.0f}")
    print(f"  tt/freeflow: mean {ratio.mean():.2f}  p90 {np.percentile(ratio, 90):.2f}  "
          f"max {ratio.max():.1f}")
    print(f"  at peak 07-09: mean {peak.mean():.2f}  "
          f"share of edge-hours >1.2x free-flow {100*(peak > 1.2).mean():.1f}%")


if __name__ == "__main__":
    main()
