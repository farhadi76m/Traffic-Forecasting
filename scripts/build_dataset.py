#!/usr/bin/env python3
"""Turn hourly edgeData simulation outputs into a training dataset.

Produces two files in --out-dir:
  traffic.csv.gz  one row per (scenario, edge, hour) with the number of
                  vehicles that entered the edge in that hour ("count").
                  Edges without data in an hour get count 0.
  edges.csv       static per-edge features (length, speed limit, lanes,
                  priority) for every passenger edge of the network.

Example:
    python scripts/build_dataset.py --net sumo/tehran_2026_area.net.xml \
        --sim-dir output/sim --out-dir output/dataset
"""
import argparse
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import sumolib


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net", required=True)
    p.add_argument("--sim-dir", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    net = sumolib.net.readNet(args.net)
    edges = sorted(e.getID() for e in net.getEdges() if e.allows("passenger"))
    edge_idx = {e: i for i, e in enumerate(edges)}

    static = pd.DataFrame({
        "edge": edges,
        "length": [net.getEdge(e).getLength() for e in edges],
        "speed_limit": [net.getEdge(e).getSpeed() for e in edges],
        "lanes": [net.getEdge(e).getLaneNumber() for e in edges],
        "priority": [net.getEdge(e).getPriority() for e in edges],
    })
    static.to_csv(os.path.join(args.out_dir, "edges.csv"), index=False)

    files = sorted(glob.glob(os.path.join(args.sim_dir, "*", "edgedata.xml")))
    if not files:
        raise SystemExit(f"no edgedata.xml under {args.sim_dir}")

    frames = []
    for s, path in enumerate(files):
        counts = np.zeros((len(edges), 24), dtype=np.int32)
        for _, iv in ET.iterparse(path, events=("start",)):
            if iv.tag == "interval":
                hour = int(float(iv.get("begin")) // 3600)
            elif iv.tag == "edge":
                i = edge_idx.get(iv.get("id"))
                if i is not None and hour < 24:
                    counts[i, hour] = int(float(iv.get("entered") or 0))
        df = pd.DataFrame({
            "scenario": s,
            "edge": np.repeat(edges, 24),
            "hour": np.tile(np.arange(24), len(edges)),
            "count": counts.reshape(-1),
        })
        frames.append(df)
        print(f"scenario {s} ({os.path.dirname(path)}): "
              f"{counts.sum()} vehicle-entries, {(counts.sum(axis=1) > 0).sum()} active edges")

    data = pd.concat(frames, ignore_index=True)
    out = os.path.join(args.out_dir, "traffic.csv.gz")
    data.to_csv(out, index=False)
    print(f"wrote {out}: {len(data)} rows, {len(files)} scenarios, {len(edges)} edges")


if __name__ == "__main__":
    main()
