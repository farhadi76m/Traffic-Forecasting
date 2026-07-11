#!/usr/bin/env python3
"""Remap a TAZ file onto a (possibly different) SUMO network.

Keeps the zone shapes from the source TAZ file but re-assigns each zone's
edge list from the target network: every passenger-allowed edge whose center
falls inside the zone polygon. If the TAZ was built on another network with a
different coordinate offset, pass that network via --taz-net so shapes are
converted through geo-coordinates.

Example:
    python scripts/build_taz.py \
        --net sumo/tehran_2026_area.net.xml \
        --taz sumo/taz_9.xml --taz-net sumo/prune_tab.net.xml \
        --out output/taz/taz_remapped.xml
"""
import argparse
import os
import xml.etree.ElementTree as ET

import networkx as nx
import sumolib
from matplotlib.path import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net", required=True, help="target network (.net.xml)")
    p.add_argument("--taz", help="source TAZ file (.xml) to remap")
    p.add_argument("--taz-net", default=None,
                   help="network the TAZ was built on (for coordinate conversion)")
    p.add_argument("--grid", type=int, default=None, metavar="N",
                   help="ignore --taz and build an NxN grid of zones instead")
    p.add_argument("--out", required=True, help="output TAZ file")
    p.add_argument("--min-edges", type=int, default=5,
                   help="drop zones with fewer valid edges (default 5)")
    args = p.parse_args()
    if not args.grid and not args.taz:
        p.error("either --taz or --grid is required")
    return args


def zone_shapes(args, net, convert):
    """Yield (zone_id, polygon_points) either from the TAZ file or a grid."""
    if args.grid:
        (xmin, ymin), (xmax, ymax) = net.getBBoxXY()
        w, h = (xmax - xmin) / args.grid, (ymax - ymin) / args.grid
        for i in range(args.grid):
            for j in range(args.grid):
                x0, y0 = xmin + i * w, ymin + j * h
                yield f"{i}_{j}", [(x0, y0), (x0 + w, y0),
                                   (x0 + w, y0 + h), (x0, y0 + h)]
    else:
        root = ET.parse(args.taz).getroot()
        for taz in root.iter("taz"):
            shape = [tuple(map(float, pt.split(",")))
                     for pt in taz.get("shape").split()]
            yield taz.get("id"), [convert(x, y) for x, y in shape]


def main():
    args = parse_args()
    net = sumolib.net.readNet(args.net)
    src_net = sumolib.net.readNet(args.taz_net) if args.taz_net else None

    def convert(x, y):
        if src_net is None:
            return x, y
        lon, lat = src_net.convertXY2LonLat(x, y)
        return net.convertLonLat2XY(lon, lat)

    # keep only passenger edges in the largest strongly connected component,
    # so od2trips never picks unreachable origins/destinations
    g = nx.DiGraph()
    car_edges = [e for e in net.getEdges() if e.allows("passenger")]
    for e in car_edges:
        g.add_edge(e.getFromNode().getID(), e.getToNode().getID())
    scc = max(nx.strongly_connected_components(g), key=len)
    print(f"largest connected component: {len(scc)} of {g.number_of_nodes()} junctions")

    # center point of every usable edge in the target net
    edge_centers = {}
    for e in car_edges:
        if e.getFromNode().getID() not in scc or e.getToNode().getID() not in scc:
            continue
        pts = e.getShape()
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        edge_centers[e.getID()] = (cx, cy)

    out_root = ET.Element("additional")
    kept, dropped = 0, 0
    for zone_id, shape in zone_shapes(args, net, convert):
        poly = Path(shape)
        edges = [eid for eid, c in edge_centers.items() if poly.contains_point(c)]
        if len(edges) < args.min_edges:
            print(f"zone {zone_id}: only {len(edges)} edges, dropped")
            dropped += 1
            continue
        ET.SubElement(out_root, "taz", id=zone_id,
                      shape=" ".join(f"{x:.2f},{y:.2f}" for x, y in shape),
                      edges=" ".join(sorted(edges)))
        print(f"zone {zone_id}: {len(edges)} edges")
        kept += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ET.indent(tree := ET.ElementTree(out_root))
    tree.write(args.out, encoding="UTF-8", xml_declaration=True)
    print(f"wrote {args.out}: {kept} zones kept, {dropped} dropped")


if __name__ == "__main__":
    main()
