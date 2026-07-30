#!/usr/bin/env python3
"""Build a PDF report of the Tehran TAZ partitions in <root>/b_tehran_*.

For each folder built by build_taz_districts.py this computes zone statistics
from the TAZ file, renders a snapshot of the zones over the road network, and
lays out a summary table plus one page per level.

    python scripts/make_taz_report.py --root neshan --out docs/taz_report.pdf
"""
import glob
import os
import re
import sys

import _path  # noqa: F401

from traffic_estimate.cli import make_parser
from traffic_estimate.geo import polygon_area
from traffic_estimate.taz import TazSet
from traffic_estimate.viz import PdfReport, taz_snapshot, use_theme

HEAD = "Traffic-Estimate - Tehran TAZ partitions"

# folders to skip, with the reason shown in the report
SKIP = {
    "b_tehran_l8": "admin_level 8 does not subdivide this network - collapses to "
                   "a single zone covering everywhere, so it carries no zoning "
                   "information (see build_taz_districts.py).",
    "b_tehran_l11.taz.xml": "duplicate re-run of b_tehran_l11 (same network, same "
                            "admin_level, near-identical output) - kept on disk "
                            "but not analysed separately.",
}

LEVEL_NOTE = {
    "b_tehran_l7": "admin_level 7 (dehestan / rural sub-district) - coarsest "
                   "usable split: 10 huge zones, good for a city-wide overview, "
                   "too coarse for OD calibration.",
    "b_tehran_l9": "admin_level 9 (mantaqe / municipal district) - Tehran's 22 "
                   "official districts, a natural, recognisable zoning.",
    "b_tehran_l10": "admin_level 10 - an intermediate level between district and "
                    "neighbourhood; 16 zones here after small ones were merged "
                    "into neighbours (--min-edges).",
    "b_tehran_l11": "admin_level 11 (mahalleh / neighbourhood) on the skeleton "
                    "network - the finest usable split, 251 zones.",
    "b_tehran_l11_new": "same admin_level 11 / same network as b_tehran_l11, "
                        "rebuilt later; fewer zones (223) because more small "
                        "neighbourhoods fell under --min-edges and got merged.",
    "b_tehran_l11_org": "admin_level 11 on tehran2.net.xml, the larger, denser "
                        "network (247k edges vs. 30k) - 342 zones, an order of "
                        "magnitude more edges per zone than the skeleton runs.",
}


def net_source(folder: str) -> str:
    configs = glob.glob(os.path.join(folder, "*.sumocfg"))
    if not configs:
        return "?"
    with open(configs[0], encoding="utf-8") as handle:
        match = re.search(r'net-file value="([^"]+)"', handle.read())
    return os.path.basename(match.group(1)) if match else "?"


def zone_stats(folder: str) -> dict:
    taz = TazSet.from_file(glob.glob(os.path.join(folder,
                                                  "*_districts.taz.xml"))[0])
    areas = [polygon_area(zone.shape) / 1e6 for zone in taz]
    edges = [len(zone) for zone in taz]
    return {"n": len(taz), "net": net_source(folder),
            "area_total": sum(areas), "area_avg": sum(areas) / len(areas),
            "area_min": min(areas), "area_max": max(areas),
            "edges_total": sum(edges), "edges_avg": sum(edges) / len(edges),
            "edges_min": min(edges), "edges_max": max(edges)}


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--root", default="neshan",
                        help="folder holding the b_tehran_* directories")
    parser.add_argument("--out", default="docs/taz_report.pdf")
    parser.add_argument("--figdir", default="docs/figures/taz",
                        help="where per-level snapshot PNGs are written")
    parser.add_argument("--refresh-figs", action="store_true",
                        help="re-render snapshots even if the PNG exists")
    args = parser.parse_args()

    use_theme(headless=True)
    folders = sorted(
        d for d in glob.glob(os.path.join(args.root, "b_tehran_*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*_districts.taz.xml")))
    if not folders:
        sys.exit(f"no b_tehran_* TAZ output found under {args.root}")

    os.makedirs(args.figdir, exist_ok=True)
    entries = []
    for folder in folders:
        name = os.path.basename(folder)
        reason = SKIP.get(name)
        stats = zone_stats(folder)
        png = None
        if not reason:
            png = os.path.join(args.figdir, f"{name}.png")
            if args.refresh_figs or not os.path.exists(png):
                taz_snapshot(folder, png, f"{name} - {stats['n']} TAZ zones")
        entries.append((name, stats, reason, png))

    pdf = PdfReport(HEAD)
    pdf.add_page()
    pdf.h1("Tehran TAZ Partitions")
    pdf.body(
        "Traffic analysis zones (TAZ) for the Tehran SUMO network, built at "
        "several OpenStreetMap admin_levels by scripts/build_taz_districts.py "
        "(neshan/b_tehran_*). Each level partitions the network's routable car "
        "edges into administrative zones, merging any zone with fewer than "
        "--min-edges edges into its nearest neighbour so every zone is large "
        "enough to route traffic to and from.", size=10.5)

    pdf.h2("Summary - zones, surface, edges")
    pdf.table(["folder", "TAZ", "avg km²/taz", "total km²", "avg edges/taz",
               "total edges"],
              [(name, stats["n"], f"{stats['area_avg']:.2f}",
                f"{stats['area_total']:.0f}", f"{stats['edges_avg']:.0f}",
                stats["edges_total"])
               for name, stats, reason, _ in entries if not reason],
              widths=[40, 14, 22, 24, 22, 22], size=8.6)
    pdf.body(
        "\"avg km²/taz\" and \"avg edges/taz\" are the mean zone surface area "
        "and routable-edge count across all zones in that partition; both shrink "
        "as admin_level goes finer (7 -> 11) or as the underlying network gets "
        "denser (l11 -> l11_org).")

    pdf.h2("Excluded from analysis")
    for name, _, reason, _ in entries:
        if reason:
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.cell(0, 5.5, name, new_x="LMARGIN", new_y="NEXT")
            pdf.body(reason)

    for name, stats, reason, png in entries:
        if reason:
            continue
        pdf.add_page()
        pdf.h2(name)
        pdf.figure(png, LEVEL_NOTE.get(name, ""), width=155)
        pdf.table(["metric", "mean", "range (min-max)"],
                  [("zone surface", f"{stats['area_avg']:.2f} km²",
                    f"{stats['area_min']:.2f} km² - {stats['area_max']:.2f} km²"),
                   ("edges per zone", f"{stats['edges_avg']:.0f}",
                    f"{stats['edges_min']:.0f} - {stats['edges_max']:.0f}")],
                  widths=[55, 55, 60])
        pdf.body(f"{stats['n']} zones, {stats['edges_total']} routable edges "
                 f"total, built from {stats['net']}.")

    pdf.save(args.out)


if __name__ == "__main__":
    main()
