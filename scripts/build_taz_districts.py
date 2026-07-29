#!/usr/bin/env python3
"""Build district GeoJSON + a TAZ file for any SUMO network, from OSM boundaries.

Administrative boundaries covering the network bbox are fetched from Overpass
(the local .osm extract has its boundary relations clipped and cannot be used),
assembled into rings, and used to partition the network. Only edges in the
largest strongly-connected component are assigned, so routing cannot emit trips
into unreachable areas. Zones under --min-edges are merged into the nearest one.

For Tehran: admin_level 9 = منطقه (district), 11 = محله (neighbourhood). Pick the
level that actually subdivides your network -- a net inside a single district
yields one useless TAZ at level 9.

Outputs, named after the network:
  <prefix>_districts.geojson   district polygons (WGS84)
  <prefix>_network.geojson     network edges (WGS84), tagged with their TAZ
  <prefix>_districts.taz.xml   TAZ for od2trips (net coordinates)
  <prefix>_districts.poly.xml  filled polygons for sumo-gui
  <prefix>_districts.sumocfg   ready-to-open view (+ .view.xml)

    build_taz_districts.py sumo/tehran_2026_area.net.xml --gui
    build_taz_districts.py sumo/prune_tab.net.xml --level 9 --min-edges 100
"""
import argparse
import os
import subprocess
import sys

import _path  # noqa: F401

from traffic_estimate.cli import load_network, require_files
from traffic_estimate.districts import DistrictTazBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build district GeoJSON + TAZ for a SUMO network from OSM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("net", help="SUMO .net.xml to build the TAZ for")
    parser.add_argument("-o", "--outdir",
                        help="output directory [default: the network's directory]")
    parser.add_argument("-p", "--prefix",
                        help="basename for the outputs [default: from the net]")
    parser.add_argument("-l", "--level", default="11",
                        help="OSM admin_level to use as zones "
                             "(Tehran: 9=منطقه, 11=محله)")
    parser.add_argument("-m", "--min-edges", type=int, default=40,
                        help="zones with fewer routable edges are merged into "
                             "the nearest zone")
    parser.add_argument("--pad", type=float, default=0.006,
                        help="padding added to the network bbox, in degrees")
    parser.add_argument("--vclass", default="passenger",
                        help="vehicle class the TAZ edges must allow")
    parser.add_argument("--no-scc", action="store_true",
                        help="keep every edge instead of only the largest "
                             "strongly-connected component")
    parser.add_argument("--refresh", action="store_true",
                        help="re-query Overpass even if a cache exists")
    parser.add_argument("--gui", action="store_true",
                        help="open the result in sumo-gui when done")
    parser.add_argument("--list-levels", action="store_true",
                        help="list the admin_levels OSM has over this network "
                             "and exit (use this first on a new map)")
    args = parser.parse_args()

    require_files(args.net)
    args.net = os.path.abspath(args.net)
    outdir = os.path.abspath(args.outdir or os.path.dirname(args.net))
    os.makedirs(outdir, exist_ok=True)
    network = load_network(args.net, vclass=args.vclass)
    prefix = args.prefix or network.name

    builder = DistrictTazBuilder(network, level=args.level,
                                 min_edges=args.min_edges, pad=args.pad,
                                 connected_only=not args.no_scc)
    south, west, north, east = builder.bbox
    print(f"  net bbox lat {south:.4f}..{north:.4f} lon {west:.4f}..{east:.4f}")

    if args.list_levels:
        probe = os.path.join(outdir, f"{prefix}_osm_levels.json")
        by_level = builder.available_levels(probe)
        print(f"\nadmin_levels covering {os.path.basename(args.net)}:\n")
        print(f"  {'level':<7}{'zones':<7}examples")
        for level in sorted(by_level, key=int):
            names = sorted(by_level[level])
            print(f"  {level:<7}{len(names):<7}{', '.join(names[:4])}"
                  f"{' ...' if len(names) > 4 else ''}")
        print("\npick the level whose zone count actually subdivides your net, then:")
        print(f"  {os.path.basename(sys.argv[0])} {args.net} --level <L>")
        os.remove(probe)
        return

    # the cache is per net + level, so a different network never reuses polygons
    cache = os.path.join(outdir, f"{prefix}_osm_boundaries_l{args.level}.json")
    builder.build(cache, refresh=args.refresh)
    print(builder.summary())

    outputs = builder.write(outdir, prefix)
    for path in vars(outputs).values():
        print(f"wrote {path}")

    if args.gui:
        print("\nopening sumo-gui ...")
        subprocess.Popen(["sumo-gui", "-c", outputs.sumocfg],
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        print(f"  -> sumo-gui -c {outputs.sumocfg}")


if __name__ == "__main__":
    main()
