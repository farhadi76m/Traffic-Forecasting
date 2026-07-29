#!/usr/bin/env python3
"""Build a TAZ file for a SUMO network: an NxN grid, or a remap of another TAZ.

Each zone claims every routable edge whose centre falls inside its polygon.
Remap mode carries a foreign TAZ's shapes across through geo-coordinates, so a
TAZ built on a different network still lands in the right place.

    python scripts/build_taz.py --net sumo/tehran_2026_area.net.xml \
        --grid 3 --out output/taz/taz_grid9.xml

    python scripts/build_taz.py --net sumo/tehran_2026_area.net.xml \
        --taz sumo/taz_9.xml --taz-net sumo/prune_tab.net.xml \
        --out output/taz/taz_remapped.xml
"""
import _path  # noqa: F401

from traffic_estimate.cli import add_network, load_network, make_parser
from traffic_estimate.taz import TazSet


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    parser.add_argument("--taz", help="source TAZ file (.xml) to remap")
    parser.add_argument("--taz-net", default=None,
                        help="network the source TAZ was built on")
    parser.add_argument("--grid", type=int, default=None, metavar="N",
                        help="ignore --taz and build an NxN grid of zones instead")
    parser.add_argument("--out", required=True, help="output TAZ file")
    parser.add_argument("--min-edges", type=int, default=5,
                        help="drop zones with fewer routable edges (default 5)")
    args = parser.parse_args()
    if not args.grid and not args.taz:
        parser.error("either --taz or --grid is required")

    network = load_network(args.net)
    print(f"largest connected component: {len(network.largest_scc)} of "
          f"{len(network.drivable)} {network.vclass} edges")

    if args.grid:
        taz = TazSet.grid(network, args.grid, args.min_edges)
    else:
        source = load_network(args.taz_net) if args.taz_net else None
        taz = TazSet.remap(network, args.taz, source, args.min_edges)

    for zone in taz:
        print(f"zone {zone.id}: {len(zone)} edges")
    taz.write(args.out)
    print(f"wrote {args.out}: {len(taz)} zones kept")


if __name__ == "__main__":
    main()
