#!/usr/bin/env python3
"""Build a zone-to-zone travel-time (impedance) matrix for a TAZ file.

This is the *cost* matrix, not demand: cell (i, j) is how many seconds it takes
to drive from zone i to zone j. It feeds the deterrence term of the gravity
model -- no routing API returns trip counts.

Backends:
  sumo    shortest path on our own net (free, offline, respects the pruned net)
  osrm    public OSRM demo server (free, rate limited, real OSM road speeds)
  arcgis  ArcGIS travelCostMatrix (needs ARCGIS_API_KEY, has historical traffic)

    python scripts/cost_matrix.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --backend sumo --out output/cost_matrix.csv
"""
import os

import _path  # noqa: F401
import numpy as np

from traffic_estimate.cli import add_network, add_taz, load_network, make_parser
from traffic_estimate.services import BACKENDS, ArcGisCost, CostMatrix
from traffic_estimate.taz import TazSet


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    add_taz(parser)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=list(BACKENDS), default="sumo")
    parser.add_argument("--api-key", default=None,
                        help="ArcGIS key; defaults to $ARCGIS_API_KEY")
    parser.add_argument("--depart", default=None,
                        help="arcgis only: departure time as epoch ms, for "
                             "traffic-aware costs")
    parser.add_argument("--allow-partial", action="store_true",
                        help="proceed even if the TAZ does not fit this net")
    args = parser.parse_args()

    network = load_network(args.net)
    taz = TazSet.from_file(args.taz)
    backend = (ArcGisCost(args.api_key, args.depart) if args.backend == "arcgis"
               else BACKENDS[args.backend]())

    routable = len(taz.representative_edges(network))
    print(f"{routable}/{len(taz)} routable zones, backend={args.backend}")
    # A TAZ built for a different net silently loses zones here. Losing half the
    # study area must not be a warning you can scroll past.
    if routable < 0.8 * len(taz) and not args.allow_partial:
        raise SystemExit(
            f"\n{len(taz) - routable} of {len(taz)} zones have no routable edge "
            f"on {os.path.basename(args.net)}.\nThat TAZ file was almost "
            f"certainly built for a different net -- its edge ids do not match "
            f"this one.\nRebuild the TAZ against this net, or pass the net it "
            f"was built for.\nPass --allow-partial to proceed anyway (you will "
            f"model only {routable} zones).")

    cost = CostMatrix.build(network, taz, backend)
    unreachable = cost.fill_unreachable()
    if unreachable:
        # island zones (e.g. Shahran) never route; make them far away rather
        # than dropping them from the model
        print(f"warning: {unreachable} unreachable pairs, filling with 3x max cost")

    cost.to_csv(args.out)
    print(f"{args.out}: {len(cost.zones)}x{len(cost.zones)}, "
          f"mean {np.mean(cost.minutes):.1f} min, max {np.max(cost.minutes):.1f} min")


if __name__ == "__main__":
    main()
