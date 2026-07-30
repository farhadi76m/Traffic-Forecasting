#!/usr/bin/env python3
"""Map the TAZ zones and the OD demand flows, with an honest provenance banner.

Left  : zones coloured by land-use balance (home-heavy <-> job-heavy). Diverging,
        because the quantity has a natural neutral midpoint (jobs == homes).
Right : morning-peak OD flows. Line width = trips, sequential blue = magnitude.

    python scripts/plot_od_map.py --taz sumo/taz_od.xml \
        --weights output/osm_weights.npz --od output/od_gravity/od_00.xml \
        --out output/figures/od_map.png
"""
import _path  # noqa: F401

from traffic_estimate.cli import add_taz, load_network, make_parser
from traffic_estimate.demand import LandUseWeights
from traffic_estimate.odmatrix import ODMatrix
from traffic_estimate.services import CostMatrix
from traffic_estimate.taz import TazSet
from traffic_estimate.viz import ODMap, use_theme


def main():
    parser = make_parser(__doc__)
    add_taz(parser)
    parser.add_argument("--weights", required=True,
                        help="osm_weights.npz from the gravity model")
    parser.add_argument("--od", required=True, help="an OD scenario XML")
    parser.add_argument("--net", default=None,
                        help="draw the road network under the zones -- the same "
                             "net the simulation runs on")
    parser.add_argument("--cost", default=None,
                        help="cost matrix CSV; zones that cannot be routed to "
                             "are flagged as islands. Without it nothing is "
                             "flagged -- do NOT hardcode zone ids")
    parser.add_argument("--hour", type=int, default=7, help="peak hour to draw")
    parser.add_argument("--label-zones", type=int, default=15,
                        help="name only the N biggest zones (0 = all)")
    parser.add_argument("--top-flows", type=int, default=150,
                        help="draw only the N heaviest zone-pair flows (0 = all)")
    parser.add_argument("--out", default="output/figures/od_map.png")
    parser.add_argument("--title",
                        default="Tehran District 5 — traffic analysis zones "
                                "and OD demand")
    parser.add_argument("--subtitle",
                        default="Zone shapes and land use are real. Trip COUNTS "
                                "are modelled, not measured — see the "
                                "reliability note.")
    parser.add_argument("--provenance", default="SYNTHETIC demand - not measured",
                        help="banner text; say plainly where the numbers came from")
    args = parser.parse_args()

    use_theme(headless=True)
    taz = TazSet.from_file(args.taz)
    weights = LandUseWeights.load(args.weights)
    islands = CostMatrix.from_csv(args.cost).islands() if args.cost else set()
    network = load_network(args.net) if args.net else None

    ODMap(taz, weights, ODMatrix.from_file(args.od), hour=args.hour,
          network=network, islands=islands, label_zones=args.label_zones,
          top_flows=args.top_flows).render(args.out, args.title, args.subtitle,
                                           args.provenance)


if __name__ == "__main__":
    main()
