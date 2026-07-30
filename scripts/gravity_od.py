#!/usr/bin/env python3
"""Build a 24h OD matrix from real OSM land use and a travel-time cost matrix.

Unlike generate_od.py, which invents a CBD and weights zones by edge count,
this derives productions from residential OSM features and attractions from
workplace/retail/education features, then runs a doubly-constrained gravity
model against real travel times:

    trips[i,j] = a_i * P_i * b_j * A_j * exp(-beta * cost[i,j])

Morning hours flow home->work, evening hours the other way.

    python scripts/cost_matrix.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --backend sumo --out output/cost_matrix.csv
    python scripts/gravity_od.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --cost output/cost_matrix.csv \
        --out-dir output/od_gravity --num 20 --daily-trips 60000
"""
import _path  # noqa: F401

from traffic_estimate.cli import (add_network, add_out_dir, add_scenarios,
                                  add_seed, add_taz, lazy_network, make_parser)
from traffic_estimate.demand import GravityGenerator, LandUseWeights, parse_hours
from traffic_estimate.services import CostMatrix
from traffic_estimate.taz import TazSet


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    add_taz(parser)
    add_out_dir(parser)
    parser.add_argument("--cost", required=True, help="CSV from cost_matrix.py")
    add_scenarios(parser, num=20, daily_trips=60000)
    parser.add_argument("--beta", type=float, default=0.12,
                        help="deterrence per minute; higher = more local trips "
                             "(0.08-0.15 is typical for urban car travel)")
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--hours", default=None,
                        help="only emit these hours, e.g. '7,8' for an AM-peak "
                             "OD. Counts still come from the same daily total "
                             "and hourly profile, so the window is a slice of a "
                             "calibrated day, not a rescaled one")
    parser.add_argument("--cache", default="output/osm_weights.npz",
                        help="land-use weight cache; keep one per map")
    parser.add_argument("--osm", default="sumo/tehran_2026_area.osm",
                        help="local OSM extract, used instead of Overpass when "
                             "present. Point this at the extract THIS net was "
                             "built from, or land use lands outside every zone")
    add_seed(parser)
    args = parser.parse_args()

    cost = CostMatrix.from_csv(args.cost)
    taz = TazSet.from_file(args.taz).select(cost.zones)
    cost = cost.reindex(taz.ids)

    # the net is only parsed when the land-use cache misses
    weights = LandUseWeights.build(taz, lazy_network(args.net), args.cache,
                                   args.osm)

    hours = parse_hours(args.hours)
    generator = GravityGenerator(taz, weights, cost.minutes,
                                 daily_trips=args.daily_trips, beta=args.beta,
                                 noise=args.noise, hours=hours, seed=args.seed)
    share = generator.shape.share_of(hours)
    print(f"{len(taz)} zones | mean trip cost "
          f"{generator.mean_trip_minutes:.1f} min | beta={args.beta} | "
          f"hours {hours[0]}-{hours[-1]} ({share * 100:.0f}% of the daily total)")
    generator.write_scenarios(args.out_dir, args.num)


if __name__ == "__main__":
    main()
