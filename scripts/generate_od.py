#!/usr/bin/env python3
"""Generate several 24-hour OD matrices in SUMO tazRelation XML format.

Each scenario is one simulated "day": a gravity matrix between the TAZ zones,
shaped over 24 hours with a morning commute peak toward the largest zone, a
midday bump and an evening peak back out. Day-to-day variation comes from a
per-scenario demand factor plus Poisson sampling of every OD cell.

    python scripts/generate_od.py --taz output/taz/taz_grid9.xml \
        --out-dir output/od --num 60 --daily-trips 6000
"""
import _path  # noqa: F401

from traffic_estimate.cli import (add_out_dir, add_scenarios, add_seed, add_taz,
                                  make_parser)
from traffic_estimate.demand import TypicalDayGenerator
from traffic_estimate.taz import TazSet


def main():
    parser = make_parser(__doc__)
    add_taz(parser)
    add_out_dir(parser)
    add_scenarios(parser, num=20, daily_trips=6000)
    parser.add_argument("--noise", type=float, default=0.08,
                        help="std of the per-scenario demand factor")
    add_seed(parser)
    args = parser.parse_args()

    taz = TazSet.from_file(args.taz)
    generator = TypicalDayGenerator(taz, daily_trips=args.daily_trips,
                                    noise=args.noise, seed=args.seed)
    print(f"{len(taz)} zones, center zone: {generator.center_zone}")
    generator.write_scenarios(args.out_dir, args.num)


if __name__ == "__main__":
    main()
