#!/usr/bin/env python3
"""Run 24-hour SUMO mesoscopic simulations for every OD scenario.

Per scenario: od2trips (OD -> trips via the TAZ) -> duarouter (routes) -> sumo
--mesosim, writing one hourly edgeData interval per edge. Trips and routes are
deleted afterwards -- they are gigabytes per congested scenario and nothing
downstream reads them.

    python scripts/run_sim.py --net sumo/tehran_2026_area.net.xml \
        --taz output/taz/taz_grid9.xml --od-dir output/od --out-dir output/sim
"""
import glob
import os
import sys

import _path  # noqa: F401

from traffic_estimate.cli import (add_network, add_out_dir, add_seed, add_taz,
                                  make_parser)
from traffic_estimate.simulation import DAY_SECONDS, ScenarioBatch, SumoRunner


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    add_taz(parser)
    parser.add_argument("--od-dir", required=True,
                        help="directory of tazRelation OD files")
    add_out_dir(parser)
    parser.add_argument("--end", type=int, default=DAY_SECONDS,
                        help="simulation end time in seconds")
    parser.add_argument("--period", type=int, default=3600,
                        help="edgeData aggregation period in seconds")
    add_seed(parser, default=7)
    parser.add_argument("--no-meso", action="store_true",
                        help="use the microscopic simulation instead")
    parser.add_argument("--jobs", type=int, default=1,
                        help="scenarios to simulate in parallel "
                             "(SUMO itself is single-threaded)")
    parser.add_argument("--threads", type=int, default=1,
                        help="duarouter routing threads per scenario")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip scenarios whose edgedata already exists")
    parser.add_argument("--keep-routes", action="store_true",
                        help="keep the trips/routes XML for debugging")
    args = parser.parse_args()

    od_files = sorted(glob.glob(os.path.join(args.od_dir, "od_*.xml")))
    if not od_files:
        sys.exit(f"no od_*.xml files in {args.od_dir}")

    runner = SumoRunner(net=args.net, taz=args.taz, meso=not args.no_meso,
                        period=args.period, seed=args.seed, threads=args.threads,
                        keep_intermediates=args.keep_routes,
                        extra_sumo_args=("--duration-log.statistics",))
    batch = ScenarioBatch(runner, od_files, args.out_dir, jobs=args.jobs,
                          skip_done=args.skip_done)
    if batch.run():
        sys.exit(1)


if __name__ == "__main__":
    main()
