#!/usr/bin/env python3
"""Calibrate OD demand magnitude against REAL measured travel times.

gravity_od.py gives the *shape* of demand; its magnitude (--daily-trips) is a
guess. Congestion is monotone in demand, so this bisects on total daily trips
until the simulated congestion index matches the measured one on the probe
routes. Only the peak window is simulated each iteration.

    python scripts/calibrate_od.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --cost output/cost_matrix.csv \
        --observed output/observed_neshan.csv --out-dir output/od_calibrated

Honest limits: travel time pins volume only loosely -- a road at half free-flow
can be moderately or heavily loaded, the two branches of the fundamental
diagram. This gets the order of magnitude right. It is not counting cars.
"""
import os

import _path  # noqa: F401

from traffic_estimate.cli import (add_network, add_taz, hour_set, load_network,
                                  make_parser)
from traffic_estimate.demand import GravityGenerator, LandUseWeights, parse_hours
from traffic_estimate.services import CostMatrix
from traffic_estimate.simulation import (DemandCalibrator, ObservedCongestion,
                                         SumoRunner, build_probe)
from traffic_estimate.taz import TazSet


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    add_taz(parser)
    parser.add_argument("--cost", required=True)
    parser.add_argument("--observed", required=True,
                        help="CSV from fetch_neshan.py or fetch_speeds.py")
    parser.add_argument("--out-dir", default="output/od_calibrated")
    parser.add_argument("--work", default="output/calib_work")
    parser.add_argument("--peak-hours", default="7,8",
                        help="hours to match on; congestion only informs when "
                             "the network is loaded. Keep this contiguous -- the "
                             "whole span is simulated each iteration")
    parser.add_argument("--lo", type=int, default=10_000,
                        help="lower bracket, trips/day")
    parser.add_argument("--hi", type=int, default=1_000_000, help="upper bracket")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--tol", type=float, default=0.02,
                        help="stop when the index is within this of observed")
    parser.add_argument("--beta", type=float, default=0.12)
    parser.add_argument("--num", type=int, default=20,
                        help="scenario days to emit once calibrated")
    parser.add_argument("--hours", default=None,
                        help="restrict the emitted OD to these hours, e.g. '7,8'")
    parser.add_argument("--osm", default=None,
                        help="local .osm extract for the land-use weights; must "
                             "be the extract this --net was built from")
    parser.add_argument("--cache", default=None,
                        help="land-use weight cache (keep one per map)")
    parser.add_argument("--threads", type=int,
                        default=max(1, (os.cpu_count() or 2) - 2),
                        help="duarouter routing threads (default: cores - 2)")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep each iteration's trips/routes for debugging")
    args = parser.parse_args()

    peak = hour_set(args.peak_hours)
    observed = ObservedCongestion.from_csv(args.observed, peak)

    network = load_network(args.net)
    cost = CostMatrix.from_csv(args.cost)
    taz = TazSet.from_file(args.taz).select(cost.zones)
    cost = cost.reindex(taz.ids)
    weights = LandUseWeights.build(taz, network, args.cache, args.osm)

    probe = build_probe(network, taz, observed)
    print(f"observed congestion index: {observed.target:.3f} -- "
          f"{probe.describe()}, hours {sorted(peak)}")
    print(f"  -> real traffic moves at {observed.target * 100:.0f}% of free "
          "flow in peak\n")

    # the gravity shape does not depend on magnitude, so it is solved once and
    # only --daily-trips moves between iterations
    generator = GravityGenerator(taz, weights, cost.minutes, beta=args.beta,
                                 daily_trips=args.lo)
    calibrator = DemandCalibrator(generator, SumoRunner(args.net, args.taz,
                                                        threads=args.threads),
                                  probe, observed, peak, args.work,
                                  keep_work=args.keep_work)
    trips, index = calibrator.calibrate(args.lo, args.hi, args.iters, args.tol)
    print(f"\ncalibrated: {trips:,} trips/day (sim index {index:.3f} vs "
          f"observed {observed.target:.3f})")

    generator.daily_trips = trips
    generator.hours = parse_hours(args.hours)
    generator.noise = 0.08
    generator.write_scenarios(args.out_dir, args.num)
    print(f"{args.num} calibrated scenarios -> {args.out_dir}")


if __name__ == "__main__":
    main()
