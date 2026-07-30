#!/usr/bin/env python3
"""Sample OD matrices from a wide prior, for training a SUMO surrogate.

generate_od.py makes many days of the *same* city, so a surrogate trained on it
learns the mean and nothing about how traffic responds to demand. Here every
scenario is a *different* city: total demand, per-zone production and
attraction, cell-level flows, the 24h shape and the commute strength are all
drawn from a log-normal prior. That prior is also the prior of the inverse
problem.

Writes od_XXX.xml for od2trips plus od.npy, the same numbers as a tensor.

    python scripts/sample_od.py --taz output/taz/taz_grid9.xml \
        --out-dir output_sur/light/od --num 300 --daily-trips 12000
"""
import os

import _path  # noqa: F401
import numpy as np

from traffic_estimate.cli import (add_out_dir, add_scenarios, add_seed, add_taz,
                                  make_parser)
from traffic_estimate.demand import PriorSampler
from traffic_estimate.taz import TazSet


def main():
    parser = make_parser(__doc__)
    add_taz(parser)
    add_out_dir(parser)
    add_scenarios(parser, num=300, daily_trips=12000)
    parser.add_argument("--sigma-total", type=float, default=0.35,
                        help="log-normal std of the total demand multiplier")
    parser.add_argument("--max-trips", type=int, default=0,
                        help="clip total daily trips (0 = no clip). Past ~130k "
                             "this net gridlocks: travel times hit 500x free "
                             "flow and SUMO takes minutes per run")
    parser.add_argument("--sigma-zone", type=float, default=0.45,
                        help="log-normal std of per-zone production/attraction")
    parser.add_argument("--sigma-cell", type=float, default=0.35,
                        help="log-normal std of per-OD-cell noise")
    parser.add_argument("--sigma-hour", type=float, default=0.30,
                        help="log-normal std of the per-hour demand shape")
    add_seed(parser, default=1)
    args = parser.parse_args()

    taz = TazSet.from_file(args.taz)
    sampler = PriorSampler(taz, daily_trips=args.daily_trips,
                           sigma_total=args.sigma_total,
                           sigma_zone=args.sigma_zone,
                           sigma_cell=args.sigma_cell,
                           sigma_hour=args.sigma_hour,
                           max_trips=args.max_trips, seed=args.seed)
    n = len(taz)
    print(f"{n} zones ({n * n} OD pairs, {n * n * 24} unknowns), "
          f"center {taz.ids[sampler.center]}")

    os.makedirs(args.out_dir, exist_ok=True)
    tensor = []
    for index, matrix in enumerate(sampler.generate(args.num)):
        matrix.write(os.path.join(args.out_dir, f"od_{index:03d}.xml"))
        tensor.append(matrix.flat.astype(np.float32))
    od = np.stack(tensor)

    np.save(os.path.join(args.out_dir, "od.npy"), od)
    with open(os.path.join(args.out_dir, "zones.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(taz.ids))

    totals = od.sum(axis=(1, 2))
    shape = od.reshape(args.num, -1)
    shape = shape / shape.sum(1, keepdims=True)
    correlation = np.corrcoef(shape)[np.triu_indices(args.num, 1)]
    print(f"wrote {args.num} scenarios to {args.out_dir}")
    print(f"  daily trips: median {np.median(totals):.0f}  "
          f"range [{totals.min():.0f}, {totals.max():.0f}]  "
          f"spread +/-{100 * totals.std() / totals.mean():.0f}%")
    print(f"  OD-shape correlation between scenarios: "
          f"median {np.median(correlation):.3f} "
          f"(generate_od.py gives >0.997 -- lower is more diverse)")


if __name__ == "__main__":
    main()
