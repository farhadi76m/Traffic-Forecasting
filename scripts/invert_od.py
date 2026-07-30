#!/usr/bin/env python3
"""The inverse problem: observed travel times -> a posterior over the OD matrix.

This is the deployment question -- you have travel times, you want demand. The
surrogate replaces the toy BPR link-cost function, and because it is
differentiable the posterior comes from MAP + Laplace instead of NUTS. The
Jacobian d(observation)/d(OD) is the classical assignment matrix; the count of
its eigenvalues above 1 is the honest answer to whether the observations
identify the OD at all.

    python scripts/invert_od.py --model output_sur/cong/model/surrogate.pt \
        --data output_sur/cong/dataset/data.npz --observe traveltime
"""
import os

import _path  # noqa: F401

from traffic_estimate.cli import add_seed, make_parser
from traffic_estimate.ml import ODInversion, SurrogateDataset


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--model", required=True, help="surrogate.pt")
    parser.add_argument("--data", required=True,
                        help="data.npz (for the held-out truth)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--observe", choices=["traveltime", "counts", "both"],
                        default="traveltime",
                        help="what the sensors report; 'traveltime' is the "
                             "deployment case")
    parser.add_argument("--n-sensors", type=int, default=300,
                        help="how many edges are observed (0 = every edge)")
    parser.add_argument("--sigma-tt", type=float, default=0.0,
                        help="noise on log(travel time / free flow). 0 = "
                             "calibrate it from the surrogate's own held-out "
                             "error (recommended)")
    parser.add_argument("--n-cases", type=int, default=5,
                        help="held-out ODs to recover")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-2)
    add_seed(parser, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.model), "inverse")
    inversion = ODInversion(args.model, SurrogateDataset.load(args.data),
                            observe=args.observe, n_sensors=args.n_sensors,
                            sigma_tt=args.sigma_tt, steps=args.steps,
                            lr=args.lr, seed=args.seed)
    inversion.run(args.n_cases, out_dir)


if __name__ == "__main__":
    main()
