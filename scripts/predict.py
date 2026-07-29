#!/usr/bin/env python3
"""Predict traffic (vehicles/hour) for an edge at a given time.

    python scripts/predict.py --model-dir output/model --edge 150770686#0 --time 8
    python scripts/predict.py --model-dir output/model --edge 150770686#0 --time 17:30
    python scripts/predict.py --model-dir output/model --edge 150770686#0 --profile
"""
import _path  # noqa: F401

from traffic_estimate.cli import make_parser, parse_hour
from traffic_estimate.ml import Predictor


def load_predictor(model_dir):
    """(meta, fn) where fn(edge_indices) -> (n, 24) hourly rates.

    Kept for demo.ipynb; new code should use Predictor.load directly.
    """
    predictor = Predictor.load(model_dir)
    return predictor.meta, predictor.rates


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--edge", required=True, help="edge id from the network")
    parser.add_argument("--time", help="hour of day, e.g. 7 or 07:30")
    parser.add_argument("--profile", action="store_true",
                        help="print all 24 hours")
    args = parser.parse_args()
    if not args.time and not args.profile:
        raise SystemExit("give --time HH[:MM] and/or --profile")

    predictor = Predictor.load(args.model_dir)
    if args.profile:
        print(predictor.format_profile(args.edge))
    if args.time is not None:
        hour = parse_hour(args.time)
        print(f"edge {args.edge} at {hour:02d}:00 -> "
              f"{predictor.at(args.edge, hour):.1f} vehicles/hour")


if __name__ == "__main__":
    main()
