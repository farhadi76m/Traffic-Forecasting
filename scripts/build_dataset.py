#!/usr/bin/env python3
"""Turn hourly edgeData simulation output into a training dataset.

Writes two files in --out-dir:
  traffic.csv.gz  one row per (scenario, edge, hour) with the vehicles that
                  entered the edge that hour; unobserved edge-hours are 0
  edges.csv       static per-edge features: length, speed limit, lanes, priority

    python scripts/build_dataset.py --net sumo/tehran_2026_area.net.xml \
        --sim-dir output/sim --out-dir output/dataset
"""
import _path  # noqa: F401

from traffic_estimate.cli import add_network, add_out_dir, load_network, make_parser
from traffic_estimate.ml import ForecastDataset


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    parser.add_argument("--sim-dir", required=True)
    add_out_dir(parser)
    args = parser.parse_args()

    network = load_network(args.net)
    ForecastDataset.build(network, args.sim_dir).save(args.out_dir)


if __name__ == "__main__":
    main()
