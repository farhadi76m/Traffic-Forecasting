#!/usr/bin/env python3
"""Pair each sampled OD matrix with what SUMO did to it -> surrogate training set.

build_dataset.py keeps only counts and throws the OD away, because the
forecaster never sees it. The surrogate needs both sides of the map:

    OD (24, zones^2)  ->  SUMO  ->  counts + travel times (edges, 24)

An edge absent from an hour's interval carried no vehicle (excludeEmpty), so its
count is 0 and its travel time is the free-flow time -- the physically right
fill, not a missing value.

    python scripts/build_surrogate_dataset.py --net sumo/tehran_2026_area.net.xml \
        --od-dir output_sur/cong/od --sim-dir output_sur/cong/sim \
        --out-dir output_sur/cong/dataset
"""
import _path  # noqa: F401

from traffic_estimate.cli import add_network, add_out_dir, load_network, make_parser
from traffic_estimate.ml import SurrogateDataset


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    parser.add_argument("--od-dir", required=True, help="dir with od.npy + od_*.xml")
    parser.add_argument("--sim-dir", required=True)
    add_out_dir(parser)
    args = parser.parse_args()

    network = load_network(args.net)
    dataset = SurrogateDataset.build(network, args.od_dir, args.sim_dir)
    path = dataset.save(args.out_dir)
    print(f"\nwrote {path}")
    print(dataset.summary())


if __name__ == "__main__":
    main()
