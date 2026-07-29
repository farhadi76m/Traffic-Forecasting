#!/usr/bin/env python3
"""Snapshot a TAZ partition: zone polygons over the road network, in lon/lat.

Reads the *_districts.geojson / *_network.geojson pair that
build_taz_districts.py writes for one output folder, and renders a single map.

    python scripts/plot_taz_snapshot.py neshan/b_tehran_l9 --out out.png
"""
import _path  # noqa: F401

from traffic_estimate.cli import make_parser
from traffic_estimate.viz import taz_snapshot, use_theme


def main():
    parser = make_parser(__doc__)
    parser.add_argument("folder",
                        help="folder with *_districts.geojson / *_network.geojson")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--max-edges", type=int, default=120_000,
                        help="subsample road edges above this count, to keep "
                             "huge networks renderable")
    args = parser.parse_args()

    use_theme(headless=True)
    taz_snapshot(args.folder, args.out, args.title, args.max_edges)


if __name__ == "__main__":
    main()
