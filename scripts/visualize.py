#!/usr/bin/env python3
"""Visualize model predictions: network heatmaps, edge profiles, accuracy.

Writes four PNGs to --out-dir:
  network_traffic.png   map coloured by predicted vehicles/hour at key hours
  edge_profiles.png     24h predicted vs simulated profiles, busiest edges
  pred_vs_actual.png    prediction vs typical-day simulated traffic
  daily_profile.png     network-wide traffic volume over the day

    python scripts/visualize.py --net sumo/tehran_2026_area.net.xml \
        --data-dir output/dataset --model-dir output/model --out-dir output/figures
"""
import _path  # noqa: F401

from traffic_estimate.cli import add_network, add_out_dir, load_network, make_parser
from traffic_estimate.ml import ForecastDataset, Predictor
from traffic_estimate.viz import ForecastFigures, use_theme
# re-exported for demo.ipynb
from traffic_estimate.viz.theme import (AQUA, BLUE, CMAP, INK, INK2,  # noqa: F401
                                        MUTED, SERIES, SURFACE)


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    add_out_dir(parser)
    parser.add_argument("--hours", type=int, nargs="+", default=[7, 13, 18, 23])
    parser.add_argument("--top-edges", type=int, default=6)
    args = parser.parse_args()

    use_theme(headless=True)  # notebooks import the module and keep inline
    network = load_network(args.net)
    dataset = ForecastDataset.load(args.data_dir)
    predictor = Predictor.load(args.model_dir)
    ForecastFigures(network, dataset.frame, predictor, args.out_dir,
                    hours=args.hours, top_edges=args.top_edges).render_all()


if __name__ == "__main__":
    main()
