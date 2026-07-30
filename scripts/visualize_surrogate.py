#!/usr/bin/env python3
"""Figures for the surrogate and inverse experiment.

  congestion.png       travel time vs free flow in each regime -- why the
                       inverse problem is solvable in one and not the other
  surrogate.png        surrogate vs SUMO on held-out OD, against the OD-blind
                       baseline the plain (edge, hour) model is equivalent to
  identifiability.png  spectrum of J'J: how many OD directions the observations
                       actually constrain
  recovery.png         true vs recovered OD with posterior error bars

    python scripts/visualize_surrogate.py --root output_sur --out-dir docs/figures
"""
import _path  # noqa: F401

from traffic_estimate.cli import make_parser
from traffic_estimate.viz import SurrogateFigures, use_theme


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--root", default="output_sur",
                        help="folder holding one directory per demand regime")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()

    use_theme(headless=True)
    SurrogateFigures(args.root, args.out_dir).render_all()


if __name__ == "__main__":
    main()
