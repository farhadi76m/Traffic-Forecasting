#!/usr/bin/env python3
"""Build the project PDF report: method, model usage, results, figures.

Reads metrics from the trained model's meta.json and embeds the figures from
--figures (regenerate them with scripts/visualize.py if stale).

    python scripts/make_report.py --model-dir output/model \
        --figures docs/figures --out docs/report.pdf
"""
import os

import _path  # noqa: F401

from traffic_estimate.cli import make_parser
from traffic_estimate.ml import Predictor
from traffic_estimate.viz import PdfReport

HEAD = "Traffic-Estimate - hourly traffic forecasting with SUMO"

METHOD = (
    "1.  Zoning - a 3x3 grid of traffic analysis zones (TAZ) over the car edges "
    "of the network's largest strongly-connected component "
    "(scripts/build_taz.py).\n"
    "2.  Demand - 60 synthetic days as SUMO tazRelation OD matrices, ~6,000 "
    "trips/day: gravity-weighted zone pairs, commute direction into the central "
    "zone at 07-09 and out at 16-19, peaks at 07:00, 13:00 and 17-19:00, "
    "Poisson day-to-day noise (scripts/generate_od.py).\n"
    "3.  Simulation - od2trips -> duarouter -> sumo --mesosim, 24 h per day, "
    "~3.5 s per day; hourly per-edge vehicle counts via edgeData "
    "(scripts/run_sim.py).\n"
    "4.  Dataset - 4.6 M rows of (day, edge, hour, vehicles entered) plus static "
    "edge features: length, speed limit, lanes, priority "
    "(scripts/build_dataset.py).\n"
    "5.  Model - a 64-d learned embedding per edge, an hour embedding with "
    "sin/cos time encoding and the static features feed a 2x256 MLP; a per-edge "
    "Fourier head (6 daily harmonics) gives every edge its own smooth 24 h "
    "profile. Trained with Poisson negative log-likelihood, the correct loss for "
    "counts, on 49 days; 3 validation and 8 test days are held out "
    "(scripts/train.py). A GRU profile decoder and a per-(edge,hour) historical "
    "average score the same, so the MLP is at the noise ceiling.")

USAGE = (
    "Input:   edge id (any SUMO edge of the network) + hour of day (0-23; "
    "\"17:30\" is accepted and truncated to the hour).\n"
    "Output:  expected vehicles entering that edge during that hour on a "
    "typical day (vehicles/hour).")

EXAMPLE = (
    "$ python scripts/predict.py --model-dir output/model \\\n"
    "      --edge \"330920957#0\" --time 7\n"
    "edge 330920957#0 at 07:00 -> 85.8 vehicles/hour\n"
    "\n"
    "$ python scripts/predict.py --model-dir output/model \\\n"
    "      --edge \"330920957#0\" --profile        # full 24h curve\n"
    "\n"
    "# python API\n"
    "from traffic_estimate.ml import Predictor\n"
    "predictor = Predictor.load(\"output/model\")\n"
    "rates = predictor.profile(\"330920957#0\")   # 24 hourly rates")


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--model-dir", default="output/model")
    parser.add_argument("--figures", default="docs/figures")
    parser.add_argument("--out", default="docs/report.pdf")
    args = parser.parse_args()

    metrics = Predictor.load(args.model_dir).metrics
    day, typical = metrics["test"], metrics["test_typical"]
    figure = lambda name: os.path.join(args.figures, name)

    pdf = PdfReport(HEAD)
    pdf.add_page()
    pdf.h1("Traffic-Estimate")
    pdf.body(
        "Hourly traffic forecasting on a SUMO road network (Tehran, District 5 "
        "area, 3,229 car edges). Synthetic 24-hour travel demand is simulated "
        "with SUMO's mesoscopic engine over many days, and a neural network "
        "learns each edge's typical daily traffic profile - a Google-Maps-style "
        "\"typical traffic\" estimator.", size=10.5)

    pdf.h2("Method")
    pdf.body(METHOD)
    pdf.h2("Model usage")
    pdf.body(USAGE)
    pdf.mono(EXAMPLE)

    pdf.h2("Results (held-out simulated days)")
    pdf.table(["metric", "single day", "typical day"],
              [("accuracy (within 20% or 3 veh)",
                f"{day['accuracy'] * 100:.1f}%", f"{typical['accuracy'] * 100:.1f}%"),
               ("R2", f"{day['r2']:.3f}", f"{typical['r2']:.3f}"),
               ("MAE (veh/h)", f"{day['mae']:.2f}", f"{typical['mae']:.2f}")],
              widths=[72, 50, 50])
    pdf.body(
        "\"Typical day\" scores the prediction against the average of the test "
        "days - the quantity a typical-traffic forecast estimates. The ~5% "
        "single-day residual is the irreducible Poisson randomness of daily "
        "demand: an oracle lookup table scores the same.")

    pdf.add_page()
    pdf.h2("Estimated traffic across the network")
    pdf.figure(figure("network_traffic.png"),
               "Predicted vehicles/hour on every edge at 07:00, 13:00, 18:00 and "
               "23:00. Arterials carry the load; 18:00 is the heaviest hour.",
               width=150)

    pdf.add_page()
    pdf.h2("Model vs simulation")
    pdf.figure(figure("edge_profiles.png"),
               "24 h profiles of the six busiest edges: model prediction (blue) "
               "vs simulated median and 10-90% band across days (green).")
    pdf.figure(figure("pred_vs_actual.png"),
               "Prediction vs typical-day simulated traffic for every (edge, "
               "hour) pair, log scale.", width=105)

    pdf.add_page()
    pdf.h2("Network-wide daily demand")
    pdf.figure(figure("daily_profile.png"),
               "Total vehicle entries per hour across all edges (mean and 10-90% "
               "band over the 60 simulated days): morning peak at 07:00, midday "
               "bump at 13:00, evening peak at 18:00.")

    pdf.save(args.out)


if __name__ == "__main__":
    main()
