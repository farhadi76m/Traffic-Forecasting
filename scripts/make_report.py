#!/usr/bin/env python3
"""Build a minimal PDF report: method, model usage, results, figures.

Reads metrics from the trained model's meta.json and embeds the figures
from docs/figures (regenerate them with scripts/visualize.py if stale).

Example:
    python scripts/make_report.py --model-dir output/model \
        --figures docs/figures --out docs/report.pdf
"""
import argparse
import json
import os

from fpdf import FPDF

INK, INK2, MUTED = (11, 11, 11), (82, 81, 78), (137, 135, 129)
BLUE, RULE = (42, 120, 214), (225, 224, 217)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="output/model")
    p.add_argument("--figures", default="docs/figures")
    p.add_argument("--out", default="docs/report.pdf")
    return p.parse_args()


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*MUTED)
            self.cell(0, 5, "Traffic-Estimate - hourly traffic forecasting with SUMO",
                      align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, str(self.page_no()), align="C")

    def h1(self, txt):
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*INK)
        self.cell(0, 10, txt, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, txt):
        self.ln(3)
        self.set_font("Helvetica", "B", 12.5)
        self.set_text_color(*INK)
        self.cell(0, 7, txt, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2.5)

    def body(self, txt):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*INK2)
        self.multi_cell(0, 5, txt, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def mono(self, txt):
        self.set_fill_color(246, 246, 244)
        self.set_font("Courier", "", 8.6)
        self.set_text_color(*INK)
        self.multi_cell(0, 4.6, txt, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def caption(self, txt):
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.2, txt, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def fig(self, path, caption, w=None):
        w = w or self.w - self.l_margin - self.r_margin
        x = (self.w - w) / 2
        self.image(path, x=x, w=w)
        self.ln(1.5)
        self.caption(caption)


def main():
    args = parse_args()
    with open(os.path.join(args.model_dir, "meta.json")) as f:
        meta = json.load(f)
    m_day = meta["metrics"]["test"]
    m_typ = meta["metrics"]["test_typical"]
    F = args.figures

    pdf = Report(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()

    pdf.h1("Traffic-Estimate")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK2)
    pdf.multi_cell(0, 5.2,
        "Hourly traffic forecasting on a SUMO road network (Tehran, District 5 "
        "area, 3,229 car edges). Synthetic 24-hour travel demand is simulated "
        "with SUMO's mesoscopic engine over many days, and a neural network "
        "learns each edge's typical daily traffic profile - a Google-Maps-style "
        "\"typical traffic\" estimator.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.h2("Method")
    pdf.body(
        "1.  Zoning - a 3x3 grid of traffic analysis zones (TAZ) over the car "
        "edges of the network's largest strongly-connected component "
        "(scripts/build_taz.py).\n"
        "2.  Demand - 60 synthetic days as SUMO tazRelation OD matrices, "
        "~6,000 trips/day: gravity-weighted zone pairs, commute direction into "
        "the central zone at 07-09 and out at 16-19, peaks at 07:00, 13:00 and "
        "17-19:00, Poisson day-to-day noise (scripts/generate_od.py).\n"
        "3.  Simulation - od2trips -> duarouter -> sumo --mesosim, 24 h per "
        "day, ~3.5 s per day; hourly per-edge vehicle counts via edgeData "
        "(scripts/run_sim.py).\n"
        "4.  Dataset - 4.6 M rows of (day, edge, hour, vehicles entered) plus "
        "static edge features: length, speed limit, lanes, priority "
        "(scripts/build_dataset.py).\n"
        "5.  Model - a 64-d learned embedding per edge, an hour embedding with "
        "sin/cos time encoding and the static features feed a 2x256 MLP; a "
        "per-edge Fourier head (6 daily harmonics) gives every edge its own "
        "smooth 24 h profile. Trained with Poisson negative log-likelihood, "
        "the correct loss for counts, on 49 days; 3 validation and 8 test days "
        "are held out (scripts/train.py). A GRU profile decoder and a "
        "per-(edge,hour) historical-average table are available as "
        "alternatives and score the same, so the MLP is at the noise ceiling.")

    pdf.h2("Model usage")
    pdf.body(
        "Input:   edge id (any SUMO edge of the network) + hour of day (0-23; "
        "\"17:30\" is accepted and truncated to the hour).\n"
        "Output:  expected vehicles entering that edge during that hour on a "
        "typical day (vehicles/hour).")
    pdf.mono(
        "$ python scripts/predict.py --model-dir output/model \\\n"
        "      --edge \"330920957#0\" --time 7\n"
        "edge 330920957#0 at 07:00 -> 85.8 vehicles/hour\n"
        "\n"
        "$ python scripts/predict.py --model-dir output/model \\\n"
        "      --edge \"330920957#0\" --profile        # full 24h curve\n"
        "\n"
        "# python API\n"
        "from predict import load_predictor\n"
        "meta, fn = load_predictor(\"output/model\")\n"
        "rates = fn(meta[\"edges\"].index(\"330920957#0\"))[0]  # 24 hourly rates")

    pdf.h2("Results (held-out simulated days)")
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*INK)
    col = [72, 50, 50]
    for txt, w in zip(["metric", "single day", "typical day"], col):
        pdf.cell(w, 6, txt, border="B")
    pdf.ln()
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*INK2)
    rows = [
        ("accuracy (within 20% or 3 veh)",
         f"{m_day['accuracy']*100:.1f}%", f"{m_typ['accuracy']*100:.1f}%"),
        ("R2", f"{m_day['r2']:.3f}", f"{m_typ['r2']:.3f}"),
        ("MAE (veh/h)", f"{m_day['mae']:.2f}", f"{m_typ['mae']:.2f}"),
    ]
    for r in rows:
        for txt, w in zip(r, col):
            pdf.cell(w, 6, txt, border="B")
        pdf.ln()
    pdf.ln(2)
    pdf.body(
        "\"Typical day\" scores the prediction against the average of the test "
        "days - the quantity a typical-traffic forecast estimates. The ~5% "
        "single-day residual is the irreducible Poisson randomness of daily "
        "demand: an oracle lookup table scores the same.")

    pdf.add_page()
    pdf.h2("Estimated traffic across the network")
    pdf.fig(os.path.join(F, "network_traffic.png"),
            "Predicted vehicles/hour on every edge at 07:00, 13:00, 18:00 and "
            "23:00. Arterials carry the load; 18:00 is the heaviest hour.",
            w=150)

    pdf.add_page()
    pdf.h2("Model vs simulation")
    pdf.fig(os.path.join(F, "edge_profiles.png"),
            "24 h profiles of the six busiest edges: model prediction (blue) "
            "vs simulated median and 10-90% band across days (green).")
    pdf.fig(os.path.join(F, "pred_vs_actual.png"),
            "Prediction vs typical-day simulated traffic for every (edge, "
            "hour) pair, log scale.", w=105)

    pdf.add_page()
    pdf.h2("Network-wide daily demand")
    pdf.fig(os.path.join(F, "daily_profile.png"),
            "Total vehicle entries per hour across all edges (mean and 10-90% "
            "band over the 60 simulated days): morning peak at 07:00, midday "
            "bump at 13:00, evening peak at 18:00.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pdf.output(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
