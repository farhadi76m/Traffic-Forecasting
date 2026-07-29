#!/usr/bin/env python3
"""Build a PDF report of the Tehran TAZ partitions in neshan/b_tehran_*.

For each b_tehran_* output folder (built by build_taz_districts.py) this:
  1. computes zone stats from the *_districts.taz.xml (count, surface, edges),
  2. renders a snapshot of the zone polygons over the road network
     (scripts/plot_taz_snapshot.py),
  3. lays out a summary table + one page per level with its map and stats.

    python scripts/make_taz_report.py --root neshan --out docs/taz_report.pdf
"""
import argparse
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

from fpdf import FPDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_taz_snapshot  # noqa: E402

INK, INK2, MUTED = (11, 11, 11), (82, 81, 78), (137, 135, 129)
BLUE, RULE = (42, 120, 214), (225, 224, 217)

# folders to skip, with the reason shown in the report
SKIP = {
    "b_tehran_l8": "admin_level 8 does not subdivide this network - collapses "
                    "to a single zone covering everywhere, so it carries no "
                    "zoning information (see build_taz_districts.py docstring).",
    "b_tehran_l11.taz.xml": "duplicate re-run of b_tehran_l11 (same network, "
                             "same admin_level, near-identical output) - kept "
                             "on disk but not analysed separately.",
}

LEVEL_NOTE = {
    "b_tehran_l7":     "admin_level 7 (dehestan / rural sub-district) - coarsest "
                        "usable split: 10 huge zones, good for a city-wide "
                        "overview, too coarse for OD calibration.",
    "b_tehran_l9":     "admin_level 9 (mantaqe / municipal district) - Tehran's "
                        "22 official districts, a natural, recognisable zoning.",
    "b_tehran_l10":    "admin_level 10 - an intermediate level between district "
                        "and neighbourhood; 16 zones here after small ones "
                        "were merged into neighbours (--min-edges).",
    "b_tehran_l11":    "admin_level 11 (mahalleh / neighbourhood) on the "
                        "skeleton network - the finest usable split, 251 zones.",
    "b_tehran_l11_new":"same admin_level 11 / same network as b_tehran_l11, "
                        "rebuilt later; fewer zones (223) because more small "
                        "neighbourhoods fell under --min-edges and got merged.",
    "b_tehran_l11_org":"admin_level 11 on tehran2.net.xml, the larger, denser "
                        "network (247k edges vs. 30k) - 342 zones, an order of "
                        "magnitude more edges per zone than the skeleton runs.",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="neshan", help="folder holding b_tehran_* dirs")
    p.add_argument("--out", default="docs/taz_report.pdf")
    p.add_argument("--figdir", default="docs/figures/taz",
                    help="where per-level snapshot PNGs are written")
    p.add_argument("--refresh-figs", action="store_true",
                    help="re-render snapshots even if the PNG already exists")
    return p.parse_args()


def shoelace(pts):
    a = 0.0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def net_source(folder):
    cfgs = glob.glob(os.path.join(folder, "*.sumocfg"))
    if not cfgs:
        return "?"
    m = re.search(r'net-file value="([^"]+)"', open(cfgs[0]).read())
    return os.path.basename(m.group(1)) if m else "?"


def zone_stats(folder):
    taz_f = glob.glob(os.path.join(folder, "*_districts.taz.xml"))[0]
    tazs = list(ET.parse(taz_f).getroot().iter("taz"))
    areas, edges = [], []
    for z in tazs:
        pts = [tuple(map(float, p.split(","))) for p in z.get("shape").split()]
        areas.append(shoelace(pts) / 1e6)          # m^2 -> km^2
        edges.append(len(z.get("edges").split()))
    return {
        "n": len(tazs),
        "area_total": sum(areas),
        "area_avg": sum(areas) / len(areas),
        "area_min": min(areas),
        "area_max": max(areas),
        "edges_total": sum(edges),
        "edges_avg": sum(edges) / len(edges),
        "edges_min": min(edges),
        "edges_max": max(edges),
        "net": net_source(folder),
    }


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*MUTED)
            self.cell(0, 5, "Traffic-Estimate - Tehran TAZ partitions",
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

    def caption(self, txt):
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.2, txt, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def fig(self, path, caption_txt, w=None):
        w = w or self.w - self.l_margin - self.r_margin
        x = (self.w - w) / 2
        self.image(path, x=x, w=w)
        self.ln(1.5)
        self.caption(caption_txt)


def main():
    args = parse_args()
    folders = sorted(
        d for d in glob.glob(os.path.join(args.root, "b_tehran_*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*_districts.taz.xml"))
    )
    if not folders:
        sys.exit(f"no b_tehran_* TAZ output found under {args.root}")

    os.makedirs(args.figdir, exist_ok=True)
    rows = []
    for folder in folders:
        name = os.path.basename(folder)
        skip_reason = SKIP.get(name)
        stats = zone_stats(folder)
        png = None
        if not skip_reason:
            png = os.path.join(args.figdir, f"{name}.png")
            if args.refresh_figs or not os.path.exists(png):
                plot_taz_snapshot.render(folder, png, title=f"{name} - {stats['n']} TAZ zones")
        rows.append((name, stats, skip_reason, png))

    pdf = Report(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()

    pdf.h1("Tehran TAZ Partitions")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK2)
    pdf.multi_cell(0, 5.2,
        "Traffic analysis zones (TAZ) for the Tehran SUMO network, built at "
        "several OpenStreetMap admin_levels by scripts/build_taz_districts.py "
        "(neshan/b_tehran_*). Each level partitions the network's routable car "
        "edges into administrative zones, merging any zone with fewer than "
        "--min-edges edges into its nearest neighbour so every zone is large "
        "enough to route traffic to and from.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.h2("Summary - zones, surface, edges")
    analysed = [(n, s, p) for n, s, sk, p in rows if not sk]
    col = [40, 14, 22, 24, 22, 22]
    heads = ["folder", "TAZ", "avg km²/taz", "total km²", "avg edges/taz", "total edges"]
    pdf.set_font("Helvetica", "B", 8.6)
    pdf.set_text_color(*INK)
    for txt, w in zip(heads, col):
        pdf.cell(w, 6, txt, border="B")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8.6)
    pdf.set_text_color(*INK2)
    for name, s, _ in analysed:
        vals = [name, str(s["n"]), f"{s['area_avg']:.2f}", f"{s['area_total']:.0f}",
                f"{s['edges_avg']:.0f}", str(s["edges_total"])]
        for txt, w in zip(vals, col):
            pdf.cell(w, 6, txt, border="B")
        pdf.ln()
    pdf.ln(3)
    pdf.body(
        "\"avg km²/taz\" and \"avg edges/taz\" are the mean zone surface area "
        "and routable-edge count across all zones in that partition; both "
        "shrink as admin_level goes finer (7 -> 11) or as the underlying "
        "network gets denser (l11 -> l11_org).")

    pdf.h2("Excluded from analysis")
    for name, s, sk, _ in rows:
        if sk:
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(*INK)
            pdf.cell(0, 5.5, name, new_x="LMARGIN", new_y="NEXT")
            pdf.body(sk)

    for name, s, sk, png in rows:
        if sk:
            continue
        pdf.add_page()
        pdf.h2(name)
        pdf.fig(png, LEVEL_NOTE.get(name, ""), w=155)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*INK)
        col2 = [55, 55, 60]
        for txt, w in zip(["metric", "mean", "range (min-max)"], col2):
            pdf.cell(w, 6, txt, border="B")
        pdf.ln()
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*INK2)
        for label, avg, lo, hi, unit in [
            ("zone surface", s["area_avg"], s["area_min"], s["area_max"], "km²"),
            ("edges per zone", s["edges_avg"], s["edges_min"], s["edges_max"], ""),
        ]:
            fmt = (lambda v: f"{v:.2f} {unit}".strip()) if unit else (lambda v: f"{v:.0f}")
            for txt, w in zip([label, fmt(avg), f"{fmt(lo)} - {fmt(hi)}"], col2):
                pdf.cell(w, 6, txt, border="B")
            pdf.ln()
        pdf.ln(2)
        pdf.body(f"{s['n']} zones, {s['edges_total']} routable edges total, "
                  f"built from {s['net']}.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pdf.output(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
