#!/usr/bin/env python3
"""Generate several 24-hour OD matrices in SUMO tazRelation XML format.

Each scenario is one simulated "day": a gravity-style OD matrix between the
TAZ zones, shaped over 24 hours with a morning peak (~07:00, commute toward
the central zone), a midday bump (~13:00) and an evening peak (17:00-19:00,
commute back). Day-to-day variation comes from a per-scenario demand factor
plus Poisson sampling of every OD cell.

Example:
    python scripts/generate_od.py --taz output/taz/taz_grid9.xml \
        --out-dir output/od --num 20 --daily-trips 6000
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

# relative demand weight for each hour 0..23 (normalized later)
HOUR_PROFILE = np.array([
    0.5, 0.3, 0.25, 0.25, 0.4, 1.0,          # 00-05  night
    2.5, 4.5, 4.2, 2.8, 2.2, 2.3,             # 06-11  morning peak at 07-08
    2.7, 3.1, 2.7, 2.8, 3.5, 4.4,             # 12-17  midday bump, evening rise
    4.6, 3.5, 2.5, 1.8, 1.2, 0.8,             # 18-23  evening peak, wind-down
])
# how strongly each hour is a "toward work" / "toward home" commute hour
MORNING = np.zeros(24); MORNING[[6, 7, 8, 9]] = [0.5, 1.0, 0.9, 0.4]
EVENING = np.zeros(24); EVENING[[13, 16, 17, 18, 19]] = [0.3, 0.5, 0.9, 1.0, 0.5]
COMMUTE_STRENGTH = 1.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taz", required=True, help="TAZ file with zones")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num", type=int, default=20, help="number of scenarios (days)")
    p.add_argument("--daily-trips", type=int, default=6000,
                   help="average total trips per day (default 6000)")
    p.add_argument("--noise", type=float, default=0.08,
                   help="std of per-scenario demand factor (default 0.08)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    zones, sizes = [], []
    for taz in ET.parse(args.taz).getroot().iter("taz"):
        zones.append(taz.get("id"))
        sizes.append(len(taz.get("edges").split()))
    sizes = np.array(sizes, float)
    n = len(zones)
    center = int(np.argmax(sizes))  # biggest zone acts as the work/CBD zone
    print(f"{n} zones, center zone: {zones[center]}")

    size_w = sizes / sizes.sum()
    hour_frac = HOUR_PROFILE / HOUR_PROFILE.sum()

    for s in range(args.num):
        day_factor = float(np.clip(rng.normal(1.0, args.noise), 0.75, 1.25))
        root = ET.Element("data")
        total = 0
        for h in range(24):
            # gravity base: production_i * attraction_j
            w = np.outer(size_w, size_w)
            # commute directionality: morning into center, evening out of it
            w[:, center] *= 1 + COMMUTE_STRENGTH * MORNING[h]
            w[center, :] *= 1 + COMMUTE_STRENGTH * EVENING[h]
            w /= w.sum()
            lam = w * args.daily_trips * hour_frac[h] * day_factor
            counts = rng.poisson(lam)
            interval = ET.SubElement(root, "interval", id=f"h{h}",
                                     begin=str(h * 3600), end=str((h + 1) * 3600))
            for i in range(n):
                for j in range(n):
                    if counts[i, j] > 0:
                        ET.SubElement(interval, "tazRelation", From=zones[i],
                                      to=zones[j], count=str(int(counts[i, j])))
            total += int(counts.sum())
        # ElementTree forbids the reserved 'from' kwarg; fix attribute name
        out = os.path.join(args.out_dir, f"od_{s:02d}.xml")
        ET.indent(tree := ET.ElementTree(root))
        tree.write(out, encoding="UTF-8", xml_declaration=True)
        with open(out) as f:
            text = f.read().replace("From=", "from=")
        with open(out, "w") as f:
            f.write(text)
        print(f"{out}: {total} trips (day factor {day_factor:.2f})")


if __name__ == "__main__":
    main()
