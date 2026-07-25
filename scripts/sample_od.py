#!/usr/bin/env python3
"""Sample OD matrices from a wide prior, for training a SUMO surrogate.

`generate_od.py` makes many *days of the same city*: one gravity matrix plus
Poisson jitter, so every scenario is the same OD (shape correlation > 0.997).
That is right for forecasting a typical day, but a surrogate trained on it
learns the mean and nothing about how traffic responds to demand.

Here every scenario is a *different city*: total demand, per-zone production
and attraction, cell-level flows, the 24h shape and the commute directionality
are all drawn from a log-normal prior. This is the sampling distribution the
surrogate is trained on, and therefore also the prior of the inverse problem.

Writes od_XX.xml (SUMO tazRelation, for od2trips) and od.npy, an
(n_scenarios, 24, n_zones*n_zones) tensor of the same numbers.

Example:
    python scripts/sample_od.py --taz output/taz/taz_grid9.xml \
        --out-dir output_sur/light/od --num 300 --daily-trips 12000
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

# base relative demand for each hour: morning peak, midday bump, evening peak
HOUR_PROFILE = np.array([
    0.5, 0.3, 0.25, 0.25, 0.4, 1.0,
    2.5, 4.5, 4.2, 2.8, 2.2, 2.3,
    2.7, 3.1, 2.7, 2.8, 3.5, 4.4,
    4.6, 3.5, 2.5, 1.8, 1.2, 0.8,
])
MORNING = np.zeros(24); MORNING[[6, 7, 8, 9]] = [0.5, 1.0, 0.9, 0.4]
EVENING = np.zeros(24); EVENING[[13, 16, 17, 18, 19]] = [0.3, 0.5, 0.9, 1.0, 0.5]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num", type=int, default=300, help="number of OD scenarios")
    p.add_argument("--daily-trips", type=int, default=12000,
                   help="median total trips per day; the prior spreads around it")
    p.add_argument("--sigma-total", type=float, default=0.35,
                   help="log-normal std of the total demand multiplier")
    p.add_argument("--max-trips", type=int, default=0,
                   help="clip total daily trips (0 = no clip). Past ~130k this net "
                        "gridlocks: travel times hit 500x free-flow and SUMO takes "
                        "minutes per run, so the congested regime needs a ceiling.")
    p.add_argument("--sigma-zone", type=float, default=0.45,
                   help="log-normal std of per-zone production/attraction")
    p.add_argument("--sigma-cell", type=float, default=0.35,
                   help="log-normal std of per-OD-cell noise")
    p.add_argument("--sigma-hour", type=float, default=0.30,
                   help="log-normal std of the per-hour demand shape")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    zones, sizes = [], []
    for taz in ET.parse(args.taz).getroot().iter("taz"):
        zones.append(taz.get("id"))
        sizes.append(len(taz.get("edges").split()))
    n = len(zones)
    size_w = np.array(sizes, float) / sum(sizes)
    center = int(np.argmax(sizes))
    print(f"{n} zones ({n*n} OD pairs, {n*n*24} unknowns), center {zones[center]}")

    od_tensor = np.zeros((args.num, 24, n * n), dtype=np.float32)

    for s in range(args.num):
        ln = lambda sig, size=None: np.exp(rng.normal(0, sig, size))
        total = args.daily_trips * ln(args.sigma_total)
        if args.max_trips:
            total = min(total, args.max_trips)
        produce = size_w * ln(args.sigma_zone, n)
        attract = size_w * ln(args.sigma_zone, n)
        cell = ln(args.sigma_cell, (n, n))
        hour_w = HOUR_PROFILE * ln(args.sigma_hour, 24)
        hour_frac = hour_w / hour_w.sum()
        commute = rng.uniform(0.0, 3.0)  # how strongly this city commutes

        root = ET.Element("data")
        for h in range(24):
            w = np.outer(produce, attract) * cell
            w[:, center] *= 1 + commute * MORNING[h]
            w[center, :] *= 1 + commute * EVENING[h]
            w /= w.sum()
            counts = rng.poisson(w * total * hour_frac[h])
            od_tensor[s, h] = counts.reshape(-1)

            iv = ET.SubElement(root, "interval", id=f"h{h}",
                               begin=str(h * 3600), end=str((h + 1) * 3600))
            for i in range(n):
                for j in range(n):
                    if counts[i, j] > 0:
                        ET.SubElement(iv, "tazRelation", From=zones[i], to=zones[j],
                                      count=str(int(counts[i, j])))

        out = os.path.join(args.out_dir, f"od_{s:03d}.xml")
        ET.indent(tree := ET.ElementTree(root))
        tree.write(out, encoding="UTF-8", xml_declaration=True)
        # ElementTree forbids the reserved 'from' kwarg; fix the attribute name
        with open(out) as f:
            text = f.read().replace("From=", "from=")
        with open(out, "w") as f:
            f.write(text)

    np.save(os.path.join(args.out_dir, "od.npy"), od_tensor)
    with open(os.path.join(args.out_dir, "zones.txt"), "w") as f:
        f.write("\n".join(zones))

    tot = od_tensor.sum(axis=(1, 2))
    print(f"wrote {args.num} scenarios to {args.out_dir}")
    print(f"  daily trips: median {np.median(tot):.0f}  "
          f"range [{tot.min():.0f}, {tot.max():.0f}]  spread +/-{100*tot.std()/tot.mean():.0f}%")
    shape = od_tensor.reshape(args.num, -1)
    shape = shape / shape.sum(1, keepdims=True)
    c = np.corrcoef(shape)
    iu = np.triu_indices(args.num, 1)
    print(f"  OD-shape correlation between scenarios: median {np.median(c[iu]):.3f} "
          f"(generate_od.py gives >0.997 -- lower is more diverse)")


if __name__ == "__main__":
    main()
