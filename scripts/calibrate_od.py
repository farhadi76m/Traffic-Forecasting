#!/usr/bin/env python3
"""Calibrate OD demand magnitude against REAL measured speeds.

gravity_od.py gives the *shape* of the demand but its magnitude (--daily-trips)
is a guess. This finds the magnitude that makes SUMO reproduce the congestion
TomTom actually measured on the street.

Method: congestion is monotone in demand, so bisect on total daily trips until
the simulated congestion index matches the observed one on the probe edges.

We compare *ratios*, not raw km/h:

    congestion index = mean(speed / freeflow_speed)   over probe edges, peak hours

because SUMO's speed limits and TomTom's freeflow baseline disagree in absolute
terms (OSM maxspeed tags vs probe-derived 85th percentile). The ratio cancels
that out -- 0.5 means "moving at half of free flow" in both worlds.

    python scripts/calibrate_od.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --cost output/cost_matrix.csv \
        --observed output/observed_speeds.csv --out-dir output/od_calibrated

Honest limits: speed pins volume only loosely (a road at half free-flow can be
moderately or heavily loaded -- the two branches of the fundamental diagram).
This gets the order of magnitude right. It is not the same as counting cars.
"""
import argparse
import csv
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--taz", required=True)
    p.add_argument("--cost", required=True)
    p.add_argument("--observed", required=True, help="CSV from fetch_speeds.py")
    p.add_argument("--out-dir", default="output/od_calibrated")
    p.add_argument("--work", default="output/calib_work")
    p.add_argument("--peak-hours", default="7,8",
                   help="hours to match on; congestion is only informative when "
                        "the network is actually loaded. Keep this contiguous -- "
                        "the whole span gets simulated each iteration, so adding "
                        "the evening peak means simulating all day")
    p.add_argument("--lo", type=int, default=10_000, help="lower bracket, trips/day")
    p.add_argument("--hi", type=int, default=1_000_000, help="upper bracket")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--tol", type=float, default=0.02,
                   help="stop when congestion index is within this of observed")
    p.add_argument("--beta", type=float, default=0.12)
    p.add_argument("--num", type=int, default=20,
                   help="scenario days to emit once calibrated")
    return p.parse_args()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(" ".join(map(str, cmd)) + "\n" + r.stderr[-2000:])
    return r.stderr


def observed_index(path, peak):
    """Mean speed/freeflow over probe edges in peak hours -- the target."""
    ratios, edges = [], set()
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["hour"]) not in peak:
                continue
            free = float(row["freeflow_kmh"])
            if free <= 0 or float(row["confidence"]) < 0.5:
                continue
            ratios.append(float(row["current_kmh"]) / free)
            edges.add(row["edge_id"])
    if not ratios:
        sys.exit(f"{path} has no usable rows for peak hours {sorted(peak)}.\n"
                 "Did fetch_speeds.py run across those hours?")
    return float(np.mean(ratios)), edges


def simulated_index(args, trips_per_day, probe_edges, peak, tag):
    """Build OD at this magnitude, simulate, return the same congestion index.

    Only the peak window is simulated -- that is the only part we measure, and
    a full 24h run per bisection iteration is far too slow. One warm-up hour
    ahead of the peak lets congestion build before we start scoring.
    """
    wdir = os.path.join(args.work, tag)
    os.makedirs(wdir, exist_ok=True)
    od_dir = os.path.join(wdir, "od")
    begin = (min(peak) - 1) * 3600          # warm-up hour
    end = (max(peak) + 1) * 3600

    run([sys.executable, os.path.join(HERE, "gravity_od.py"),
         "--net", args.net, "--taz", args.taz, "--cost", args.cost,
         "--out-dir", od_dir, "--num", "1", "--beta", str(args.beta),
         "--daily-trips", str(int(trips_per_day)), "--noise", "0"])

    od = os.path.join(od_dir, "od_00.xml")
    trips = os.path.join(wdir, "trips.xml")
    routes = os.path.join(wdir, "routes.xml")
    edgedata = os.path.join(wdir, "edgedata.xml")

    run(["od2trips", "--taz-files", args.taz, "--tazrelation-files", od,
         "-o", trips, "--ignore-vehicle-type", "--seed", "7",
         "--begin", str(begin), "--end", str(end),
         "--departpos", "random", "--arrivalpos", "random"])
    run(["duarouter", "-n", args.net, "--route-files", trips, "-o", routes,
         "--begin", str(begin), "--end", str(end),
         "--ignore-errors", "--no-warnings", "--repair", "--seed", "7"])

    add = os.path.join(wdir, "ed.add.xml")
    with open(add, "w") as f:
        f.write(f'<additional>\n  <edgeData id="ed" file="{os.path.abspath(edgedata)}"'
                f' period="3600" excludeEmpty="true"/>\n</additional>\n')
    run(["sumo", "-n", args.net, "-r", routes, "--additional-files", add,
         "--begin", str(begin), "--end", str(end), "--mesosim", "--seed", "7",
         "--no-step-log", "--no-warnings", "--ignore-route-errors"])

    net = sumolib.net.readNet(args.net, withInternal=False)
    ratios = []
    for iv in ET.parse(edgedata).getroot().iter("interval"):
        hour = int(float(iv.get("begin")) // 3600)
        if hour not in peak:
            continue
        for e in iv.iter("edge"):
            eid = e.get("id")
            if eid not in probe_edges or e.get("speed") is None:
                continue
            limit = net.getEdge(eid).getSpeed()
            if limit > 0:
                ratios.append(min(float(e.get("speed")) / limit, 1.0))
    if not ratios:
        raise RuntimeError("no probe edges appeared in the simulation output")
    return float(np.mean(ratios))


def main():
    args = parse_args()
    peak = {int(h) for h in args.peak_hours.split(",")}
    target, probe_edges = observed_index(args.observed, peak)
    print(f"observed congestion index: {target:.3f} "
          f"({len(probe_edges)} probe edges, hours {sorted(peak)})")
    print(f"  -> real traffic moves at {target * 100:.0f}% of free flow in peak\n")

    lo, hi = args.lo, args.hi
    best = None
    for i in range(args.iters):
        mid = int(np.sqrt(lo * hi))  # geometric: demand spans orders of magnitude
        idx = simulated_index(args, mid, probe_edges, peak, f"it{i}")
        err = idx - target
        print(f"  iter {i}: {mid:>8,} trips/day -> index {idx:.3f} "
              f"({'too free' if err > 0 else 'too jammed'}, err {err:+.3f})")
        if best is None or abs(err) < abs(best[1] - target):
            best = (mid, idx)
        if abs(err) < args.tol:
            print("  converged")
            break
        if err > 0:
            lo = mid       # simulation flows too freely -> needs more demand
        else:
            hi = mid
    trips, idx = best
    print(f"\ncalibrated: {trips:,} trips/day (sim index {idx:.3f} vs "
          f"observed {target:.3f})")

    run([sys.executable, os.path.join(HERE, "gravity_od.py"),
         "--net", args.net, "--taz", args.taz, "--cost", args.cost,
         "--out-dir", args.out_dir, "--num", str(args.num),
         "--beta", str(args.beta), "--daily-trips", str(trips)])
    print(f"{args.num} calibrated scenarios -> {args.out_dir}")


if __name__ == "__main__":
    main()
