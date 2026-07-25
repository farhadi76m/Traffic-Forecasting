#!/usr/bin/env python3
"""Observe REAL Tehran travel times between TAZ pairs, via Neshan (neshan.org).

TomTom and HERE do not serve Iran. Neshan is an Iranian mapping platform, so it
works from a Tehran IP and actually has live Tehran traffic. Get a free key at
platform.neshan.org (Persian; the free tier is generous enough for this).

For each probe zone-pair we record:
    freeflow_s  shortest-path time on an EMPTY net (our own speed limits)
    observed_s  Neshan's traffic-aware driving time, right now (measured)
    ratio       freeflow_s / observed_s -- 1.0 = free flowing, 0.4 = crawling

That ratio is the calibration target: we scale OD demand until SUMO reproduces
it on the same routes. Because both sides are route travel times on the same
road network, this is directly comparable -- no unit or baseline fudging.

    # check the key works
    python scripts/fetch_neshan.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --once

    # a real observed 24h profile (leave running for a day)
    python scripts/fetch_neshan.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --poll 3600 --hours 24 \
        --out output/observed_neshan.csv
"""
import argparse
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import requests
import sumolib

NESHAN = "https://api.neshan.org/v4/direction"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--taz", required=True)
    p.add_argument("--out", default="output/observed_neshan.csv")
    p.add_argument("--api-key", default=os.environ.get("NESHAN_API_KEY"))
    p.add_argument("--pairs", type=int, default=20,
                   help="zone pairs to probe per round (1 API call each)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll", type=int, default=3600)
    p.add_argument("--hours", type=float, default=24)
    return p.parse_args()


def zone_reps(net, taz_file):
    """One representative edge per zone (same rule as cost_matrix.py)."""
    reps = {}
    for taz in sumolib.xml.parse(taz_file, "taz"):
        pts = [tuple(map(float, p.split(","))) for p in taz.shape.split()]
        cx, cy = np.mean(pts, axis=0)
        cand = [net.getEdge(e) for e in taz.edges.split()
                if net.hasEdge(e) and net.getEdge(e).allows("passenger")]
        if not cand:
            continue
        mid = lambda e: e.getShape()[len(e.getShape()) // 2]  # noqa: E731
        reps[taz.id] = min(cand, key=lambda e: (mid(e)[0] - cx) ** 2 + (mid(e)[1] - cy) ** 2)
    return reps


def probe_pairs(net, reps, n):
    """Pick routable zone pairs, preferring long trips.

    Short hops barely move the congestion ratio -- a 2-minute trip is 2 minutes
    in free flow and 3 in a jam. Long crosstown trips are where congestion shows
    up, so they carry the calibration signal.
    """
    zones = sorted(reps)
    cand = []
    for i, a in enumerate(zones):
        for b in zones:
            if a == b:
                continue
            route, ff = net.getOptimalPath(reps[a], reps[b], fastest=True,
                                           vClass="passenger")
            if route and ff > 120:  # skip trivially short pairs
                cand.append((a, b, ff, route))
    if not cand:
        raise SystemExit("no routable zone pairs -- is the net one connected component?")
    cand.sort(key=lambda c: -c[2])
    return cand[:n]


def fetch_neshan(key, olat, olon, dlat, dlon):
    r = requests.get(NESHAN, headers={"Api-Key": key},
                     params={"type": "car", "origin": f"{olat:.6f},{olon:.6f}",
                             "destination": f"{dlat:.6f},{dlon:.6f}"}, timeout=30)
    js = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code != 200 or js.get("status") == "ERROR":
        raise RuntimeError(f"HTTP {r.status_code}: {js.get('message', r.text[:120])}")
    routes = js.get("routes") or []
    if not routes:
        return None
    legs = routes[0]["legs"][0]
    return legs["duration"]["value"], legs["distance"]["value"]


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("need a Neshan key: --api-key or $NESHAN_API_KEY\n"
                         "  free key: https://platform.neshan.org")

    for f_ in (args.net, args.taz):
        if not os.path.exists(f_):
            raise SystemExit(f"no such file: {f_}")

    net = sumolib.net.readNet(args.net, withInternal=False)
    reps = zone_reps(net, args.taz)
    if not reps:
        raise SystemExit(
            f"none of the zones in {os.path.basename(args.taz)} have a routable "
            f"edge on {os.path.basename(args.net)} -- that TAZ was built for a "
            f"different net")
    pairs = probe_pairs(net, reps, args.pairs)
    print(f"{len(reps)} zones, probing {len(pairs)} pairs "
          f"(freeflow {min(p[2] for p in pairs) / 60:.1f}-"
          f"{max(p[2] for p in pairs) / 60:.1f} min)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp", "hour", "from_zone", "to_zone",
                    "freeflow_s", "observed_s", "ratio", "neshan_dist_m"])

    mid = lambda e: e.getShape()[len(e.getShape()) // 2]  # noqa: E731
    rounds = 1 if args.once else max(1, int(args.hours * 3600 / args.poll))
    for r in range(rounds):
        now = datetime.now(timezone.utc).astimezone()
        ok, ratios = 0, []
        for a, b, ff, _route in pairs:
            olon, olat = net.convertXY2LonLat(*mid(reps[a]))
            dlon, dlat = net.convertXY2LonLat(*mid(reps[b]))
            time.sleep(0.3)
            try:
                got = fetch_neshan(args.api_key, olat, olon, dlat, dlon)
            except Exception as e:  # noqa: BLE001 - report, keep polling
                print(f"  {a}->{b}: {e}")
                if ok == 0 and r == 0 and pairs.index((a, b, ff, _route)) >= 4:
                    raise SystemExit(
                        "\nfirst 5 calls all failed. 480 = bad/absent key; "
                        "4xx = quota. Check the key at platform.neshan.org")
                continue
            if not got:
                continue
            obs, dist = got
            ratio = ff / obs if obs else 0.0
            w.writerow([now.isoformat(timespec="seconds"), now.hour, a, b,
                        f"{ff:.0f}", obs, f"{ratio:.3f}", dist])
            ratios.append(ratio)
            ok += 1
        f.flush()
        if not ok:
            raise SystemExit(f"round {r + 1}: every probe failed (see above)")
        print(f"[{now:%Y-%m-%d %H:%M}] round {r + 1}/{rounds}: {ok} pairs, "
              f"congestion index {np.mean(ratios):.3f} "
              f"(traffic moves at {np.mean(ratios) * 100:.0f}% of free flow)")
        if r + 1 < rounds:
            time.sleep(args.poll)
    f.close()
    print(f"\n{args.out} -- MEASURED Tehran travel times, not synthetic.")


if __name__ == "__main__":
    main()
