#!/usr/bin/env python3
"""Fetch REAL measured road speeds for the net's area from TomTom (or HERE).

This is the only genuinely observed traffic data in the pipeline. Everything
else (trip counts, hour profile) is synthesized -- these speeds are measured
from probe vehicles, and are what we calibrate the OD magnitude against.

TomTom Flow Segment Data returns, per point: currentSpeed and freeFlowSpeed
(km/h) plus a confidence. currentSpeed is a snapshot of *now*, so to calibrate
a 24h profile you must poll across a real day:

    # one snapshot, to check the key works
    python scripts/fetch_speeds.py --net sumo/prune_tab.net.xml --once

    # a real 24h observed profile (runs for a day, ~50 calls/hour)
    python scripts/fetch_speeds.py --net sumo/prune_tab.net.xml \
        --poll 3600 --hours 24 --out output/observed_speeds.csv

Free tier is 2500 calls/day; 50 probes x 24 hours = 1200, comfortably inside.
Get a key at developer.tomtom.com (free, no card).
"""
import argparse
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import requests
import sumolib

TOMTOM = ("https://api.tomtom.com/traffic/services/4/flowSegmentData/"
          "absolute/10/json")
HERE = "https://data.traffic.hereapi.com/v7/flow"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--out", default="output/observed_speeds.csv")
    p.add_argument("--provider", choices=["tomtom", "here"], default="tomtom")
    p.add_argument("--api-key", default=os.environ.get("TOMTOM_API_KEY")
                   or os.environ.get("HERE_API_KEY"))
    p.add_argument("--probes", type=int, default=50,
                   help="how many road segments to sample (1 API call each)")
    p.add_argument("--min-speed", type=float, default=11.0,
                   help="only probe edges faster than this m/s (~40km/h), i.e. "
                        "arterials -- side streets have no useful probe data")
    p.add_argument("--once", action="store_true", help="single snapshot, then exit")
    p.add_argument("--poll", type=int, default=3600, help="seconds between rounds")
    p.add_argument("--hours", type=float, default=24, help="how long to keep polling")
    return p.parse_args()


def pick_probes(net, n, min_speed):
    """Choose n arterial edges spread across the net (one API call each).

    Greedy farthest-point selection so probes cover the district instead of
    clustering on one highway.
    """
    cand = [e for e in net.getEdges()
            if e.allows("passenger") and e.getSpeed() >= min_speed]
    if not cand:
        raise SystemExit(f"no passenger edges faster than {min_speed} m/s")
    mids = np.array([e.getShape()[len(e.getShape()) // 2] for e in cand])

    chosen = [int(np.argmax(mids[:, 0] + mids[:, 1]))]  # arbitrary corner start
    d = np.linalg.norm(mids - mids[chosen[0]], axis=1)
    while len(chosen) < min(n, len(cand)):
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(mids - mids[k], axis=1))
    return [(cand[k], mids[k]) for k in chosen]


class ApiError(Exception):
    """Carries the status and body so the caller can say what actually broke."""

    def __init__(self, status, body):
        self.status = status
        super().__init__(f"HTTP {status}: {body[:160]}")


def fetch_tomtom(lat, lon, key):
    r = requests.get(TOMTOM, params={"key": key, "point": f"{lat:.6f},{lon:.6f}",
                                     "unit": "KMPH"}, timeout=30)
    if r.status_code != 200:
        raise ApiError(r.status_code, r.text.strip())
    d = r.json()["flowSegmentData"]
    return d["currentSpeed"], d["freeFlowSpeed"], d.get("confidence", 1.0)


def fetch_here(lat, lon, key):
    r = requests.get(HERE, params={"apiKey": key, "locationReferencing": "shape",
                                   "in": f"circle:{lat:.6f},{lon:.6f};r=50"},
                     timeout=30)
    r.raise_for_status()
    res = r.json().get("results", [])
    if not res:
        return None, None, 0.0
    cf = res[0]["currentFlow"]
    return cf["speed"] * 3.6, cf["freeFlow"] * 3.6, cf.get("confidence", 1.0)


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "need an API key: --api-key or $TOMTOM_API_KEY\n"
            "free key (no card): https://developer.tomtom.com/user/register")

    print(f"api key is {args.api_key[:4]}{'*' * (len(args.api_key) - 8)}{args.api_key[-4:]}")
    net = sumolib.net.readNet(args.net, withInternal=False)
    probes = pick_probes(net, args.probes, args.min_speed)
    print(f"{len(probes)} probe segments across the net")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp", "hour", "edge_id", "lat", "lon",
                    "current_kmh", "freeflow_kmh", "confidence"])

    fetch = fetch_tomtom if args.provider == "tomtom" else fetch_here
    rounds = 1 if args.once else max(1, int(args.hours * 3600 / args.poll))
    for r in range(rounds):
        now = datetime.now(timezone.utc).astimezone()
        ok = 0
        fails = 0
        for edge, (x, y) in probes:
            lon, lat = net.convertXY2LonLat(x, y)
            time.sleep(0.25)  # free tier allows ~5 req/s; throttle failures too,
            try:              # or a 4xx storm turns into a 429 storm
                cur, free, conf = fetch(lat, lon, args.api_key)
            except ApiError as e:
                fails += 1
                print(f"  {edge.getID()} ({lat:.4f},{lon:.4f}): {e}")
                # 403 = key/entitlement, 400 = malformed or unsupported point.
                # If the first handful all fail the same way it is not the data,
                # it is the account -- stop instead of burning 50 calls.
                if fails == 5 and ok == 0:
                    raise SystemExit(
                        f"\n5 consecutive failures, 0 successes (HTTP {e.status}).\n"
                        "  403 -> key lacks the Traffic API entitlement, or is over quota\n"
                        "  400 -> point rejected (check lat/lon order in the output above)\n"
                        "  Any -> TomTom may be geo-blocking this IP; try --provider here")
                continue
            except requests.RequestException as e:
                fails += 1
                print(f"  {edge.getID()}: {type(e).__name__} (network/proxy)")
                continue
            if cur is None:
                continue
            w.writerow([now.isoformat(timespec="seconds"), now.hour, edge.getID(),
                        f"{lat:.6f}", f"{lon:.6f}", cur, free, conf])
            ok += 1
        f.flush()
        if ok == 0:
            raise SystemExit(f"round {r + 1}: every probe failed -- see errors above")
        print(f"[{now:%Y-%m-%d %H:%M}] round {r + 1}/{rounds}: "
              f"{ok} segments logged, {fails} failed")
        if r + 1 < rounds:
            time.sleep(args.poll)
    f.close()
    print(f"\n{args.out} written -- this is measured data, not synthetic.")


if __name__ == "__main__":
    main()
