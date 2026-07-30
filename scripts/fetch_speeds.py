#!/usr/bin/env python3
"""Fetch REAL measured road speeds for the net's area from TomTom (or HERE).

TomTom Flow Segment Data returns currentSpeed and freeFlowSpeed (km/h) plus a
confidence, per point. currentSpeed is a snapshot of *now*, so calibrating a 24h
profile means polling across a real day.

Note: neither provider serves Iran -- for Tehran use fetch_neshan.py instead.

    # one snapshot, to check the key works
    python scripts/fetch_speeds.py --net sumo/prune_tab.net.xml --once

    # a real 24h observed profile (~50 calls/hour)
    python scripts/fetch_speeds.py --net sumo/prune_tab.net.xml \
        --poll 3600 --hours 24 --out output/observed_speeds.csv
"""
import time

import _path  # noqa: F401
import requests

from traffic_estimate.cli import add_network, add_polling, load_network, make_parser
from traffic_estimate.services import SPEED_PROVIDERS, ApiError, arterial_probes
from traffic_estimate.services.observations import CsvRecorder, poll_rounds

COLUMNS = ["timestamp", "hour", "edge_id", "lat", "lon", "current_kmh",
           "freeflow_kmh", "confidence"]


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    add_polling(parser, "output/observed_speeds.csv",
                "how many road segments to sample (1 API call each)", probes=50)
    parser.add_argument("--provider", choices=list(SPEED_PROVIDERS),
                        default="tomtom")
    parser.add_argument("--min-speed", type=float, default=11.0,
                        help="only probe edges faster than this m/s (~40km/h): "
                             "side streets have no useful probe data")
    args = parser.parse_args()

    provider = SPEED_PROVIDERS[args.provider](args.api_key)
    key = provider.api_key
    print(f"api key is {key[:4]}{'*' * (len(key) - 8)}{key[-4:]}")

    network = load_network(args.net)
    probes = arterial_probes(network, args.probes, args.min_speed)
    print(f"{len(probes)} probe segments across the net")

    with CsvRecorder(args.out, COLUMNS) as recorder:
        for index, total, now in poll_rounds(args.once, args.poll, args.hours):
            ok, failed = 0, 0
            for edge in probes:
                lon, lat = network.lonlat_of(edge)
                # the free tier allows ~5 req/s; throttle failures too, or a 4xx
                # storm turns into a 429 storm
                time.sleep(0.25)
                try:
                    current, free, confidence = provider.flow(lat, lon)
                except ApiError as exc:
                    failed += 1
                    print(f"  {edge.getID()} ({lat:.4f},{lon:.4f}): {exc}")
                    # if the first handful all fail the same way it is the
                    # account, not the data -- stop instead of burning 50 calls
                    if failed == 5 and ok == 0:
                        raise SystemExit(
                            f"\n5 consecutive failures, 0 successes "
                            f"(HTTP {exc.status}).\n  {provider.diagnosis}")
                    continue
                except requests.RequestException as exc:
                    failed += 1
                    print(f"  {edge.getID()}: {type(exc).__name__} (network/proxy)")
                    continue
                if current is None:
                    continue
                recorder.row(now.isoformat(timespec="seconds"), now.hour,
                             edge.getID(), f"{lat:.6f}", f"{lon:.6f}", current,
                             free, confidence)
                ok += 1
            recorder.flush()
            if ok == 0:
                raise SystemExit(f"round {index + 1}: every probe failed "
                                 "-- see errors above")
            print(f"[{now:%Y-%m-%d %H:%M}] round {index + 1}/{total}: "
                  f"{ok} segments logged, {failed} failed")

    print(f"\n{args.out} written -- this is measured data, not synthetic.")


if __name__ == "__main__":
    main()
