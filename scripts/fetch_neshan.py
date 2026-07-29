#!/usr/bin/env python3
"""Observe REAL Tehran travel times between TAZ pairs, via Neshan (neshan.org).

TomTom and HERE do not serve Iran. Neshan is Iranian, so it works from a Tehran
IP and has live Tehran traffic. Free key at platform.neshan.org.

Per probe zone-pair we record:
    freeflow_s  shortest-path time on an EMPTY net (our own speed limits)
    observed_s  Neshan's traffic-aware driving time, right now (measured)
    ratio       freeflow_s / observed_s -- 1.0 free flowing, 0.4 crawling

That ratio is the calibration target. Both sides are route travel times on the
same road network, so they are directly comparable -- no unit fudging.

    # check the key works
    python scripts/fetch_neshan.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --once

    # a real observed 24h profile (leave running for a day)
    python scripts/fetch_neshan.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --poll 3600 --hours 24 \
        --out output/observed_neshan.csv
"""
import time

import _path  # noqa: F401
import numpy as np

from traffic_estimate.cli import (add_network, add_polling, add_taz,
                                  load_network, make_parser, require_files)
from traffic_estimate.services import CsvRecorder, NeshanProvider, long_zone_pairs
from traffic_estimate.services.observations import poll_rounds
from traffic_estimate.taz import TazSet

COLUMNS = ["timestamp", "hour", "from_zone", "to_zone", "freeflow_s",
           "observed_s", "ratio", "neshan_dist_m"]


def main():
    parser = make_parser(__doc__)
    add_network(parser)
    add_taz(parser)
    add_polling(parser, "output/observed_neshan.csv",
                "zone pairs to probe per round (1 API call each)", probes=20)
    args = parser.parse_args()

    require_files(args.net, args.taz)
    provider = NeshanProvider(args.api_key)
    network = load_network(args.net)
    taz = TazSet.from_file(args.taz)
    reps = taz.representative_edges(network)
    pairs = long_zone_pairs(network, taz, args.probes)
    print(f"{len(reps)} zones, probing {len(pairs)} pairs "
          f"(freeflow {min(p[2] for p in pairs) / 60:.1f}-"
          f"{max(p[2] for p in pairs) / 60:.1f} min)")

    with CsvRecorder(args.out, COLUMNS) as recorder:
        for index, total, now in poll_rounds(args.once, args.poll, args.hours):
            ok, ratios = 0, []
            for position, (origin, destination, free_flow) in enumerate(pairs):
                time.sleep(0.3)
                try:
                    result = provider.duration(
                        network.lonlat_of(reps[origin]),
                        network.lonlat_of(reps[destination]))
                except Exception as exc:  # report, keep polling
                    print(f"  {origin}->{destination}: {exc}")
                    if ok == 0 and index == 0 and position >= 4:
                        raise SystemExit(f"\nfirst 5 calls all failed. "
                                         f"{provider.diagnosis}")
                    continue
                if not result:
                    continue
                observed, distance = result
                ratio = free_flow / observed if observed else 0.0
                recorder.row(now.isoformat(timespec="seconds"), now.hour, origin,
                             destination, f"{free_flow:.0f}", observed,
                             f"{ratio:.3f}", distance)
                ratios.append(ratio)
                ok += 1
            recorder.flush()
            if not ok:
                raise SystemExit(f"round {index + 1}: every probe failed (see above)")
            print(f"[{now:%Y-%m-%d %H:%M}] round {index + 1}/{total}: {ok} pairs, "
                  f"congestion index {np.mean(ratios):.3f} "
                  f"(traffic moves at {np.mean(ratios) * 100:.0f}% of free flow)")

    print(f"\n{args.out} -- MEASURED Tehran travel times, not synthetic.")


if __name__ == "__main__":
    main()
