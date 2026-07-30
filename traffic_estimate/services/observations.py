"""Recording measured traffic over time: an append-only CSV and a poll loop."""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from typing import Iterator, Sequence

import numpy as np

from ..geo import farthest_point_sample
from ..network import RoadNetwork
from ..taz import TazSet
from .cost import SumoCost


class CsvRecorder:
    """Appends rows to a CSV, writing the header only when the file is new."""

    def __init__(self, path: str, columns: Sequence[str]):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        is_new = not os.path.exists(path)
        self.handle = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle)
        if is_new:
            self.writer.writerow(columns)

    def __enter__(self) -> "CsvRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def row(self, *values) -> None:
        self.writer.writerow(values)

    def flush(self) -> None:
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def poll_rounds(once: bool, poll: int, hours: float
                ) -> Iterator[tuple[int, int, datetime]]:
    """Yield (round, total rounds, local time), sleeping between rounds."""
    total = 1 if once else max(1, int(hours * 3600 / poll))
    for index in range(total):
        yield index, total, datetime.now(timezone.utc).astimezone()
        if index + 1 < total:
            time.sleep(poll)


def arterial_probes(network: RoadNetwork, count: int, min_speed: float) -> list:
    """Fast edges spread across the net -- side streets have no probe data."""
    candidates = [e for e in network.drivable if e.getSpeed() >= min_speed]
    if not candidates:
        raise SystemExit(f"no {network.vclass} edges faster than {min_speed} m/s")
    midpoints = network.midpoints(candidates)
    return [candidates[k] for k in farthest_point_sample(midpoints, count)]


def long_zone_pairs(network: RoadNetwork, taz: TazSet, count: int,
                    min_seconds: float = 120.0) -> list[tuple[str, str, float]]:
    """Routable zone pairs, longest first, with their free-flow seconds.

    Short hops barely move the congestion ratio; long crosstown trips carry the
    calibration signal. Costs come from one Dijkstra per origin -- the pairwise
    form is 117k full searches at 343 zones and does not finish.
    """
    reps = taz.representative_edges(network)
    zones = sorted(reps)
    if not zones:
        raise SystemExit("no zone has a routable edge on this net -- that TAZ "
                         "was built for a different network")
    matrix = SumoCost().compute(network, [reps[z] for z in zones])
    pairs = [(a, b, float(matrix[i, j]))
             for i, a in enumerate(zones) for j, b in enumerate(zones)
             if i != j and np.isfinite(matrix[i, j]) and matrix[i, j] > min_seconds]
    if not pairs:
        raise SystemExit("no routable zone pairs -- is the net one component?")
    pairs.sort(key=lambda pair: -pair[2])
    return pairs[:count]
