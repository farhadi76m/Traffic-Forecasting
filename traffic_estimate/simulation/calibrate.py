"""Calibrate OD magnitude against measured congestion.

The gravity model gives the *shape* of demand; its magnitude is a guess. Because
congestion rises monotonically with demand, bisecting on total daily trips finds
the magnitude at which SUMO reproduces the congestion that was actually measured.

Both sides are compared as a ratio, never as raw km/h:

    congestion index = free-flow time / actual time, on the same routes

so the provider's free-flow baseline and SUMO's speed limits cannot disagree in
units. 0.5 means "moving at half of free flow" in both worlds.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..demand.generators import GravityGenerator
from ..edgedata import EdgeData
from ..network import RoadNetwork
from ..taz import TazSet
from .runner import SumoRunner


@dataclass
class ObservedCongestion:
    """The measured target, from fetch_neshan or fetch_speeds output."""

    target: float
    zone_pairs: set[tuple[str, str]]
    segment_ids: set[str]

    @property
    def is_route_based(self) -> bool:
        return bool(self.zone_pairs)

    @classmethod
    def from_csv(cls, path: str, peak_hours: set[int]) -> "ObservedCongestion":
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            sys.exit(f"{path} is empty")

        route_based = "from_zone" in rows[0]
        ratios, pairs, segments = [], set(), set()
        for row in rows:
            if int(row["hour"]) not in peak_hours:
                continue
            if route_based:
                ratios.append(float(row["ratio"]))
                pairs.add((row["from_zone"], row["to_zone"]))
            else:
                free = float(row["freeflow_kmh"])
                if free <= 0 or float(row["confidence"]) < 0.5:
                    continue
                ratios.append(float(row["current_kmh"]) / free)
                segments.add(row["edge_id"])
        if not ratios:
            sys.exit(f"{path} has no usable rows for peak hours "
                     f"{sorted(peak_hours)}.\nDid the fetcher run across those hours?")
        return cls(float(np.mean(ratios)), pairs, segments)


class CongestionProbe(ABC):
    """Measures the simulated congestion index the same way the observation did."""

    edge_ids: list[str]

    @property
    def index_map(self) -> dict[str, int]:
        return {edge_id: i for i, edge_id in enumerate(self.edge_ids)}

    @abstractmethod
    def index(self, data: EdgeData, peak: set[int]) -> float:
        ...

    @staticmethod
    def _mean(ratios: list[float]) -> float:
        if not ratios:
            raise RuntimeError("no probes appeared in the simulation output")
        return float(np.mean(ratios))

    def describe(self) -> str:
        return f"{len(self.edge_ids)} edges"


class RouteProbe(CongestionProbe):
    """Zone-pair routes, as measured by Neshan: free-flow time over simulated time."""

    def __init__(self, network: RoadNetwork, taz: TazSet,
                 pairs: set[tuple[str, str]]):
        reps = taz.representative_edges(network)
        self.routes: dict[tuple[str, str], tuple[float, list[str]]] = {}
        for origin, destination in pairs:
            if origin not in reps or destination not in reps:
                continue
            route, free_flow = network.shortest_route(reps[origin],
                                                      reps[destination])
            if route:
                self.routes[(origin, destination)] = (
                    free_flow, [e.getID() for e in route])
        if not self.routes:
            sys.exit("none of the observed zone pairs are routable on this net")
        self.edge_ids = sorted({e for _, edges in self.routes.values() for e in edges})
        self.free_flow = network.free_flow_time(self.edge_ids)

    def describe(self) -> str:
        return f"{len(self.routes)} zone pairs (Neshan, measured)"

    def index(self, data: EdgeData, peak: set[int]) -> float:
        rows = self.index_map
        # an edge missing from edgedata carried no traffic, so it ran free
        travel = data.traveltime_filled(self.free_flow)
        ratios = []
        for free_flow, edges in self.routes.values():
            index = [rows[e] for e in edges]
            for hour in peak:
                simulated = float(travel[index, hour].sum())
                if simulated > 0:
                    ratios.append(min(free_flow / simulated, 1.0))
        return self._mean(ratios)


class SegmentProbe(CongestionProbe):
    """Individual links, as measured by TomTom: simulated speed over speed limit."""

    def __init__(self, network: RoadNetwork, segment_ids: set[str]):
        self.edge_ids = sorted(e for e in segment_ids if network.has_edge(e))
        self.limits = np.array([network.edge(e).getSpeed() for e in self.edge_ids])

    def describe(self) -> str:
        return f"{len(self.edge_ids)} road segments (TomTom)"

    def index(self, data: EdgeData, peak: set[int]) -> float:
        ratios = []
        for hour in sorted(peak):
            speed = data.speed[:, hour]
            usable = ~np.isnan(speed) & (self.limits > 0)
            ratios += np.minimum(speed[usable] / self.limits[usable], 1.0).tolist()
        return self._mean(ratios)


def build_probe(network: RoadNetwork, taz: TazSet,
                observed: ObservedCongestion) -> CongestionProbe:
    return (RouteProbe(network, taz, observed.zone_pairs) if observed.is_route_based
            else SegmentProbe(network, observed.segment_ids))


class DemandCalibrator:
    """Bisects daily trips until the simulated congestion index matches.

    Only the peak window is simulated -- it is the only part measured, and a
    full 24 h run per iteration is far too slow. One warm-up hour ahead of the
    peak lets congestion build before scoring starts.
    """

    def __init__(self, generator: GravityGenerator, runner: SumoRunner,
                 probe: CongestionProbe, observed: ObservedCongestion,
                 peak: set[int], work_dir: str, keep_work: bool = False):
        self.generator = generator
        self.generator.noise = 0.0
        self.runner = runner
        self.probe = probe
        self.observed = observed
        self.peak = set(peak)
        self.work_dir = work_dir
        self.keep_work = keep_work

    @property
    def window(self) -> tuple[int, int]:
        return max(min(self.peak) - 1, 0), max(self.peak)

    def simulated_index(self, trips_per_day: float, tag: str) -> float:
        first, last = self.window
        directory = os.path.join(self.work_dir, tag)
        os.makedirs(directory, exist_ok=True)
        try:
            # emit only the simulated hours: a 24 h matrix at 343 zones is
            # millions of rows od2trips would immediately discard
            self.generator.daily_trips = trips_per_day
            self.generator.hours = np.arange(first, last + 1)
            od_file = self.generator.matrix().write(
                os.path.join(directory, "od.xml"))
            edgedata = self.runner.run(od_file, directory,
                                       begin=first * 3600, end=(last + 1) * 3600)
            return self.probe.index(
                EdgeData.read(edgedata, self.probe.index_map), self.peak)
        finally:
            # the index is all we keep; the routes behind it are worth GBs
            if not self.keep_work:
                shutil.rmtree(directory, ignore_errors=True)

    def calibrate(self, lo: int, hi: int, iterations: int = 8,
                  tol: float = 0.02) -> tuple[int, float]:
        target = self.observed.target
        best = None
        for step in range(iterations):
            # geometric: plausible demand spans orders of magnitude
            trips = int(np.sqrt(lo * hi))
            index = self.simulated_index(trips, f"it{step}")
            error = index - target
            print(f"  iter {step}: {trips:>8,} trips/day -> index {index:.3f} "
                  f"({'too free' if error > 0 else 'too jammed'}, err {error:+.3f})")
            if best is None or abs(error) < abs(best[1] - target):
                best = (trips, index)
            if abs(error) < tol:
                print("  converged")
                break
            if error > 0:
                lo = trips      # flows too freely -> needs more demand
            else:
                hi = trips
        return best
