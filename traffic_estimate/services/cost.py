"""Zone-to-zone travel-time (impedance) matrices.

This is the *cost* side of the gravity model, never demand: no routing API
sells trip counts.
"""
from __future__ import annotations

import heapq
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import requests

from ..network import RoadNetwork
from ..xmlio import ensure_parent

ARCGIS_URL = ("https://route-api.arcgis.com/arcgis/rest/services/World/"
              "OriginDestinationCostMatrix/NAServer/"
              "OriginDestinationCostMatrix_World/solveODCostMatrix")
OSRM_URL = "https://router.project-osrm.org/table/v1/driving/"
UNREACHABLE_FACTOR = 3.0


class CostBackend(ABC):
    """Fills an n x n matrix of seconds for the given zone anchors."""

    @abstractmethod
    def compute(self, network: RoadNetwork, anchors: list) -> np.ndarray:
        ...


class SumoCost(CostBackend):
    """One Dijkstra per origin over our own net.

    sumolib's getOptimalPath is a single-pair search, so an n x n matrix costs
    n^2 full searches -- fine for 6 zones, hopeless for 300 over 218k edges.
    One relaxation per origin reads off every destination on the way.
    """

    def __init__(self, progress_every: int = 25):
        self.progress_every = progress_every

    def compute(self, network: RoadNetwork, anchors: list) -> np.ndarray:
        n = len(anchors)
        matrix = np.full((n, n), np.nan)
        index = {edge.getID(): j for j, edge in enumerate(anchors)}
        cost_of = network.travel_time

        for i, source in enumerate(anchors):
            # (cost, edge id) only: Edge objects are not orderable and must
            # never end up as a heap tie-breaker
            queue = [(cost_of(source), source.getID())]
            settled: set[str] = set()
            remaining = n
            while queue and remaining:
                distance, edge_id = heapq.heappop(queue)
                if edge_id in settled:
                    continue
                settled.add(edge_id)
                j = index.get(edge_id)
                if j is not None:
                    matrix[i, j] = distance
                    remaining -= 1
                for nxt in network.edge(edge_id).getOutgoing():
                    if nxt.getID() not in settled and nxt.allows(network.vclass):
                        heapq.heappush(queue, (distance + cost_of(nxt), nxt.getID()))
            matrix[i, i] = cost_of(source)
            if (i + 1) % self.progress_every == 0 or i + 1 == n:
                print(f"  {i + 1}/{n} origins", flush=True)
        return matrix


class OsrmCost(CostBackend):
    """Public OSRM demo server: free, rate limited, real OSM road speeds."""

    def compute(self, network: RoadNetwork, anchors: list) -> np.ndarray:
        lonlats = [network.lonlat_of(edge) for edge in anchors]
        coords = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in lonlats)
        response = requests.get(OSRM_URL + coords,
                                params={"annotations": "duration"}, timeout=120)
        response.raise_for_status()
        return np.array(response.json()["durations"], dtype=float)


class ArcGisCost(CostBackend):
    """ArcGIS travelCostMatrix; charged per pair, so origins are chunked."""

    def __init__(self, api_key: str | None = None, depart: str | None = None,
                 chunk: int = 100):
        self.api_key = api_key or os.environ.get("ARCGIS_API_KEY")
        self.depart = depart
        self.chunk = chunk

    def compute(self, network: RoadNetwork, anchors: list) -> np.ndarray:
        if not self.api_key:
            raise SystemExit("ArcGIS backend needs --api-key or $ARCGIS_API_KEY "
                             "(free key: developers.arcgis.com)")
        lonlats = [network.lonlat_of(edge) for edge in anchors]
        n = len(lonlats)
        matrix = np.full((n, n), np.nan)
        per_call = max(1, self.chunk // max(1, n // 100 + 1))
        for start in range(0, n, per_call):
            origins = lonlats[start:start + per_call]
            params = {
                "f": "json", "token": self.api_key,
                "origins": self._featureset(origins, start),
                "destinations": self._featureset(lonlats),
                "travelMode": "",
                "impedanceAttributeName": "TravelTime",
                "outputType": "esriNAODOutputSparseMatrix",
                "returnOrigins": "false", "returnDestinations": "false",
            }
            if self.depart:
                params.update(timeOfDay=self.depart, timeOfDayUsage="start")
            response = requests.post(ARCGIS_URL, data=params, timeout=180)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise SystemExit(f"ArcGIS error: {payload['error']}")
            # sparse: {"<origin>": {"<dest>": [cost, ...]}}, 1-based, minutes
            for origin, dests in payload["odCostMatrix"].items():
                if not origin.isdigit():
                    continue
                for dest, values in dests.items():
                    matrix[start + int(origin) - 1, int(dest) - 1] = values[0] * 60.0
            print(f"  origins {start}-{start + len(origins) - 1} done")
            time.sleep(0.2)
        return matrix

    @staticmethod
    def _featureset(points, offset: int = 0) -> str:
        features = [{"geometry": {"x": lon, "y": lat},
                     "attributes": {"Name": str(offset + k),
                                    "ObjectID": offset + k + 1}}
                    for k, (lon, lat) in enumerate(points)]
        return str({"spatialReference": {"wkid": 4326},
                    "features": features}).replace("'", '"')


BACKENDS = {"sumo": SumoCost, "osrm": OsrmCost, "arcgis": ArcGisCost}


class CostMatrix:
    """Seconds between zone anchors, plus the zone order they are indexed by."""

    def __init__(self, zones: list[str], seconds: np.ndarray):
        self.zones = list(zones)
        self.seconds = np.asarray(seconds, float)

    def __repr__(self) -> str:
        return (f"CostMatrix({len(self.zones)} zones, mean "
                f"{self.seconds.mean() / 60:.1f} min)")

    @property
    def minutes(self) -> np.ndarray:
        return self.seconds / 60.0

    @classmethod
    def build(cls, network: RoadNetwork, taz, backend: CostBackend
              ) -> "CostMatrix":
        reps = taz.representative_edges(network)
        zones = [z.id for z in taz if z.id in reps]
        matrix = backend.compute(network, [reps[z] for z in zones])
        return cls(zones, matrix)

    def fill_unreachable(self) -> int:
        """Island zones never route. Make them very far away, not missing."""
        missing = int(np.isnan(self.seconds).sum())
        if missing:
            self.seconds[np.isnan(self.seconds)] = (
                np.nanmax(self.seconds) * UNREACHABLE_FACTOR)
        return missing

    def reindex(self, zones: list[str]) -> "CostMatrix":
        order = [self.zones.index(z) for z in zones]
        return CostMatrix(zones, self.seconds[np.ix_(order, order)])

    def islands(self) -> set[str]:
        """Zones whose row is the sentinel fill, i.e. unreachable from the rest.

        The sentinel sits a clear factor above the largest real cost; comparing
        against the median instead would flag the slowest genuine pair.
        """
        off_diagonal = self.seconds[~np.eye(len(self.zones), dtype=bool)]
        distinct = np.unique(off_diagonal)
        if len(distinct) < 2 or distinct[-1] < 2.5 * distinct[-2]:
            return set()
        hits = (self.seconds >= distinct[-1] - 1e-6).sum(axis=1)
        return {zone for zone, n in zip(self.zones, hits) if n > 0}

    def to_csv(self, path: str) -> str:
        ensure_parent(path)
        with open(path, "w", encoding="utf-8") as out:
            out.write("," + ",".join(self.zones) + "\n")
            for zone, row in zip(self.zones, self.seconds):
                out.write(zone + "," + ",".join(f"{v:.1f}" for v in row) + "\n")
        return path

    @classmethod
    def from_csv(cls, path: str) -> "CostMatrix":
        with open(path, encoding="utf-8") as handle:
            rows = [line.strip().split(",") for line in handle if line.strip()]
        return cls(rows[0][1:],
                   np.array([[float(v) for v in r[1:]] for r in rows[1:]]))
