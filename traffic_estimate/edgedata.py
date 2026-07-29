"""SUMO edgeData output: hourly per-edge counts, travel times and speeds."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from .xmlio import iter_elements

HOURS = 24


def find_edgedata(directory: str) -> str | None:
    """run_sim gzips its output; older runs left it plain. Accept both."""
    for name in ("edgedata.xml.gz", "edgedata.xml"):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


def scenario_dirs(sim_dir: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(sim_dir, "*"))
                  if os.path.isdir(d) and find_edgedata(d))


@dataclass
class EdgeData:
    """(n_edges, hours) arrays. NaN travel time means the edge carried no traffic."""

    counts: np.ndarray
    traveltime: np.ndarray
    speed: np.ndarray

    @classmethod
    def read(cls, path: str, edge_index: dict[str, int],
             hours: int = HOURS) -> "EdgeData":
        n = len(edge_index)
        counts = np.zeros((n, hours), np.float32)
        traveltime = np.full((n, hours), np.nan, np.float32)
        speed = np.full((n, hours), np.nan, np.float32)
        hour = -1
        for element in iter_elements(path, ("interval", "edge")):
            if element.tag == "interval":
                hour = int(float(element.get("begin")) // 3600)
                continue
            if not 0 <= hour < hours:
                continue
            row = edge_index.get(element.get("id"))
            if row is None:
                continue
            counts[row, hour] = float(element.get("entered") or 0)
            if (value := element.get("traveltime")) is not None:
                traveltime[row, hour] = float(value)
            if (value := element.get("speed")) is not None:
                speed[row, hour] = float(value)
        return cls(counts, traveltime, speed)

    def traveltime_filled(self, free_flow: np.ndarray) -> np.ndarray:
        """An unobserved edge-hour is an empty road, so it runs at free flow."""
        return np.where(np.isnan(self.traveltime), free_flow[:, None],
                        self.traveltime)

    @property
    def active_edges(self) -> int:
        return int((self.counts.sum(axis=1) > 0).sum())

    @property
    def total_entries(self) -> int:
        return int(self.counts.sum())
