"""Origin-destination demand: an array plus its SUMO tazRelation serialisation."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .xmlio import XML_HEADER, ensure_parent, escape, parse_root

SECONDS_PER_HOUR = 3600


class ODMatrix:
    """Trip counts shaped (hours, zones, zones), indexed by `zones` and `hours`."""

    def __init__(self, zones: Sequence[str], counts: np.ndarray,
                 hours: Sequence[int] | None = None):
        self.zones = list(zones)
        self.counts = np.asarray(counts)
        self.hours = np.arange(self.counts.shape[0]) if hours is None \
            else np.asarray(hours, dtype=int)
        if self.counts.shape != (len(self.hours), len(self.zones), len(self.zones)):
            raise ValueError(f"counts {self.counts.shape} does not match "
                             f"{len(self.hours)} hours x {len(self.zones)} zones")

    def __repr__(self) -> str:
        return (f"ODMatrix({len(self.zones)} zones, hours "
                f"{self.hours.min()}-{self.hours.max()}, {self.total:,} trips)")

    @property
    def n_zones(self) -> int:
        return len(self.zones)

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    @property
    def flat(self) -> np.ndarray:
        """(hours, zones^2), the layout the surrogate consumes."""
        return self.counts.reshape(len(self.hours), -1)

    def flows(self, hour: int, include_diagonal: bool = False) -> dict:
        """{(origin, destination): trips} for one hour."""
        where = np.flatnonzero(self.hours == hour)
        if not len(where):
            return {}
        block = self.counts[where[0]]
        rows, cols = np.nonzero(block)
        return {(self.zones[i], self.zones[j]): int(block[i, j])
                for i, j in zip(rows, cols) if include_diagonal or i != j}

    def write(self, path: str) -> str:
        """Stream the XML out: a city-scale day is millions of relations."""
        ensure_parent(path)
        ids = [escape(z, {'"': "&quot;"}) for z in self.zones]
        with open(path, "w", encoding="utf-8") as out:
            out.write(XML_HEADER)
            out.write("<data>\n")
            for k, hour in enumerate(self.hours):
                begin, end = hour * SECONDS_PER_HOUR, (hour + 1) * SECONDS_PER_HOUR
                out.write(f'    <interval id="h{hour}" begin="{begin}" end="{end}">\n')
                rows, cols = np.nonzero(self.counts[k])
                block = self.counts[k]
                for i, j in zip(rows, cols):
                    out.write(f'        <tazRelation from="{ids[i]}" to="{ids[j]}"'
                              f' count="{int(block[i, j])}"/>\n')
                out.write("    </interval>\n")
            out.write("</data>\n")
        return path

    @classmethod
    def from_file(cls, path: str, zones: Sequence[str] | None = None) -> "ODMatrix":
        index = {z: i for i, z in enumerate(zones)} if zones else {}
        order = list(zones) if zones else []
        records = []
        for interval in parse_root(path).iter("interval"):
            hour = int(float(interval.get("begin")) // SECONDS_PER_HOUR)
            for rel in interval.iter("tazRelation"):
                origin, dest = rel.get("from"), rel.get("to")
                if zones is None:
                    for z in (origin, dest):
                        if z not in index:
                            index[z] = len(order)
                            order.append(z)
                elif origin not in index or dest not in index:
                    continue
                records.append((hour, index[origin], index[dest],
                                int(float(rel.get("count")))))
        hours = sorted({r[0] for r in records}) or [0]
        slot = {h: k for k, h in enumerate(hours)}
        counts = np.zeros((len(hours), len(order), len(order)), dtype=np.int64)
        for hour, i, j, value in records:
            counts[slot[hour], i, j] += value
        return cls(order, counts, hours)
