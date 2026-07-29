"""Traffic analysis zones: the partition every demand model is expressed over."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from .geo import format_shape, parse_shape, points_in_polygon, polygon_area
from .network import RoadNetwork
from .xmlio import XML_HEADER, attrs, ensure_parent, parse_root


@dataclass
class Zone:
    id: str
    shape: np.ndarray
    edges: list[str] = field(default_factory=list)
    name: str = ""
    color: tuple[int, int, int] | None = None

    def __len__(self) -> int:
        return len(self.edges)

    @property
    def centroid(self) -> np.ndarray:
        return self.shape.mean(axis=0)

    @property
    def area_km2(self) -> float:
        return polygon_area(self.shape) / 1e6


class TazSet(Sequence[Zone]):
    """An ordered set of zones, loadable from or writable to a SUMO TAZ file."""

    def __init__(self, zones: Sequence[Zone]):
        self.zones = list(zones)

    def __len__(self) -> int:
        return len(self.zones)

    def __getitem__(self, index):
        return self.zones[index]

    def __iter__(self) -> Iterator[Zone]:
        return iter(self.zones)

    def __repr__(self) -> str:
        return f"TazSet({len(self)} zones, {sum(len(z) for z in self)} edges)"

    @property
    def ids(self) -> list[str]:
        return [z.id for z in self.zones]

    @property
    def sizes(self) -> np.ndarray:
        return np.array([len(z) for z in self.zones], dtype=float)

    @property
    def centroids(self) -> np.ndarray:
        return np.array([z.centroid for z in self.zones])

    def select(self, ids: Sequence[str]) -> "TazSet":
        keep = set(ids)
        return TazSet([z for z in self.zones if z.id in keep])

    # --- construction ------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "TazSet":
        zones = []
        for taz in parse_root(path).iter("taz"):
            zones.append(Zone(id=taz.get("id"),
                              shape=parse_shape(taz.get("shape", "")),
                              edges=(taz.get("edges") or "").split(),
                              name=taz.get("name", "")))
        return cls(zones)

    @classmethod
    def grid(cls, network: RoadNetwork, n: int, min_edges: int = 5) -> "TazSet":
        """An n x n grid of zones over the network bounding box."""
        (xmin, ymin), (xmax, ymax) = network.net.getBBoxXY()
        width, height = (xmax - xmin) / n, (ymax - ymin) / n
        shapes = {}
        for i in range(n):
            for j in range(n):
                x0, y0 = xmin + i * width, ymin + j * height
                shapes[f"{i}_{j}"] = np.array([
                    (x0, y0), (x0 + width, y0),
                    (x0 + width, y0 + height), (x0, y0 + height)])
        return cls.from_shapes(network, shapes, min_edges)

    @classmethod
    def remap(cls, network: RoadNetwork, taz_file: str,
              source_net: RoadNetwork | None = None,
              min_edges: int = 5) -> "TazSet":
        """Keep another TAZ file's zone shapes, re-assign edges from `network`.

        A TAZ built on a different net has a different coordinate offset, so its
        shapes are carried across through geo-coordinates.
        """
        shapes = {}
        for zone in cls.from_file(taz_file):
            shape = zone.shape
            if source_net is not None:
                shape = np.array([network.to_xy(*source_net.to_lonlat(x, y))
                                  for x, y in shape])
            shapes[zone.id] = shape
        return cls.from_shapes(network, shapes, min_edges)

    @classmethod
    def from_shapes(cls, network: RoadNetwork, shapes: dict[str, np.ndarray],
                    min_edges: int = 5) -> "TazSet":
        """Assign every routable edge to the zone containing its centre."""
        edges = network.routable
        ids = np.array([e.getID() for e in edges])
        centers = network.centers(edges)
        zones = []
        for zone_id, shape in shapes.items():
            inside = points_in_polygon(centers, shape)
            if inside.sum() < min_edges:
                print(f"zone {zone_id}: only {int(inside.sum())} edges, dropped")
                continue
            zones.append(Zone(id=zone_id, shape=shape,
                              edges=sorted(ids[inside].tolist())))
        return cls(zones)

    # --- use ---------------------------------------------------------------

    def representative_edges(self, network: RoadNetwork) -> dict[str, object]:
        """One routing anchor per zone: the edge whose midpoint is most central.

        The polygon centroid itself can land on a building or off the network.
        """
        reps = {}
        for zone in self.zones:
            candidates = [network.edge(e) for e in zone.edges
                          if network.has_edge(e)
                          and network.edge(e).allows(network.vclass)]
            if not candidates:
                continue
            target = zone.centroid
            reps[zone.id] = min(candidates, key=lambda e: float(
                np.sum((np.array(network.midpoint(e)) - target) ** 2)))
        return reps

    def write(self, path: str) -> str:
        ensure_parent(path)
        with open(path, "w", encoding="utf-8") as out:
            out.write(XML_HEADER)
            out.write('<tazs xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
                      ' xsi:noNamespaceSchemaLocation='
                      '"http://sumo.dlr.de/xsd/taz_file.xsd">\n')
            for zone in self.zones:
                colour = ",".join(map(str, zone.color)) if zone.color else None
                out.write("    <taz " + attrs(
                    id=zone.id, name=zone.name or None,
                    shape=format_shape(zone.shape), color=colour,
                    edges=" ".join(zone.edges)) + "/>\n")
            out.write("</tazs>\n")
        return path
