"""Per-zone trip productions and attractions from OSM land use."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..geo import points_in_polygon
from ..network import RoadNetwork
from ..services.overpass import OverpassClient
from ..taz import TazSet

# tag -> accepted values (None = any value). Left side produces trips, right
# side attracts them.
PRODUCTION_TAGS = [
    ("building", {"residential", "apartments", "house", "detached", "dormitory"}),
    ("landuse", {"residential"}),
]
ATTRACTION_TAGS = [
    ("shop", None),
    ("office", None),
    ("amenity", {"school", "university", "hospital", "marketplace", "bank",
                 "restaurant", "cafe"}),
    ("building", {"commercial", "retail", "office", "industrial", "school",
                  "university", "hospital"}),
    ("landuse", {"commercial", "retail", "industrial"}),
]

PRODUCTION_QUERY = """
  way["building"~"^(residential|apartments|house|detached|dormitory)$"]({bbox});
  way["landuse"="residential"]({bbox});
"""
ATTRACTION_QUERY = """
  node["shop"]({bbox});
  node["office"]({bbox});
  node["amenity"~"^(school|university|hospital|marketplace|bank|restaurant|cafe)$"]({bbox});
  way["building"~"^(commercial|retail|office|industrial|school|university|hospital)$"]({bbox});
  way["landuse"~"^(commercial|retail|industrial)$"]({bbox});
"""


def _matches(tags: dict, rules) -> bool:
    return any(k in tags and (v is None or tags[k] in v) for k, v in rules)


def read_osm_points(osm_file: str, network: RoadNetwork
                    ) -> tuple[np.ndarray, np.ndarray]:
    """POI centroids from a local .osm extract, in network xy coordinates."""
    nodes: dict[str, tuple[float, float]] = {}
    produce: list[tuple[float, float]] = []
    attract: list[tuple[float, float]] = []

    for _, element in ET.iterparse(osm_file, events=("end",)):
        if element.tag == "node":
            point = (float(element.get("lon")), float(element.get("lat")))
            nodes[element.get("id")] = point
        elif element.tag == "way":
            refs = [nodes[nd.get("ref")] for nd in element.findall("nd")
                    if nd.get("ref") in nodes]
            point = tuple(np.mean(refs, axis=0)) if refs else None
        else:
            continue
        tags = {t.get("k"): t.get("v") for t in element.findall("tag")}
        element.clear()
        if point is None or not tags:
            continue
        if _matches(tags, PRODUCTION_TAGS):
            produce.append(point)
        if _matches(tags, ATTRACTION_TAGS):
            attract.append(point)

    def to_xy(points):
        return (np.array([network.to_xy(lon, lat) for lon, lat in points])
                if points else np.empty((0, 2)))

    return to_xy(produce), to_xy(attract)


@dataclass
class LandUseWeights:
    """Homes and jobs per zone; the margins of the gravity model."""

    zones: list[str]
    productions: np.ndarray
    attractions: np.ndarray

    def for_zones(self, zone_ids: Sequence[str]) -> "LandUseWeights":
        index = {z: i for i, z in enumerate(self.zones)}
        keep = [index[z] for z in zone_ids if z in index]
        return LandUseWeights(list(zone_ids), self.productions[keep],
                              self.attractions[keep])

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, zones=np.array(self.zones, object),
                 prod=self.productions, attr=self.attractions)
        return path

    @classmethod
    def load(cls, path: str) -> "LandUseWeights":
        blob = np.load(path, allow_pickle=True)
        return cls(list(blob["zones"]), blob["prod"], blob["attr"])

    @classmethod
    def build(cls, taz: TazSet, network: Callable[[], RoadNetwork] | RoadNetwork,
              cache: str | None = None, osm_file: str | None = None,
              overpass: OverpassClient | None = None) -> "LandUseWeights":
        """Cached land-use counts per zone.

        `network` may be a callable so a city-sized net is only parsed when the
        cache misses -- calibration calls this once per bisection iteration.
        """
        if cache and os.path.exists(cache):
            cached = cls.load(cache)
            if cached.zones == taz.ids:
                print(f"land-use weights from cache {cache}")
                return cached

        net = network() if callable(network) else network
        if osm_file and os.path.exists(osm_file):
            print(f"reading land use from {osm_file}")
            produce_xy, attract_xy = read_osm_points(osm_file, net)
        else:
            client = overpass or OverpassClient()
            south, west, north, east = net.bbox_lonlat()
            bbox = f"{south},{west},{north},{east}"
            print(f"querying Overpass over {bbox} ...")
            produce_xy = np.array([net.to_xy(lon, lat) for lon, lat
                                   in client.centroids(PRODUCTION_QUERY, bbox)])
            attract_xy = np.array([net.to_xy(lon, lat) for lon, lat
                                   in client.centroids(ATTRACTION_QUERY, bbox)])

        counts = []
        for label, points in [("residential", produce_xy), ("work/retail", attract_xy)]:
            print(f"  {len(points)} {label} features")
            counts.append(np.array([
                points_in_polygon(points, z.shape).sum() if len(points) else 0
                for z in taz], dtype=float))
        productions, attractions = counts

        weights = cls(taz.ids, np.maximum(productions, 1.0),
                      np.maximum(attractions, 1.0))
        weights._warn_if_floored(productions, attractions)
        if cache:
            weights.save(cache)
        return weights

    @staticmethod
    def _warn_if_floored(productions: np.ndarray, attractions: np.ndarray) -> None:
        """A floored zone carries no signal; a mostly-floored matrix is fiction."""
        empty_p, empty_a = int((productions == 0).sum()), int((attractions == 0).sum())
        if not (empty_p or empty_a):
            return
        n = len(productions)
        print(f"  WARNING: {empty_p}/{n} zones have ZERO residential features and "
              f"{empty_a}/{n} have ZERO workplace features.\n"
              "  Those zones are floored to 1 and carry no real signal. A road-only\n"
              "  .osm extract has few buildings -- pass --osm '' to use Overpass, "
              "which has\n  far better building coverage.")
