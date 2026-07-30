"""SUMO network access. The only module that talks to sumolib.net."""
from __future__ import annotations

import os
from functools import cached_property
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import sumolib

STATIC_FEATURES = ("length", "speed_limit", "lanes", "priority")


class RoadNetwork:
    """A SUMO net plus the queries the pipeline actually makes of it.

    Reading a city-sized net costs minutes and gigabytes, so callers should pass
    one instance around rather than re-reading the file.
    """

    def __init__(self, path: str, *, vclass: str = "passenger",
                 with_internal: bool = False):
        self.path = path
        self.vclass = vclass
        self.net = sumolib.net.readNet(path, withInternal=with_internal)

    def __repr__(self) -> str:
        return (f"RoadNetwork({os.path.basename(self.path)}, "
                f"{len(self.drivable)} {self.vclass} edges)")

    @property
    def name(self) -> str:
        return os.path.basename(self.path).split(".net.xml")[0].split(".xml")[0]

    # --- edge sets ---------------------------------------------------------

    @cached_property
    def drivable(self) -> list:
        """Every non-internal edge the vclass is allowed on."""
        return [e for e in self.net.getEdges()
                if not e.isSpecial() and e.allows(self.vclass)]

    @cached_property
    def largest_scc(self) -> set[str]:
        """Edge ids of the largest strongly-connected component of the road graph.

        Edge-level, not junction-level: an edge is only usable if a route can
        both reach and leave it, which is what od2trips/duarouter need.
        """
        ids = {e.getID() for e in self.drivable}
        graph = nx.DiGraph()
        graph.add_nodes_from(ids)
        graph.add_edges_from((e.getID(), nxt.getID()) for e in self.drivable
                             for nxt in e.getOutgoing() if nxt.getID() in ids)
        return max(nx.strongly_connected_components(graph), key=len)

    @cached_property
    def routable(self) -> list:
        """Drivable edges restricted to the largest SCC."""
        keep = self.largest_scc
        return [e for e in self.drivable if e.getID() in keep]

    def edges(self, connected_only: bool = True) -> list:
        return self.routable if connected_only else self.drivable

    def edge(self, edge_id: str):
        return self.net.getEdge(edge_id)

    def has_edge(self, edge_id: str) -> bool:
        return self.net.hasEdge(edge_id)

    def sorted_ids(self, connected_only: bool = False) -> list[str]:
        return sorted(e.getID() for e in self.edges(connected_only))

    # --- geometry ----------------------------------------------------------

    @staticmethod
    def midpoint(edge) -> tuple[float, float]:
        """The point half way along the edge, following its geometry."""
        return tuple(sumolib.geomhelper.positionAtShapeOffset(
            edge.getShape(), edge.getLength() / 2.0))

    @staticmethod
    def center(edge) -> tuple[float, float]:
        return tuple(np.mean(edge.getShape(), axis=0))

    def midpoints(self, edges: Sequence) -> np.ndarray:
        return np.array([self.midpoint(e) for e in edges])

    def centers(self, edges: Sequence) -> np.ndarray:
        return np.array([self.center(e) for e in edges])

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        return self.net.convertXY2LonLat(x, y)

    def to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        return self.net.convertLonLat2XY(lon, lat)

    def lonlat_of(self, edge) -> tuple[float, float]:
        return self.to_lonlat(*self.midpoint(edge))

    @property
    def boundary(self) -> tuple[float, float, float, float]:
        return self.net.getBoundary()

    def bbox_lonlat(self, pad: float = 0.0) -> tuple[float, float, float, float]:
        """(south, west, north, east), padded in degrees."""
        x0, y0, x1, y1 = self.boundary
        west, south = self.to_lonlat(x0, y0)
        east, north = self.to_lonlat(x1, y1)
        return south - pad, west - pad, north + pad, east + pad

    # --- attributes --------------------------------------------------------

    def static_features(self, edge_ids: Iterable[str]) -> np.ndarray:
        """(n, 4) length, speed limit, lane count, priority."""
        edges = [self.edge(e) for e in edge_ids]
        return np.array([[e.getLength(), e.getSpeed(),
                          e.getLaneNumber(), e.getPriority()] for e in edges],
                        dtype=np.float32)

    def free_flow_time(self, edge_ids: Iterable[str]) -> np.ndarray:
        """Seconds to traverse each edge at its speed limit."""
        edges = [self.edge(e) for e in edge_ids]
        return np.array([self.travel_time(e) for e in edges], dtype=np.float32)

    @staticmethod
    def travel_time(edge) -> float:
        return edge.getLength() / max(edge.getSpeed(), 0.1)

    def shortest_route(self, src, dst) -> tuple[list, float]:
        """Fastest path and its free-flow seconds. sumolib defaults to metres."""
        route, cost = self.net.getOptimalPath(src, dst, fastest=True,
                                              vClass=self.vclass)
        return list(route or []), cost
