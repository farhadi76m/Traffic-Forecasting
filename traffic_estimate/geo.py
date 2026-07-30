"""Planar geometry on numpy arrays. shapely/geopandas are not in the env."""
from __future__ import annotations

from functools import reduce
from typing import Iterable, Sequence

import numpy as np

Ring = Sequence[tuple[float, float]]
Box = tuple[float, float, float, float]


def parse_shape(text: str) -> np.ndarray:
    """SUMO shape attribute ("x,y x,y ...") -> (n, 2) array."""
    return np.array([[float(v) for v in p.split(",")] for p in text.split()])


def format_shape(points: Iterable[Sequence[float]], precision: int = 2) -> str:
    return " ".join(f"{x:.{precision}f},{y:.{precision}f}" for x, y in points)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray cast, vectorised over `points`. Rings need not be closed."""
    pts = np.atleast_2d(np.asarray(points, float))
    x, y = pts[:, 0], pts[:, 1]
    poly = np.asarray(polygon, float)
    inside = np.zeros(len(pts), bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if y1 == y2:
            continue
        crosses = (y1 > y) != (y2 > y)
        cut = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < cut)
    return inside


def points_in_rings(points: np.ndarray, rings: Sequence[Ring]) -> np.ndarray:
    """Even-odd across every ring, so inner rings punch holes."""
    return reduce(np.logical_xor, (points_in_polygon(points, r) for r in rings))


def point_in_rings(x: float, y: float, rings: Sequence[Ring]) -> bool:
    return bool(points_in_rings(np.array([[x, y]]), rings)[0])


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area in the units of `points`, squared."""
    p = np.asarray(points, float)
    x, y = p[:, 0], p[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def chain_rings(segments: Sequence[Ring]) -> tuple[list[list], int]:
    """Chain way geometries end-to-end into closed rings.

    Returns the rings and how many had to be force-closed across a gap, which is
    the signal that the source extract is clipped.
    """
    rings, used, forced = [], [False] * len(segments), 0
    for i, seg in enumerate(segments):
        if used[i]:
            continue
        used[i] = True
        chain = list(seg)
        extended = True
        while extended and chain[0] != chain[-1]:
            extended = False
            for j, other in enumerate(segments):
                if used[j]:
                    continue
                if other[0] == chain[-1]:
                    chain += other[1:]
                elif other[-1] == chain[-1]:
                    chain += other[::-1][1:]
                elif other[-1] == chain[0]:
                    chain = list(other[:-1]) + chain
                elif other[0] == chain[0]:
                    chain = list(other[::-1][:-1]) + chain
                else:
                    continue
                used[j] = True
                extended = True
        if len(chain) < 4:
            continue
        if chain[0] != chain[-1]:
            forced += 1
            chain.append(chain[0])
        rings.append(chain)
    return rings, forced


def clip_to_box(ring: Ring, box: Box) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip against (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = box
    planes = [
        (lambda p: p[0] >= xmin,
         lambda a, b: (xmin, a[1] + (b[1] - a[1]) * (xmin - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= xmax,
         lambda a, b: (xmax, a[1] + (b[1] - a[1]) * (xmax - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= ymin,
         lambda a, b: (a[0] + (b[0] - a[0]) * (ymin - a[1]) / (b[1] - a[1]), ymin)),
        (lambda p: p[1] <= ymax,
         lambda a, b: (a[0] + (b[0] - a[0]) * (ymax - a[1]) / (b[1] - a[1]), ymax)),
    ]
    poly = list(ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring)
    for inside, intersect in planes:
        if not poly:
            return []
        out = []
        for k, cur in enumerate(poly):
            prev = poly[k - 1]
            cur_in, prev_in = inside(cur), inside(prev)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
        poly = out
    return poly


def farthest_point_sample(points: np.ndarray, k: int) -> list[int]:
    """Greedy max-min spread, so samples cover the area instead of clustering."""
    pts = np.asarray(points, float)
    chosen = [int(np.argmax(pts.sum(axis=1)))]
    dist = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    while len(chosen) < min(k, len(pts)):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return chosen
