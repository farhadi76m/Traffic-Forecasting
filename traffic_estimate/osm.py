"""OSM sources: Overpass boundary relations and OSM-XML -> GeoJSON."""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

from .geo import chain_rings

# closed ways carrying any of these keys are areas, not lines
AREA_KEYS = {"building", "building:part", "landuse", "natural", "leisure",
             "amenity", "shop", "historic", "place", "boundary", "waterway",
             "man_made", "public_transport", "aeroway"}
RELATION_TYPES = ("multipolygon", "boundary")


@dataclass
class BoundaryZone:
    """One administrative area, as closed lon/lat rings."""

    id: int
    name: str
    name_fa: str
    level: str
    rings: list
    edges: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    color: tuple[int, int, int] | None = None
    taz_id: str = ""

    @property
    def centroid(self) -> tuple[float, float]:
        points = [p for ring in self.rings for p in ring]
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))


def boundary_zones(overpass_json: dict, levels: set[str] | None) -> list[BoundaryZone]:
    """Relations at the given admin levels, with their ways chained into rings.

    levels=None accepts every level, which is how a new map is probed.
    """
    zones = []
    for element in overpass_json.get("elements", []):
        tags = element.get("tags", {})
        if levels is not None and tags.get("admin_level") not in levels:
            continue
        segments = [[(p["lon"], p["lat"]) for p in member["geometry"]]
                    for member in element.get("members", [])
                    if member.get("type") == "way"
                    and member.get("role") in ("outer", "")
                    and member.get("geometry")]
        rings, _ = chain_rings(segments)
        if not rings:
            continue
        zones.append(BoundaryZone(
            id=element["id"],
            name=tags.get("name:en") or tags.get("name") or str(element["id"]),
            name_fa=tags.get("name", ""),
            level=tags.get("admin_level"),
            rings=rings))
    return zones


def zones_to_geojson(zones: list[BoundaryZone]) -> dict:
    features = []
    for zone in zones:
        properties = {"osm_id": zone.id, "name": zone.name,
                      "name_fa": zone.name_fa, "admin_level": zone.level}
        if zone.taz_id:
            properties.update(taz_id=zone.taz_id, edge_count=len(zone.edges))
        if zone.merged:
            properties["merged_zones"] = zone.merged
        rings = [[list(p) for p in ring] for ring in zone.rings]
        geometry = ({"type": "Polygon", "coordinates": rings} if len(rings) == 1
                    else {"type": "MultiPolygon",
                          "coordinates": [[r] for r in rings]})
        features.append({"type": "Feature", "properties": properties,
                         "geometry": geometry})
    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: str, payload: dict, indent: int | None = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
    return path


def parse_tag_filter(spec: str | None) -> list[tuple[str, str | None]]:
    """'highway,boundary=administrative' -> [('highway', None), ('boundary', ...)]"""
    rules = []
    for part in (spec or "").split(","):
        part = part.strip()
        if part:
            key, _, value = part.partition("=")
            rules.append((key.strip(), value.strip() or None))
    return rules


class OsmGeoJson:
    """Convert an OSM XML extract to GeoJSON. No geopandas or osmium needed.

    Extracts are often clipped: a relation survives but the ways it is built
    from do not, and its polygon then cannot be rebuilt. Those are reported
    rather than emitted as broken geometry.
    """

    def __init__(self, path: str, tag_filter=(), include_points: bool = True):
        self.path = path
        self.rules = list(tag_filter)
        self.include_points = include_points
        self.kinds: Counter = Counter()
        self.clipped: list[tuple[str, int, int]] = []
        self.incomplete_ways = 0

    def _matches(self, tags: dict) -> bool:
        if not self.rules:
            return bool(tags)
        return any(k in tags and (v is None or tags[k] == v) for k, v in self.rules)

    @staticmethod
    def _tags(element) -> dict:
        return {t.get("k"): t.get("v") for t in element.findall("tag")}

    @staticmethod
    def _is_area(tags: dict, closed: bool) -> bool:
        if not closed or tags.get("area") == "no":
            return False
        return tags.get("area") == "yes" or bool(AREA_KEYS & tags.keys())

    def convert(self) -> list[dict]:
        root = ET.parse(self.path).getroot()
        nodes, node_tags = {}, {}
        for node in root.findall("node"):
            nodes[node.get("id")] = (float(node.get("lon")), float(node.get("lat")))
            if tags := self._tags(node):
                node_tags[node.get("id")] = tags
        ways = {w.get("id"): {"refs": [nd.get("ref") for nd in w.findall("nd")],
                              "tags": self._tags(w)}
                for w in root.findall("way")}
        relations = root.findall("relation")
        print(f"  {len(nodes)} nodes, {len(ways)} ways, {len(relations)} relations")

        features = []
        if self.include_points:
            features += self._point_features(nodes, node_tags)
        features += self._way_features(nodes, ways, relations)
        features += self._relation_features(nodes, ways, relations)
        return features

    def _point_features(self, nodes, node_tags) -> list[dict]:
        features = []
        for node_id, tags in node_tags.items():
            if not self._matches(tags):
                continue
            self.kinds["Point"] += 1
            features.append(self._feature("node", node_id, tags,
                                          {"type": "Point",
                                           "coordinates": list(nodes[node_id])}))
        return features

    def _way_features(self, nodes, ways, relations) -> list[dict]:
        in_relation = {m.get("ref") for rel in relations
                       if self._tags(rel).get("type") in RELATION_TYPES
                       for m in rel.findall("member") if m.get("type") == "way"}
        features = []
        for way_id, way in ways.items():
            tags = way["tags"]
            if not self._matches(tags) or (not tags and way_id in in_relation):
                continue
            coords = [nodes[r] for r in way["refs"] if r in nodes]
            if len(coords) != len(way["refs"]):
                self.incomplete_ways += 1
            if len(coords) < 2:
                continue
            closed = len(coords) > 3 and way["refs"][0] == way["refs"][-1]
            kind = "Polygon" if self._is_area(tags, closed) else "LineString"
            shape = [list(c) for c in coords]
            self.kinds[kind] += 1
            features.append(self._feature(
                "way", way_id, tags,
                {"type": kind, "coordinates": [shape] if kind == "Polygon" else shape}))
        return features

    def _relation_features(self, nodes, ways, relations) -> list[dict]:
        features = []
        for relation in relations:
            tags = self._tags(relation)
            if tags.get("type") not in RELATION_TYPES or not self._matches(tags):
                continue
            members = [m for m in relation.findall("member") if m.get("type") == "way"]
            segments = []
            for member in members:
                if member.get("role") not in ("outer", ""):
                    continue
                way = ways.get(member.get("ref"))
                points = [nodes[r] for r in way["refs"] if r in nodes] if way else []
                if len(points) >= 2:
                    segments.append(points)
            name = tags.get("name:en") or tags.get("name") or relation.get("id")
            rings, forced = chain_rings(segments) if segments else ([], 0)
            if not rings:
                self.clipped.append((name, len(segments), len(members)))
                continue
            if len(segments) < len(members) or forced:
                self.clipped.append((name, len(segments), len(members)))
            coords = [[list(p) for p in r] for r in rings]
            geometry = ({"type": "Polygon", "coordinates": coords} if len(coords) == 1
                        else {"type": "MultiPolygon",
                              "coordinates": [[c] for c in coords]})
            self.kinds[geometry["type"]] += 1
            features.append(self._feature("relation", relation.get("id"), tags,
                                          geometry))
        return features

    @staticmethod
    def _feature(osm_type: str, osm_id: str, tags: dict, geometry: dict) -> dict:
        return {"type": "Feature", "id": f"{osm_type}/{osm_id}",
                "properties": {"osm_id": osm_id, "osm_type": osm_type, **tags},
                "geometry": geometry}

    def report(self) -> None:
        print(f"  geometry: {dict(self.kinds)}")
        if self.incomplete_ways:
            print(f"  note: {self.incomplete_ways} ways referenced nodes not in "
                  "the extract (drawn from the nodes that are present)")
        if self.clipped:
            print(f"\n  WARNING: {len(self.clipped)} relation(s) are clipped in "
                  "this extract -- their member ways are missing, so the "
                  "polygons are incomplete:")
            for name, have, total in self.clipped[:10]:
                print(f"    {name}: only {have}/{total} member ways present")
            if len(self.clipped) > 10:
                print(f"    ... and {len(self.clipped) - 10} more")
            print("  For district boundaries, fetch them from Overpass instead "
                  "(build_taz_districts.py does this).")
