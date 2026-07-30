"""TAZ zones from OSM administrative boundaries.

The local .osm extract cannot be used: its boundary relations keep the relation
but drop the member ways, so the polygons must come from Overpass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .geo import clip_to_box, format_shape, points_in_rings
from .network import RoadNetwork
from .osm import BoundaryZone, boundary_zones, write_geojson, zones_to_geojson
from .services.overpass import OverpassClient
from .taz import TazSet, Zone
from .xmlio import XML_HEADER, attrs, quoteattr

PALETTE = [
    (231, 76, 60), (46, 134, 193), (241, 196, 15), (39, 174, 96),
    (155, 89, 182), (230, 126, 34), (26, 188, 156), (52, 73, 94),
    (192, 57, 43), (41, 128, 185), (211, 84, 0), (127, 140, 141),
]

VIEW_SETTINGS = """<viewsettings>
    <scheme name="district view">
        <opengl dither="0" fps="0"/>
        <background backgroundColor="255,255,255" showGrid="0"/>
        <edges laneEdgeMode="0" scaleMode="0" laneShowBorders="1"
               showLinkDecals="1" showRails="1" hideConnectors="0"
               edgeName_show="0" streetName_show="0">
            <colorScheme name="uniform">
                <entry color="80,80,80"/>
            </colorScheme>
        </edges>
        <polys polyType_show="0">
            <polyName show="1" size="55" color="0,0,0"/>
        </polys>
    </scheme>
</viewsettings>
"""


@dataclass
class DistrictOutputs:
    districts_geojson: str
    network_geojson: str
    taz: str
    poly: str
    view: str
    sumocfg: str


class DistrictTazBuilder:
    """Partition a network into administrative zones and write the SUMO files.

    Only edges in the largest strongly-connected component are assigned, so
    od2trips and duarouter cannot emit trips into unreachable areas.
    """

    def __init__(self, network: RoadNetwork, *, level: str = "11",
                 min_edges: int = 40, pad: float = 0.006,
                 connected_only: bool = True,
                 overpass: OverpassClient | None = None):
        self.network = network
        self.level = level
        self.min_edges = min_edges
        self.pad = pad
        self.connected_only = connected_only
        self.overpass = overpass or OverpassClient()
        self.zones: list[BoundaryZone] = []

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.network.bbox_lonlat(self.pad)

    def available_levels(self, cache: str) -> dict[str, list[str]]:
        """What admin levels OSM offers over this network; levels differ per country."""
        data = self.overpass.boundaries(self.bbox, {"[0-9]+"}, cache)
        by_level: dict[str, list[str]] = {}
        for zone in boundary_zones(data, None):
            by_level.setdefault(zone.level, []).append(zone.name)
        return by_level

    def build(self, cache: str | None = None, refresh: bool = False
              ) -> list[BoundaryZone]:
        data = self.overpass.boundaries(self.bbox, {"9", self.level}, cache, refresh)
        zones = boundary_zones(data, {self.level})
        print(f"  {len(zones)} admin_level={self.level} polygons")
        if not zones:
            raise SystemExit(f"no admin_level={self.level} boundaries cover this "
                             "network; try a different --level")

        self._assign_edges(zones)
        zones = [z for z in zones if z.edges]
        self.zones = self._merge_small(zones)
        self._name_and_colour(self.zones)
        return self.zones

    # --- steps -------------------------------------------------------------

    def _assign_edges(self, zones: list[BoundaryZone]) -> None:
        """First zone containing an edge's midpoint claims it."""
        edges = self.network.edges(self.connected_only)
        total = len(self.network.drivable)
        if self.connected_only:
            print(f"  {len(edges)}/{total} '{self.network.vclass}' edges in the "
                  f"largest strongly-connected component "
                  f"({len(edges) / total * 100:.1f}%)")
        else:
            print(f"  {total} '{self.network.vclass}' edges (not SCC-filtered)")
        if not edges:
            raise SystemExit(f"no edges allow vclass '{self.network.vclass}'")

        ids = np.array([e.getID() for e in edges])
        lonlat = np.array([self.network.to_lonlat(*self.network.midpoint(e))
                           for e in edges])
        free = np.ones(len(edges), bool)
        for zone in zones:
            hit = free & points_in_rings(lonlat, zone.rings)
            zone.edges = ids[hit].tolist()
            free &= ~hit

    def _merge_small(self, zones: list[BoundaryZone]) -> list[BoundaryZone]:
        keep = [z for z in zones if len(z.edges) >= self.min_edges]
        small = [z for z in zones if len(z.edges) < self.min_edges]
        if not keep:
            raise SystemExit("no zone reaches --min-edges; lower the threshold")
        for zone in small:
            target = min(keep, key=lambda k: float(
                np.sum((np.array(k.centroid) - np.array(zone.centroid)) ** 2)))
            print(f"  merging {zone.name} ({len(zone.edges)} edges) -> {target.name}")
            target.edges += zone.edges
            target.merged.append(zone.name)
            target.rings = target.rings + zone.rings
        return keep

    @staticmethod
    def _name_and_colour(zones: list[BoundaryZone]) -> None:
        seen: dict[str, int] = {}
        for index, zone in enumerate(zones):
            zone.color = PALETTE[index % len(PALETTE)]
            taz_id = "".join(c if c.isalnum() else "_" for c in zone.name)
            while "__" in taz_id:
                taz_id = taz_id.replace("__", "_")
            taz_id = taz_id.strip("_") or f"taz{zone.id}"
            if taz_id in seen:  # two names can sanitise to the same id
                seen[taz_id] += 1
                taz_id = f"{taz_id}_{seen[taz_id]}"
            else:
                seen[taz_id] = 0
            zone.taz_id = taz_id

    # --- output ------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"\n  {len(self.zones)} final TAZs:"]
        for zone in sorted(self.zones, key=lambda z: -len(z.edges)):
            extra = f"  (+{', '.join(zone.merged)})" if zone.merged else ""
            lines.append(f"    {len(zone.edges):>5} edges  {zone.taz_id}{extra}")
        return "\n".join(lines)

    def _xy_rings(self, zone: BoundaryZone) -> list[list]:
        box = self.network.boundary
        rings = []
        for ring in zone.rings:
            clipped = clip_to_box(
                [self.network.to_xy(lon, lat) for lon, lat in ring], box)
            if len(clipped) >= 3:
                rings.append(clipped)
        return rings

    def as_taz_set(self) -> TazSet:
        zones = []
        for zone in self.zones:
            rings = self._xy_rings(zone)
            if not rings:
                continue
            biggest = max(rings, key=len)
            zones.append(Zone(id=zone.taz_id,
                              shape=np.array(biggest + [biggest[0]]),
                              edges=sorted(zone.edges),
                              name=zone.name_fa or zone.name,
                              color=zone.color))
        return TazSet(zones)

    def write(self, outdir: str, prefix: str) -> DistrictOutputs:
        os.makedirs(outdir, exist_ok=True)
        path = lambda suffix: os.path.join(outdir, f"{prefix}{suffix}")

        districts = write_geojson(path("_districts.geojson"),
                                  zones_to_geojson(self.zones), indent=1)
        network_gj = write_geojson(path("_network.geojson"),
                                   self._network_geojson())
        taz = self.as_taz_set().write(path("_districts.taz.xml"))
        poly = self._write_polygons(path("_districts.poly.xml"))
        view = path("_districts.view.xml")
        with open(view, "w", encoding="utf-8") as handle:
            handle.write(VIEW_SETTINGS)
        sumocfg = self._write_sumocfg(path("_districts.sumocfg"), outdir, poly, view)
        return DistrictOutputs(districts, network_gj, taz, poly, view, sumocfg)

    def _network_geojson(self) -> dict:
        zone_of = {edge_id: zone.taz_id for zone in self.zones
                   for edge_id in zone.edges}
        features = []
        for edge in self.network.edges(self.connected_only):
            features.append({
                "type": "Feature",
                "properties": {
                    "id": edge.getID(), "name": edge.getName(),
                    "type": edge.getType(),
                    "speed_kmh": round(edge.getSpeed() * 3.6, 1),
                    "lanes": edge.getLaneNumber(),
                    "length_m": round(edge.getLength(), 1),
                    "taz": zone_of.get(edge.getID()),
                },
                "geometry": {"type": "LineString",
                             "coordinates": [list(self.network.to_lonlat(x, y))
                                             for x, y in edge.getShape()]},
            })
        return {"type": "FeatureCollection", "features": features}

    def _write_polygons(self, path: str) -> str:
        """Translucent fills, so the network stays readable underneath."""
        with open(path, "w", encoding="utf-8") as out:
            out.write(XML_HEADER + "<additional>\n")
            for zone in self.zones:
                colour = "%d,%d,%d" % zone.color
                for k, ring in enumerate(self._xy_rings(zone)):
                    poly_id = zone.taz_id if k == 0 else f"{zone.taz_id}.{k}"
                    out.write("    <poly " + attrs(
                        id=poly_id, type="district", color=f"{colour},70",
                        fill="1", layer="-10", lineWidth="4",
                        shape=format_shape(ring + [ring[0]])) + "/>\n")
            out.write("</additional>\n")
        return path

    def _write_sumocfg(self, path: str, outdir: str, poly: str, view: str) -> str:
        # Only the poly file: sumo-gui turns a taz 'shape' into a polygon of the
        # same id, so loading the taz here too would collide on every zone id.
        net_rel = os.path.relpath(self.network.path, outdir)
        with open(path, "w", encoding="utf-8") as out:
            out.write(
                "<configuration>\n    <input>\n"
                f"        <net-file value={quoteattr(net_rel)}/>\n"
                f"        <additional-files value={quoteattr(os.path.basename(poly))}/>\n"
                "    </input>\n    <gui_only>\n"
                f"        <gui-settings-file value={quoteattr(os.path.basename(view))}/>\n"
                "    </gui_only>\n</configuration>\n")
        return path
