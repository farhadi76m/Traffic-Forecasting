"""Assemble OSM boundary relations (Overpass 'out geom' JSON) into closed rings.

The osmium extract in sumo/tehran_2026_area.osm has its boundary relations
clipped -- the member ways are not in the file -- so polygons have to come from
Overpass instead of the local .osm.
"""
import json


def rings_from_relation(rel):
    """Chain the 'outer' member ways of a relation into closed rings."""
    segs = []
    for m in rel.get("members", []):
        if m.get("type") != "way" or m.get("role") not in ("outer", ""):
            continue
        geom = m.get("geometry")
        if geom:
            segs.append([(p["lon"], p["lat"]) for p in geom])

    rings, used = [], [False] * len(segs)
    for i, seg in enumerate(segs):
        if used[i]:
            continue
        used[i] = True
        chain = list(seg)
        extended = True
        while extended and chain[0] != chain[-1]:
            extended = False
            for j, other in enumerate(segs):
                if used[j]:
                    continue
                if other[0] == chain[-1]:
                    chain += other[1:]
                elif other[-1] == chain[-1]:
                    chain += other[::-1][1:]
                elif other[-1] == chain[0]:
                    chain = other[:-1] + chain
                elif other[0] == chain[0]:
                    chain = other[::-1][:-1] + chain
                else:
                    continue
                used[j] = True
                extended = True
        if len(chain) >= 4:
            if chain[0] != chain[-1]:
                chain.append(chain[0])
            rings.append(chain)
    return rings


def load_relations(path, levels):
    """-> [{id, name, name_fa, level, rings}] for relations at the given admin levels.

    levels=None accepts every level (used to probe what a new map offers).
    """
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if levels is not None and tags.get("admin_level") not in levels:
            continue
        rings = rings_from_relation(el)
        if not rings:
            continue
        out.append({
            "id": el["id"],
            "name": tags.get("name:en") or tags.get("name") or str(el["id"]),
            "name_fa": tags.get("name", ""),
            "level": tags.get("admin_level"),
            "rings": rings,
        })
    return out


def point_in_rings(x, y, rings):
    """Even-odd ray cast over every ring of a (multi)polygon."""
    inside = False
    for ring in rings:
        for i in range(len(ring) - 1):
            x1, y1 = ring[i]
            x2, y2 = ring[i + 1]
            if (y1 > y) != (y2 > y):
                if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                    inside = not inside
    return inside


def rings_to_geojson(zones, extra=None):
    feats = []
    for z in zones:
        props = {
            "osm_id": z["id"],
            "name": z["name"],
            "name_fa": z["name_fa"],
            "admin_level": z["level"],
        }
        if extra:
            props.update(extra.get(z["id"], {}))
        if z.get("merged"):
            props["merged_zones"] = z["merged"]
        coords = [[list(p) for p in ring] for ring in z["rings"]]
        if len(coords) == 1:
            geom = {"type": "Polygon", "coordinates": coords}
        else:
            geom = {"type": "MultiPolygon", "coordinates": [[c] for c in coords]}
        feats.append({"type": "Feature", "properties": props, "geometry": geom})
    return {"type": "FeatureCollection", "features": feats}
