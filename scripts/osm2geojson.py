#!/usr/bin/env python3
"""Convert any OSM XML file to GeoJSON (WGS84). No geopandas/osmium needed.

Ways become LineStrings, or Polygons when they are closed and carry an area-ish
tag (building, landuse, natural, ...). type=multipolygon/boundary relations are
assembled from their member ways.

OSM extracts are often clipped: an extract can keep a relation but drop the ways
it is built from, in which case its polygon CANNOT be rebuilt from the file. This
script reports those instead of silently emitting broken geometry -- if your
boundaries come out incomplete, fetch them from Overpass rather than the extract
(that is what build_taz_districts.py does).

Usage:
  osm2geojson.py map.osm                        # everything tagged -> map.geojson
  osm2geojson.py map.osm --only highway         # just the roads
  osm2geojson.py map.osm --only boundary=administrative --split
  osm2geojson.py map.osm --only building,landuse -o out/areas.geojson
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# closed ways with any of these keys are areas, not lines
AREA_KEYS = {"building", "building:part", "landuse", "natural", "leisure",
             "amenity", "shop", "historic", "place", "boundary", "waterway",
             "man_made", "public_transport", "aeroway"}


def parse_only(spec):
    """'highway,boundary=administrative' -> [('highway', None), ('boundary', 'administrative')]"""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out.append((k.strip(), v.strip() or None))
    return out


def matches(tags, only):
    if not only:
        return bool(tags)
    return any(k in tags and (v is None or tags[k] == v) for k, v in only)


def is_area(tags, closed):
    if not closed:
        return False
    if tags.get("area") == "no":
        return False
    return tags.get("area") == "yes" or bool(AREA_KEYS & tags.keys())


def ring_chain(segments):
    """Chain way geometries into closed rings; returns (rings, n_unclosed)."""
    rings, used, unclosed = [], [False] * len(segments), 0
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
                    chain = other[:-1] + chain
                elif other[0] == chain[0]:
                    chain = other[::-1][:-1] + chain
                else:
                    continue
                used[j] = True
                extended = True
        if len(chain) < 4:
            continue
        if chain[0] != chain[-1]:
            unclosed += 1
            chain.append(chain[0])  # force-close a small gap
        rings.append(chain)
    return rings, unclosed


def main():
    ap = argparse.ArgumentParser(
        description="Convert an OSM XML file to GeoJSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("osm", help="input .osm / .osm.xml file")
    ap.add_argument("-o", "--out", help="output .geojson [default: <osm>.geojson]")
    ap.add_argument("--only", help="comma-separated tag filter, e.g. "
                                   "'highway' or 'boundary=administrative,landuse'")
    ap.add_argument("--split", action="store_true",
                    help="write one file per primary tag key instead of a single file")
    ap.add_argument("--no-points", action="store_true",
                    help="skip standalone tagged nodes (POIs)")
    args = ap.parse_args()

    if not os.path.isfile(args.osm):
        sys.exit(f"no such file: {args.osm}")
    only = parse_only(args.only)

    print(f"reading {args.osm}")
    root = ET.parse(args.osm).getroot()

    nodes, node_tags = {}, {}
    for n in root.findall("node"):
        nodes[n.get("id")] = (float(n.get("lon")), float(n.get("lat")))
        tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
        if tags:
            node_tags[n.get("id")] = tags

    ways = {}
    for w in root.findall("way"):
        ways[w.get("id")] = {
            "refs": [nd.get("ref") for nd in w.findall("nd")],
            "tags": {t.get("k"): t.get("v") for t in w.findall("tag")},
        }
    print(f"  {len(nodes)} nodes, {len(ways)} ways, {len(root.findall('relation'))} relations")

    feats = []
    kinds = Counter()
    skipped_missing_nodes = 0

    # --- nodes (POIs) -------------------------------------------------------
    if not args.no_points:
        for nid, tags in node_tags.items():
            if not matches(tags, only):
                continue
            feats.append({
                "type": "Feature",
                "id": f"node/{nid}",
                "properties": {"osm_id": nid, "osm_type": "node", **tags},
                "geometry": {"type": "Point", "coordinates": list(nodes[nid])},
            })
            kinds["Point"] += 1

    # --- ways ---------------------------------------------------------------
    way_in_rel = set()
    for rel in root.findall("relation"):
        rtags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if rtags.get("type") in ("multipolygon", "boundary"):
            for m in rel.findall("member"):
                if m.get("type") == "way":
                    way_in_rel.add(m.get("ref"))

    for wid, w in ways.items():
        tags = w["tags"]
        if not matches(tags, only):
            continue
        if not tags and wid in way_in_rel:
            continue  # untagged member of a multipolygon: emitted via the relation
        coords = [nodes[r] for r in w["refs"] if r in nodes]
        if len(coords) < 2 or len(coords) != len(w["refs"]):
            skipped_missing_nodes += 1
            if len(coords) < 2:
                continue
        closed = len(coords) > 3 and w["refs"][0] == w["refs"][-1]
        if is_area(tags, closed):
            geom = {"type": "Polygon", "coordinates": [[list(c) for c in coords]]}
            kinds["Polygon"] += 1
        else:
            geom = {"type": "LineString", "coordinates": [list(c) for c in coords]}
            kinds["LineString"] += 1
        feats.append({
            "type": "Feature",
            "id": f"way/{wid}",
            "properties": {"osm_id": wid, "osm_type": "way", **tags},
            "geometry": geom,
        })

    # --- relations (multipolygon / boundary) --------------------------------
    clipped = []
    for rel in root.findall("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if tags.get("type") not in ("multipolygon", "boundary"):
            continue
        if not matches(tags, only):
            continue
        members = [m for m in rel.findall("member") if m.get("type") == "way"]
        segs, have = [], 0
        for m in members:
            if m.get("role") not in ("outer", ""):
                continue
            w = ways.get(m.get("ref"))
            if not w:
                continue
            pts = [nodes[r] for r in w["refs"] if r in nodes]
            if len(pts) >= 2:
                segs.append(pts)
                have += 1
        name = tags.get("name:en") or tags.get("name") or rel.get("id")
        if not segs:
            clipped.append((name, 0, len(members)))
            continue
        rings, unclosed = ring_chain(segs)
        if not rings:
            clipped.append((name, have, len(members)))
            continue
        if have < len(members) or unclosed:
            clipped.append((name, have, len(members)))
        coords = [[list(p) for p in r] for r in rings]
        geom = ({"type": "Polygon", "coordinates": coords} if len(coords) == 1
                else {"type": "MultiPolygon", "coordinates": [[c] for c in coords]})
        kinds[geom["type"]] += 1
        feats.append({
            "type": "Feature",
            "id": f"relation/{rel.get('id')}",
            "properties": {"osm_id": rel.get("id"), "osm_type": "relation", **tags},
            "geometry": geom,
        })

    if not feats:
        sys.exit("no features matched --only; nothing written")

    # --- write --------------------------------------------------------------
    def dump(path, features):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features},
                      f, ensure_ascii=False)
        print(f"wrote {path}  ({len(features)} features)")

    base = args.out or os.path.splitext(args.osm)[0] + ".geojson"
    if args.split:
        keys = [k for k, _ in only] or ["highway", "building", "boundary",
                                        "landuse", "natural", "waterway", "amenity"]
        stem, ext = os.path.splitext(base)
        for k in keys:
            sub = [f for f in feats if k in f["properties"]]
            if sub:
                dump(f"{stem}_{k}{ext}", sub)
    else:
        dump(base, feats)

    print(f"  geometry: {dict(kinds)}")
    if skipped_missing_nodes:
        print(f"  note: {skipped_missing_nodes} ways referenced nodes not in the "
              f"extract (drawn from the nodes that are present)")
    if clipped:
        print(f"\n  WARNING: {len(clipped)} relation(s) are clipped in this extract -- "
              f"their member ways are missing, so the polygons are incomplete:")
        for name, have, total in clipped[:10]:
            print(f"    {name}: only {have}/{total} member ways present")
        if len(clipped) > 10:
            print(f"    ... and {len(clipped) - 10} more")
        print("  For district/boundary polygons, fetch them from Overpass instead "
              "(build_taz_districts.py does this).")


if __name__ == "__main__":
    main()
