#!/usr/bin/env python3
"""Convert any OSM XML file to GeoJSON (WGS84). No geopandas or osmium needed.

Ways become LineStrings, or Polygons when closed and carrying an area-ish tag.
multipolygon/boundary relations are assembled from their member ways.

Extracts are often clipped: one can keep a relation but drop the ways it is
built from, in which case its polygon CANNOT be rebuilt. Those are reported
rather than emitted as broken geometry -- if your boundaries come out
incomplete, fetch them from Overpass instead (build_taz_districts.py does).

    osm2geojson.py map.osm                        # everything tagged
    osm2geojson.py map.osm --only highway         # just the roads
    osm2geojson.py map.osm --only boundary=administrative --split
    osm2geojson.py map.osm --only building,landuse -o out/areas.geojson
"""
import argparse
import os
import sys

import _path  # noqa: F401

from traffic_estimate.osm import OsmGeoJson, parse_tag_filter, write_geojson


def main():
    parser = argparse.ArgumentParser(
        description="Convert an OSM XML file to GeoJSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("osm", help="input .osm / .osm.xml file")
    parser.add_argument("-o", "--out",
                        help="output .geojson [default: <osm>.geojson]")
    parser.add_argument("--only",
                        help="comma-separated tag filter, e.g. 'highway' or "
                             "'boundary=administrative,landuse'")
    parser.add_argument("--split", action="store_true",
                        help="write one file per primary tag key")
    parser.add_argument("--no-points", action="store_true",
                        help="skip standalone tagged nodes (POIs)")
    args = parser.parse_args()

    if not os.path.isfile(args.osm):
        sys.exit(f"no such file: {args.osm}")

    rules = parse_tag_filter(args.only)
    print(f"reading {args.osm}")
    converter = OsmGeoJson(args.osm, rules, include_points=not args.no_points)
    features = converter.convert()
    if not features:
        sys.exit("no features matched --only; nothing written")

    def dump(path, subset):
        write_geojson(path, {"type": "FeatureCollection", "features": subset})
        print(f"wrote {path}  ({len(subset)} features)")

    base = args.out or os.path.splitext(args.osm)[0] + ".geojson"
    if args.split:
        keys = [k for k, _ in rules] or ["highway", "building", "boundary",
                                         "landuse", "natural", "waterway",
                                         "amenity"]
        stem, ext = os.path.splitext(base)
        for key in keys:
            subset = [f for f in features if key in f["properties"]]
            if subset:
                dump(f"{stem}_{key}{ext}", subset)
    else:
        dump(base, features)
    converter.report()


if __name__ == "__main__":
    main()
