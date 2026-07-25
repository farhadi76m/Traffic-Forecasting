#!/usr/bin/env python3
"""Build district GeoJSON + a TAZ file for any SUMO network, from OSM boundaries.

Administrative boundaries covering the network's bbox are fetched from Overpass
(the local .osm extract has its boundary relations clipped and cannot be used),
assembled into rings, and used to partition the network into TAZs. Only edges in
the largest strongly-connected component are assigned, so od2trips/duarouter
cannot emit trips into unreachable areas. Zones left with fewer than --min-edges
edges are merged into the nearest kept zone.

For Tehran: admin_level 9 = منطقه (district), 11 = محله (neighbourhood). Pick the
level that actually subdivides your network -- a net inside a single district
yields a single, useless TAZ at level 9.

Usage:
  build_taz_districts.py sumo/tehran_2026_area.net.xml --gui
  build_taz_districts.py sumo/prune_tab.net.xml --level 9 --min-edges 100

Outputs, named after the network (<prefix> = net filename without .net.xml):
  <prefix>_districts.geojson   district polygons (WGS84)
  <prefix>_network.geojson     network edges (WGS84), tagged with their TAZ
  <prefix>_districts.taz.xml   TAZ for od2trips (net coords)
  <prefix>_districts.poly.xml  filled polygons for sumo-gui
  <prefix>_districts.sumocfg   ready-to-open view (+ .view.xml)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import xml.sax.saxutils as saxutils

import sumolib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osm_boundaries import load_relations, point_in_rings, rings_to_geojson  # noqa: E402

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
MAX_TRIES = 4

# 8 visually distinct zone colours (R,G,B)
PALETTE = [
    (231, 76, 60), (46, 134, 193), (241, 196, 15), (39, 174, 96),
    (155, 89, 182), (230, 126, 34), (26, 188, 156), (52, 73, 94),
    (192, 57, 43), (41, 128, 185), (211, 84, 0), (127, 140, 141),
]


def fetch_boundaries(bbox, levels, cache):
    """Download admin boundaries covering bbox=(south, west, north, east).

    Overpass rate-limits and sometimes truncates responses, so the reply is
    validated as JSON *before* it becomes the cache file -- a failed download
    must never poison the cache.
    """
    if os.path.exists(cache):
        try:
            json.load(open(cache, encoding="utf-8"))
            print(f"  using cached {os.path.basename(cache)}")
            return cache
        except (ValueError, OSError):
            print("  cached boundary file is corrupt, refetching")
            os.remove(cache)

    lv = "|".join(sorted(levels))
    query = (
        f'[out:json][timeout:180];\n'
        f'(rel["boundary"="administrative"]["admin_level"~"^({lv})$"]'
        f'({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););\n'
        f'out geom;'
    )
    tmp = cache + ".part"
    last = ""
    for attempt in range(1, MAX_TRIES + 1):
        mirror = OVERPASS_MIRRORS[(attempt - 1) % len(OVERPASS_MIRRORS)]
        print(f"  querying Overpass for admin_level {lv} "
              f"(try {attempt}/{MAX_TRIES}, {mirror.split('/')[2]}) ...")
        try:
            res = subprocess.run(
                ["curl", "-sS", "--retry", "2", "--max-time", "240",
                 "-A", "traffic-estimate/1.0 (SUMO TAZ builder)",
                 "-X", "POST", "-d", query, mirror, "-o", tmp],
                capture_output=True, text=True, timeout=300,
            )
            if res.returncode != 0:
                last = res.stderr.strip() or f"curl exit {res.returncode}"
            else:
                data = json.load(open(tmp, encoding="utf-8"))  # ValueError if truncated
                if "elements" not in data:
                    last = f"unexpected response: {str(data)[:120]}"
                else:
                    os.replace(tmp, cache)  # atomic: only a good reply lands
                    print(f"  got {len(data['elements'])} boundary relations")
                    return cache
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        print(f"    failed ({last})")
        if attempt < MAX_TRIES:
            wait = 5 * attempt
            print(f"    retrying in {wait}s ...")
            time.sleep(wait)

    if os.path.exists(tmp):
        os.remove(tmp)
    sys.exit(f"could not fetch boundaries from Overpass after {MAX_TRIES} tries "
             f"({last}).\nOverpass rate-limits heavy use; wait a minute and rerun, "
             f"or drop a previously downloaded Overpass JSON at:\n  {cache}")


def largest_scc(edges):
    """Kosaraju: biggest strongly-connected set of edges (edge = node in the graph)."""
    ids = {e.getID() for e in edges}
    succ = {e.getID(): [o.getID() for o in e.getOutgoing() if o.getID() in ids]
            for e in edges}
    pred = {i: [] for i in ids}
    for u, vs in succ.items():
        for v in vs:
            pred[v].append(u)

    order, seen = [], set()
    for s in ids:  # iterative DFS, post-order
        if s in seen:
            continue
        stack = [(s, iter(succ[s]))]
        seen.add(s)
        while stack:
            node, it = stack[-1]
            for nxt in it:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, iter(succ[nxt])))
                    break
            else:
                order.append(stack.pop()[0])

    seen, best = set(), set()
    for s in reversed(order):
        if s in seen:
            continue
        comp, stack = set(), [s]
        seen.add(s)
        while stack:
            node = stack.pop()
            comp.add(node)
            for p in pred[node]:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        if len(comp) > len(best):
            best = comp
    return best


def clip_to_box(ring, box):
    """Sutherland-Hodgman clip of a ring against rectangle (xmin,ymin,xmax,ymax)."""
    xmin, ymin, xmax, ymax = box
    edges_of_box = [
        (lambda p: p[0] >= xmin, lambda a, b: (xmin, a[1] + (b[1]-a[1])*(xmin-a[0])/(b[0]-a[0]))),
        (lambda p: p[0] <= xmax, lambda a, b: (xmax, a[1] + (b[1]-a[1])*(xmax-a[0])/(b[0]-a[0]))),
        (lambda p: p[1] >= ymin, lambda a, b: (a[0] + (b[0]-a[0])*(ymin-a[1])/(b[1]-a[1]), ymin)),
        (lambda p: p[1] <= ymax, lambda a, b: (a[0] + (b[0]-a[0])*(ymax-a[1])/(b[1]-a[1]), ymax)),
    ]
    poly = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring[:]
    for inside, isect in edges_of_box:
        if not poly:
            return []
        out = []
        for i, cur in enumerate(poly):
            prev = poly[i - 1]
            cin, pin = inside(cur), inside(prev)
            if cin:
                if not pin:
                    out.append(isect(prev, cur))
                out.append(cur)
            elif pin:
                out.append(isect(prev, cur))
        poly = out
    return poly


def main():
    ap = argparse.ArgumentParser(
        description="Build district GeoJSON + TAZ for a SUMO network from OSM boundaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("net", help="SUMO .net.xml to build the TAZ for")
    ap.add_argument("-o", "--outdir",
                    help="output directory [default: the network's directory]")
    ap.add_argument("-p", "--prefix",
                    help="basename for the outputs "
                         "[default: derived from the network filename]")
    ap.add_argument("-l", "--level", default="11",
                    help="OSM admin_level to use as zones (Tehran: 9=منطقه, 11=محله)")
    ap.add_argument("-m", "--min-edges", type=int, default=40,
                    help="zones with fewer routable edges are merged into the nearest zone")
    ap.add_argument("--pad", type=float, default=0.006,
                    help="padding added to the network bbox, in degrees")
    ap.add_argument("--vclass", default="passenger",
                    help="vehicle class the TAZ edges must allow")
    ap.add_argument("--no-scc", action="store_true",
                    help="keep every edge instead of only the largest strongly-connected "
                         "component (routing will then drop trips into disconnected areas)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-query Overpass even if a cached boundary file exists")
    ap.add_argument("--gui", action="store_true",
                    help="open the result in sumo-gui when done")
    ap.add_argument("--list-levels", action="store_true",
                    help="list the admin_levels OSM has over this network and exit "
                         "(use this first on a new map: the levels differ per country)")
    args = ap.parse_args()

    if not os.path.isfile(args.net):
        sys.exit(f"no such network: {args.net}")
    args.net = os.path.abspath(args.net)
    outdir = os.path.abspath(args.outdir) if args.outdir else os.path.dirname(args.net)
    # tehran_2026_area.net.xml -> tehran_2026_area
    prefix = args.prefix or os.path.basename(args.net).split(".net.xml")[0].split(".xml")[0]
    os.makedirs(outdir, exist_ok=True)

    print(f"reading {args.net}")
    net = sumolib.net.readNet(args.net)
    x0, y0, x1, y1 = net.getBoundary()
    s, w = net.convertXY2LonLat(x0, y0)[::-1]
    n, e = net.convertXY2LonLat(x1, y1)[::-1]
    bbox = (s - args.pad, w - args.pad, n + args.pad, e + args.pad)
    print(f"  net bbox lat {bbox[0]:.4f}..{bbox[2]:.4f} lon {bbox[1]:.4f}..{bbox[3]:.4f}")

    if args.list_levels:
        probe = os.path.join(outdir, f"{prefix}_osm_levels.json")
        fetch_boundaries(bbox, {"[0-9]+"}, probe)
        by_level = {}
        for z in load_relations(probe, None):
            by_level.setdefault(z["level"], []).append(z["name"])
        print(f"\nadmin_levels covering {os.path.basename(args.net)}:\n")
        print(f"  {'level':<7}{'zones':<7}examples")
        for lvl in sorted(by_level, key=lambda x: int(x)):
            names = by_level[lvl]
            print(f"  {lvl:<7}{len(names):<7}{', '.join(sorted(names)[:4])}"
                  f"{' ...' if len(names) > 4 else ''}")
        print("\npick the level whose zone count actually subdivides your net, then:")
        print(f"  {os.path.basename(sys.argv[0])} {args.net} --level <L>")
        os.remove(probe)
        return

    # cache is per net+level+bbox, so a different network never reuses these polygons
    cache = os.path.join(outdir, f"{prefix}_osm_boundaries_l{args.level}.json")
    if args.refresh and os.path.exists(cache):
        os.remove(cache)
    fetch_boundaries(bbox, {"9", args.level}, cache)
    zones = load_relations(cache, {args.level})
    print(f"  {len(zones)} admin_level={args.level} polygons")
    if not zones:
        sys.exit(f"no admin_level={args.level} boundaries cover this network; "
                 f"try a different --level")

    # --- assign edges -------------------------------------------------------
    car = [ed for ed in net.getEdges()
           if not ed.isSpecial() and ed.allows(args.vclass)]
    if not car:
        sys.exit(f"no edges allow vclass '{args.vclass}'")
    total = len(car)
    if args.no_scc:
        print(f"  {total} '{args.vclass}' edges (--no-scc: not filtered)")
    else:
        scc = largest_scc(car)
        car = [ed for ed in car if ed.getID() in scc]
        print(f"  {len(car)}/{total} '{args.vclass}' edges in the largest "
              f"strongly-connected component ({len(car) / total * 100:.1f}%)")

    members = {z["id"]: [] for z in zones}
    for ed in car:
        px, py = sumolib.geomhelper.positionAtShapeOffset(ed.getShape(), ed.getLength() / 2.0)
        lon, lat = net.convertXY2LonLat(px, py)
        for z in zones:
            if point_in_rings(lon, lat, z["rings"]):
                members[z["id"]].append(ed.getID())
                break

    zones = [z for z in zones if members[z["id"]]]
    for z in zones:
        cx = sum(p[0] for r in z["rings"] for p in r) / sum(len(r) for r in z["rings"])
        cy = sum(p[1] for r in z["rings"] for p in r) / sum(len(r) for r in z["rings"])
        z["centroid"] = (cx, cy)

    # --- merge small zones into the nearest kept zone ------------------------
    keep = [z for z in zones if len(members[z["id"]]) >= args.min_edges]
    small = [z for z in zones if len(members[z["id"]]) < args.min_edges]
    if not keep:
        sys.exit("no zone reaches --min-edges; lower the threshold")
    for z in small:
        tgt = min(keep, key=lambda k: (k["centroid"][0] - z["centroid"][0]) ** 2
                                      + (k["centroid"][1] - z["centroid"][1]) ** 2)
        print(f"  merging {z['name']} ({len(members[z['id']])} edges) -> {tgt['name']}")
        members[tgt["id"]] += members[z["id"]]
        tgt.setdefault("merged", []).append(z["name"])
        tgt["rings"] = tgt["rings"] + z["rings"]  # keep both shapes for display

    zones = keep
    seen = {}
    for i, z in enumerate(zones):
        z["color"] = PALETTE[i % len(PALETTE)]
        tid = "".join(c if c.isalnum() else "_" for c in z["name"])
        while "__" in tid:
            tid = tid.replace("__", "_")
        tid = tid.strip("_") or f"taz{z['id']}"
        if tid in seen:  # two zones can sanitise to the same id
            seen[tid] += 1
            tid = f"{tid}_{seen[tid]}"
        else:
            seen[tid] = 0
        z["taz_id"] = tid
    print(f"\n  {len(zones)} final TAZs:")
    for z in sorted(zones, key=lambda z: -len(members[z["id"]])):
        extra = f"  (+{', '.join(z['merged'])})" if z.get("merged") else ""
        print(f"    {len(members[z['id']]):>5} edges  {z['taz_id']}{extra}")

    counts = {z["id"]: {"edge_count": len(members[z["id"]]), "taz_id": z["taz_id"]}
              for z in zones}

    # --- 1. district GeoJSON ------------------------------------------------
    gj = os.path.join(outdir, f"{prefix}_districts.geojson")
    with open(gj, "w", encoding="utf-8") as f:
        json.dump(rings_to_geojson(zones, counts), f, ensure_ascii=False, indent=1)
    print(f"\nwrote {gj}")

    # --- 2. network GeoJSON -------------------------------------------------
    zone_of = {eid: z["taz_id"] for z in zones for eid in members[z["id"]]}
    feats = []
    for ed in car:
        coords = [list(net.convertXY2LonLat(px, py)) for px, py in ed.getShape()]
        feats.append({
            "type": "Feature",
            "properties": {
                "id": ed.getID(),
                "name": ed.getName(),
                "type": ed.getType(),
                "speed_kmh": round(ed.getSpeed() * 3.6, 1),
                "lanes": ed.getLaneNumber(),
                "length_m": round(ed.getLength(), 1),
                "taz": zone_of.get(ed.getID()),
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    ngj = os.path.join(outdir, f"{prefix}_network.geojson")
    with open(ngj, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    print(f"wrote {ngj}  ({len(feats)} edges)")

    # --- 3. TAZ + 4. polygons (net coordinates) -----------------------------
    box = net.getBoundary()
    taz_f = os.path.join(outdir, f"{prefix}_districts.taz.xml")
    poly_f = os.path.join(outdir, f"{prefix}_districts.poly.xml")
    with open(taz_f, "w", encoding="utf-8") as ft, open(poly_f, "w", encoding="utf-8") as fp:
        ft.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        ft.write('<!-- TAZ from OSM admin_level=%s for %s, built by build_taz_districts.py -->\n'
                 % (args.level, os.path.basename(args.net)))
        ft.write('<tazs xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/taz_file.xsd">\n')
        fp.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')

        for z in zones:
            col = "%d,%d,%d" % z["color"]
            xy_rings = []
            for ring in z["rings"]:
                pts = [net.convertLonLat2XY(lon, lat) for lon, lat in ring]
                cl = clip_to_box(pts, box)
                if len(cl) >= 3:
                    xy_rings.append(cl)
            if not xy_rings:
                continue
            biggest = max(xy_rings, key=len)
            shape = " ".join("%.2f,%.2f" % p for p in biggest + [biggest[0]])
            nm = saxutils.quoteattr(z["name_fa"] or z["name"])

            ft.write('    <taz id="%s" name=%s shape="%s" color="%s" edges="%s"/>\n'
                     % (z["taz_id"], nm, shape, col, " ".join(sorted(members[z["id"]]))))
            # translucent fill so the network stays readable underneath
            for k, ring in enumerate(xy_rings):
                sh = " ".join("%.2f,%.2f" % p for p in ring + [ring[0]])
                pid = z["taz_id"] if k == 0 else "%s.%d" % (z["taz_id"], k)
                fp.write('    <poly id="%s" type="district" color="%s,70" fill="1" '
                         'layer="-10" lineWidth="4" shape="%s"/>\n' % (pid, col, sh))
        ft.write("</tazs>\n")
        fp.write("</additional>\n")
    print(f"wrote {taz_f}")
    print(f"wrote {poly_f}")

    # --- 5. view settings + sumocfg so it opens ready to look at ------------
    view_f = os.path.join(outdir, f"{prefix}_districts.view.xml")
    with open(view_f, "w", encoding="utf-8") as f:
        f.write(
            '<viewsettings>\n'
            '    <scheme name="district view">\n'
            '        <opengl dither="0" fps="0"/>\n'
            '        <background backgroundColor="255,255,255" showGrid="0"/>\n'
            '        <edges laneEdgeMode="0" scaleMode="0" laneShowBorders="1"\n'
            '               showLinkDecals="1" showRails="1" hideConnectors="0"\n'
            '               edgeName_show="0" streetName_show="0">\n'
            '            <colorScheme name="uniform">\n'
            '                <entry color="80,80,80"/>\n'
            '            </colorScheme>\n'
            '        </edges>\n'
            '        <polys polyType_show="0">\n'
            '            <polyName show="1" size="55" color="0,0,0"/>\n'
            '        </polys>\n'
            '    </scheme>\n'
            '</viewsettings>\n'
        )
    cfg_f = os.path.join(outdir, f"{prefix}_districts.sumocfg")
    net_rel = os.path.relpath(args.net, outdir)
    # Only the poly file: sumo-gui turns a taz 'shape' into a polygon of the same
    # id, so loading the taz file here too would collide on every zone id.
    with open(cfg_f, "w", encoding="utf-8") as f:
        f.write(
            '<configuration>\n'
            '    <input>\n'
            '        <net-file value="%s"/>\n'
            '        <additional-files value="%s"/>\n'
            '    </input>\n'
            '    <gui_only>\n'
            '        <gui-settings-file value="%s"/>\n'
            '    </gui_only>\n'
            '</configuration>\n'
            % (net_rel, os.path.basename(poly_f), os.path.basename(view_f))
        )
    print(f"wrote {view_f}")
    print(f"wrote {cfg_f}")

    if args.gui:
        print(f"\nopening sumo-gui ...")
        subprocess.Popen(["sumo-gui", "-c", cfg_f],
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        print(f"  -> sumo-gui -c {cfg_f}")


if __name__ == "__main__":
    main()
