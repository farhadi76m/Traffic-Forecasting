#!/usr/bin/env python3
"""Build a 24h OD matrix from real OSM land-use + a travel-time cost matrix.

Unlike generate_od.py (which invents a single CBD zone and weights zones by
edge count), this derives each zone's trip productions from residential OSM
features and its attractions from workplace/retail/education features, then
runs a doubly-constrained gravity model against real travel times.

    trips[i,j] = a_i * P_i * b_j * A_j * exp(-beta * cost[i,j])

with a_i / b_j found by IPF (Furness) so row sums match productions and column
sums match attractions. Morning hours flow home->work, evening work->home.

Example:
    python scripts/cost_matrix.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --backend sumo --out output/cost_matrix.csv
    python scripts/gravity_od.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --cost output/cost_matrix.csv \
        --out-dir output/od_gravity --num 20 --daily-trips 60000
"""
import argparse
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import sumolib

# the main instance 504s under load and rejects the default requests User-Agent
# with a 406, so we send a UA and fall through to mirrors
OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
UA = {"User-Agent": "traffic-estimate/1.0 (SUMO OD builder)"}

# OSM tags that stand in for "trips start here" vs "trips end here"
PRODUCTION_Q = """
  way["building"~"^(residential|apartments|house|detached|dormitory)$"]({bbox});
  way["landuse"="residential"]({bbox});
"""
ATTRACTION_Q = """
  node["shop"]({bbox});
  node["office"]({bbox});
  node["amenity"~"^(school|university|hospital|marketplace|bank|restaurant|cafe)$"]({bbox});
  way["building"~"^(commercial|retail|office|industrial|school|university|hospital)$"]({bbox});
  way["landuse"~"^(commercial|retail|industrial)$"]({bbox});
"""

HOUR_PROFILE = np.array([
    0.5, 0.3, 0.25, 0.25, 0.4, 1.0,
    2.5, 4.5, 4.2, 2.8, 2.2, 2.3,
    2.7, 3.1, 2.7, 2.8, 3.5, 4.4,
    4.6, 3.5, 2.5, 1.8, 1.2, 0.8,
])
# share of each hour's trips that is home->work (rest is the reverse direction)
OUTBOUND = np.array([
    .5, .5, .5, .5, .6, .75,
    .85, .9, .88, .8, .6, .5,
    .5, .45, .45, .4, .3, .2,
    .15, .2, .3, .4, .45, .5,
])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--taz", required=True)
    p.add_argument("--cost", required=True, help="CSV from cost_matrix.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num", type=int, default=20, help="number of scenario days")
    p.add_argument("--daily-trips", type=int, default=60000)
    p.add_argument("--beta", type=float, default=0.12,
                   help="deterrence per minute; higher = more local trips "
                        "(0.08-0.15 is typical for urban car travel)")
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--hours", default=None,
                   help="only emit these hours, e.g. '7,8' for an AM-peak "
                        "2-hour OD. Counts still come from the same daily "
                        "total and hourly profile, so the window is a slice "
                        "of a calibrated day, not a rescaled one. "
                        "Default: all 24 hours")
    p.add_argument("--cache", default="output/osm_weights.npz")
    p.add_argument("--osm", default="sumo/tehran_2026_area.osm",
                   help="local OSM extract; used instead of Overpass when present. "
                        "The public Overpass mirrors rate-limit and 502 constantly, "
                        "so the local file is both faster and reproducible")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# tag -> which side of the trip it feeds, when reading the local .osm extract
PROD_TAGS = [("building", {"residential", "apartments", "house", "detached",
                          "dormitory"}),
             ("landuse", {"residential"})]
ATTR_TAGS = [("shop", None), ("office", None),
             ("amenity", {"school", "university", "hospital", "marketplace",
                          "bank", "restaurant", "cafe"}),
             ("building", {"commercial", "retail", "office", "industrial",
                           "school", "university", "hospital"}),
             ("landuse", {"commercial", "retail", "industrial"})]


def osm_local_points(osm_file, net):
    """Read POI centroids straight out of the .osm extract, in net xy coords.

    Returns (production_pts, attraction_pts). Ways are reduced to the mean of
    their node coordinates -- good enough for assigning a building to a zone.
    """
    nodes = {}
    prod, attr = [], []

    def classify(tags):
        hit_p = any(k in tags and (v is None or tags[k] in v) for k, v in PROD_TAGS)
        hit_a = any(k in tags and (v is None or tags[k] in v) for k, v in ATTR_TAGS)
        return hit_p, hit_a

    for _, el in ET.iterparse(osm_file, events=("end",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("lon")), float(el.get("lat")))
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if tags:
                p, a = classify(tags)
                if p or a:
                    pt = nodes[el.get("id")]
                    (prod if p else attr).append(pt)
                    if p and a:
                        attr.append(pt)
        elif el.tag == "way":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            p, a = classify(tags)
            if p or a:
                pts = [nodes[nd.get("ref")] for nd in el.findall("nd")
                       if nd.get("ref") in nodes]
                if pts:
                    c = (float(np.mean([q[0] for q in pts])),
                         float(np.mean([q[1] for q in pts])))
                    if p:
                        prod.append(c)
                    if a:
                        attr.append(c)
            el.clear()
    to_xy = lambda ps: np.array([net.convertLonLat2XY(lo, la)  # noqa: E731
                                 for lo, la in ps]) if ps else np.empty((0, 2))
    return to_xy(prod), to_xy(attr)


def taz_polygons(taz_file):
    zones, polys = [], []
    for taz in sumolib.xml.parse(taz_file, "taz"):
        zones.append(taz.id)
        polys.append(np.array([tuple(map(float, p.split(","))) for p in taz.shape.split()]))
    return zones, polys


def points_in_poly(pts, poly):
    """Ray casting; shapely isn't installed in the traffic env."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    x1, y1 = poly[-1]
    for x2, y2 in poly:
        crosses = ((y1 > y) != (y2 > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xint)
        x1, y1 = x2, y2
    return inside


def overpass_points(query, bbox):
    """Fetch OSM features, return their centroids as (lon, lat)."""
    import requests
    q = f"[out:json][timeout:180];({query.format(bbox=bbox)});out center;"
    last = None
    # the public mirrors are heavily loaded and 429/504 in bursts; one pass over
    # them fails often, a few passes with backoff almost always gets through
    for attempt in range(4):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": q}, headers=UA, timeout=300)
                r.raise_for_status()
                return [(c["lon"], c["lat"])
                        for el in r.json()["elements"]
                        for c in [el.get("center", el)] if "lat" in c and "lon" in c]
            except Exception as e:
                last = e
                print(f"  {url.split('/')[2]} failed ({type(e).__name__})")
        wait = 15 * 2 ** attempt
        if attempt < 3:
            print(f"  all mirrors busy, retrying in {wait}s "
                  f"(attempt {attempt + 2}/4)")
            time.sleep(wait)
    raise SystemExit(f"all Overpass mirrors failed after 4 rounds: {last}\n"
                     "Try again later, or use --osm <file> with a building-rich "
                     "extract.")


def osm_weights(get_net, zones, polys, cache, osm_file=None):
    """get_net is a callable: the net is only parsed when the cache misses,
    which matters when this runs once per calibration iteration on a net that
    costs minutes to read."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["zones"]) == zones:
            print(f"weights from cache {cache}")
            return z["prod"], z["attr"]

    net = get_net()
    if osm_file and os.path.exists(osm_file):
        print(f"reading land use from {osm_file} (no Overpass needed)")
        prod_xy, attr_xy = osm_local_points(osm_file, net)
    else:
        b = net.getBoundary()
        lon0, lat0 = net.convertXY2LonLat(b[0], b[1])
        lon1, lat1 = net.convertXY2LonLat(b[2], b[3])
        bbox = f"{lat0},{lon0},{lat1},{lon1}"
        print(f"querying Overpass over {bbox} ...")
        prod_xy = np.array([net.convertLonLat2XY(lo, la)
                            for lo, la in overpass_points(PRODUCTION_Q, bbox)])
        attr_xy = np.array([net.convertLonLat2XY(lo, la)
                            for lo, la in overpass_points(ATTRACTION_Q, bbox)])

    prod = np.zeros(len(zones))
    attr = np.zeros(len(zones))
    for name, xy, target in [("residential", prod_xy, prod),
                             ("work/retail", attr_xy, attr)]:
        print(f"  {len(xy)} {name} features")
        for i, poly in enumerate(polys):
            target[i] = points_in_poly(xy, poly).sum() if len(xy) else 0

    # A zone with no OSM tags still has residents/shops, so floor it rather than
    # let it drop out of the model. But a floored zone contributes NO information
    # -- if many are floored the whole matrix is fiction, and that must be said
    # out loud rather than hidden behind a plausible-looking number.
    empty_p = int((prod == 0).sum())
    empty_a = int((attr == 0).sum())
    if empty_p or empty_a:
        print(f"  WARNING: {empty_p}/{len(zones)} zones have ZERO residential "
              f"features and {empty_a}/{len(zones)} have ZERO workplace features.")
        print("  Those zones are floored to 1 and carry no real signal. A road-only "
              ".osm\n  extract has few buildings -- pass --osm '' to use Overpass, "
              "which has far\n  better building coverage.")
    prod = np.maximum(prod, 1.0)
    attr = np.maximum(attr, 1.0)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez(cache, zones=np.array(zones, object), prod=prod, attr=attr)
    return prod, attr


def furness(prod, attr, deterrence, iters=50, tol=1e-4):
    """Doubly-constrained gravity: scale rows/cols until both margins match."""
    attr = attr * (prod.sum() / attr.sum())  # margins must be consistent
    a = np.ones_like(prod)
    b = np.ones_like(attr)
    for _ in range(iters):
        a = prod / np.maximum((deterrence * (b * attr)).sum(1), 1e-9)
        b = attr / np.maximum((deterrence.T * (a * prod)).sum(1), 1e-9)
        t = (a[:, None] * prod[:, None]) * (b[None, :] * attr[None, :]) * deterrence
        if np.abs(t.sum(1) - prod).max() < tol * prod.sum():
            break
    return t / t.sum()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cached_net = []

    def get_net():
        if not cached_net:
            print(f"loading {os.path.basename(args.net)} ...", flush=True)
            cached_net.append(sumolib.net.readNet(args.net, withInternal=False))
        return cached_net[0]

    zones, polys = taz_polygons(args.taz)

    rows = [l.strip().split(",") for l in open(args.cost) if l.strip()]
    cost_zones = rows[0][1:]
    cost = np.array([[float(v) for v in r[1:]] for r in rows[1:]]) / 60.0  # minutes
    keep = [i for i, z in enumerate(zones) if z in cost_zones]
    zones = [zones[i] for i in keep]
    polys = [polys[i] for i in keep]
    order = [cost_zones.index(z) for z in zones]
    cost = cost[np.ix_(order, order)]
    n = len(zones)

    prod, attr = osm_weights(get_net, zones, polys, args.cache, args.osm)
    prod, attr = prod[keep] if len(prod) != n else prod, attr[keep] if len(attr) != n else attr

    deterrence = np.exp(-args.beta * cost)
    base = furness(prod, attr, deterrence)          # home -> work shape
    hour_frac = HOUR_PROFILE / HOUR_PROFILE.sum()
    hours = ([int(h) for h in args.hours.split(",")] if args.hours
             else list(range(24)))
    print(f"{n} zones | mean trip cost "
          f"{(base * cost).sum():.1f} min | beta={args.beta} | "
          f"hours {hours[0]}-{hours[-1]} "
          f"({hour_frac[hours].sum() * 100:.0f}% of the daily total)")

    for s in range(args.num):
        day = float(np.clip(rng.normal(1.0, args.noise), 0.75, 1.25))
        root = ET.Element("data")
        total = 0
        for h in hours:
            # evening reverses the commute: transpose the same gravity shape
            w = OUTBOUND[h] * base + (1 - OUTBOUND[h]) * base.T
            counts = rng.poisson(w * args.daily_trips * hour_frac[h] * day)
            iv = ET.SubElement(root, "interval", id=f"h{h}",
                               begin=str(h * 3600), end=str((h + 1) * 3600))
            for i, j in zip(*np.nonzero(counts)):
                ET.SubElement(iv, "tazRelation", From=zones[i], to=zones[j],
                              count=str(int(counts[i, j])))
            total += int(counts.sum())
        out = os.path.join(args.out_dir, f"od_{s:02d}.xml")
        ET.indent(tree := ET.ElementTree(root))
        tree.write(out, encoding="UTF-8", xml_declaration=True)
        with open(out) as f:  # ElementTree forbids the reserved 'from' kwarg
            text = f.read().replace("From=", "from=")
        with open(out, "w") as f:
            f.write(text)
        print(f"{out}: {total} trips (day factor {day:.2f})")


if __name__ == "__main__":
    main()
