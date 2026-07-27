#!/usr/bin/env python3
"""Build a zone-to-zone travel-time (impedance) matrix for a TAZ file.

This is the *cost* matrix, not a demand matrix: cell (i, j) is how many seconds
it takes to drive from zone i to zone j. It is the input to the deterrence term
of the gravity model in gravity_od.py -- no routing API can give you trip
counts directly.

Backends:
  sumo    shortest path on our own net (free, offline, respects the pruned net)
  osrm    public OSRM demo server (free, rate limited, real OSM road speeds)
  arcgis  ArcGIS travelCostMatrix service (needs ARCGIS_API_KEY, has live/
          historical traffic, see developers.arcgis.com/rest/routing/)

Example:
    python scripts/cost_matrix.py --net sumo/prune_tab.net.xml \
        --taz sumo/taz_od.xml --backend sumo --out output/cost_matrix.csv
"""
import argparse
import heapq
import os
import time

import numpy as np
import sumolib

ARCGIS_URL = ("https://route-api.arcgis.com/arcgis/rest/services/World/"
              "OriginDestinationCostMatrix/NAServer/"
              "OriginDestinationCostMatrix_World/solveODCostMatrix")
OSRM_URL = "https://router.project-osrm.org/table/v1/driving/"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--taz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", choices=["sumo", "osrm", "arcgis"], default="sumo")
    p.add_argument("--api-key", default=os.environ.get("ARCGIS_API_KEY"),
                   help="ArcGIS key; defaults to $ARCGIS_API_KEY")
    p.add_argument("--depart", default=None,
                   help="arcgis only: departure time as epoch ms, for traffic-aware "
                        "costs (e.g. next Tuesday 08:00 local)")
    p.add_argument("--allow-partial", action="store_true",
                   help="proceed even if the TAZ file does not fit this net")
    return p.parse_args()


def zone_centroids(net, taz_file):
    """Return (zone_ids, xy centroids, one representative edge per zone).

    The centroid of the TAZ polygon can land on a building, so for routing we
    snap to the middle of the zone's most central edge instead.
    """
    zones, xys, edges = [], [], []
    for taz in sumolib.xml.parse(taz_file, "taz"):
        pts = [tuple(map(float, p.split(","))) for p in taz.shape.split()]
        cx, cy = np.mean(pts, axis=0)
        zone_edges = [net.getEdge(e) for e in taz.edges.split()
                      if net.hasEdge(e) and net.getEdge(e).allows("passenger")]
        if not zone_edges:
            print(f"  skip {taz.id}: no passenger edges")
            continue
        # edge whose midpoint is closest to the polygon centroid
        best = min(zone_edges, key=lambda e: (e.getShape()[len(e.getShape()) // 2][0] - cx) ** 2
                                             + (e.getShape()[len(e.getShape()) // 2][1] - cy) ** 2)
        mid = best.getShape()[len(best.getShape()) // 2]
        zones.append(taz.id)
        xys.append(mid)
        edges.append(best)
    return zones, np.array(xys), edges


def costs_sumo(net, edges, vclass="passenger"):
    """Travel-time matrix by one Dijkstra per ORIGIN, not one per pair.

    net.getOptimalPath is a single-pair search, so filling an n x n matrix with
    it costs n^2 full searches -- fine for 6 zones, hopeless for a city (300
    zones over 218k edges is ~90,000 of them). One relaxation per origin sweeps
    the whole graph and reads off every destination on the way, which is the
    same answer in n searches instead of n^2.

    Cost is seconds and includes both the origin and destination edge, so the
    diagonal stays the single-edge time the pairwise version used.
    """
    n = len(edges)
    m = np.full((n, n), np.nan)
    index = {e.getID(): j for j, e in enumerate(edges)}

    def tt(e):
        return e.getLength() / max(e.getSpeed(), 1.0)

    for i, src in enumerate(edges):
        # (cost, edge id) only -- Edge objects are not orderable, so they must
        # never end up as a heap tie-breaker
        pq = [(tt(src), src.getID())]
        settled = {}
        remaining = n
        while pq and remaining:
            d, eid = heapq.heappop(pq)
            if eid in settled:
                continue
            settled[eid] = d
            j = index.get(eid)
            if j is not None:
                m[i, j] = d
                remaining -= 1
            for nxt in net.getEdge(eid).getOutgoing():
                if nxt.getID() not in settled and nxt.allows(vclass):
                    heapq.heappush(pq, (d + tt(nxt), nxt.getID()))
        m[i, i] = tt(src)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} origins", flush=True)
    return m


def costs_osrm(lonlats):
    import requests
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in lonlats)
    r = requests.get(OSRM_URL + coords, params={"annotations": "duration"}, timeout=120)
    r.raise_for_status()
    return np.array(r.json()["durations"], dtype=float)


def costs_arcgis(lonlats, api_key, depart=None, chunk=100):
    """ArcGIS travelCostMatrix. Charged per origin-destination pair, so the
    request is chunked over origins to stay under the per-call pair limit."""
    import requests
    if not api_key:
        raise SystemExit("ArcGIS backend needs --api-key or $ARCGIS_API_KEY "
                         "(free key: developers.arcgis.com)")

    def featureset(pts, offset=0):
        return {
            "spatialReference": {"wkid": 4326},
            "features": [{"geometry": {"x": lon, "y": lat},
                          "attributes": {"Name": str(offset + k), "ObjectID": offset + k + 1}}
                         for k, (lon, lat) in enumerate(pts)],
        }

    n = len(lonlats)
    m = np.full((n, n), np.nan)
    per_call = max(1, chunk // max(1, n // 100 + 1))
    for start in range(0, n, per_call):
        origins = lonlats[start:start + per_call]
        params = {
            "f": "json",
            "token": api_key,
            "origins": str(featureset(origins, start)).replace("'", '"'),
            "destinations": str(featureset(lonlats)).replace("'", '"'),
            "travelMode": "",  # empty => service default driving mode
            "impedanceAttributeName": "TravelTime",
            "outputType": "esriNAODOutputSparseMatrix",
            "returnOrigins": "false",
            "returnDestinations": "false",
        }
        if depart:
            params["timeOfDay"] = depart
            params["timeOfDayUsage"] = "start"
        r = requests.post(ARCGIS_URL, data=params, timeout=180)
        r.raise_for_status()
        js = r.json()
        if "error" in js:
            raise SystemExit(f"ArcGIS error: {js['error']}")
        # sparse matrix: {"<origin idx>": {"<dest idx>": [cost, ...]}}, 1-based
        for o, dests in js["odCostMatrix"].items():
            if not o.isdigit():
                continue  # skip the "costAttributeNames" entry
            for d, vals in dests.items():
                m[start + int(o) - 1, int(d) - 1] = vals[0] * 60.0  # minutes -> s
        print(f"  origins {start}-{start + len(origins) - 1} done")
        time.sleep(0.2)
    return m


def main():
    args = parse_args()
    net = sumolib.net.readNet(args.net, withInternal=False)
    n_taz = sum(1 for _ in sumolib.xml.parse(args.taz, "taz"))
    zones, xys, edges = zone_centroids(net, args.taz)
    print(f"{len(zones)}/{n_taz} routable zones, backend={args.backend}")

    # A TAZ file built for a different net silently loses zones here -- taz_od.xml
    # is a prune_tab artifact and drops 6 of its 13 zones on tehran_2026_area.
    # Losing half the study area must not be a warning you can scroll past.
    if len(zones) < 0.8 * n_taz and not args.allow_partial:
        raise SystemExit(
            f"\n{n_taz - len(zones)} of {n_taz} zones have no routable edge on "
            f"{os.path.basename(args.net)}.\nThat TAZ file was almost certainly "
            f"built for a different net -- its edge ids do not match this one.\n"
            f"Rebuild the TAZ against this net, or pass the net it was built for.\n"
            f"Pass --allow-partial to proceed anyway (you will model only "
            f"{len(zones)} zones).")

    if args.backend == "sumo":
        m = costs_sumo(net, edges)
    else:
        lonlats = [net.convertXY2LonLat(x, y) for x, y in xys]
        m = (costs_osrm(lonlats) if args.backend == "osrm"
             else costs_arcgis(lonlats, args.api_key, args.depart))

    unreachable = int(np.isnan(m).sum())
    if unreachable:
        # island zones (e.g. Shahran) never route; fill so gravity just sees them
        # as very far away rather than dropping them
        print(f"warning: {unreachable} unreachable pairs, filling with 3x max cost")
        m[np.isnan(m)] = np.nanmax(m) * 3

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("," + ",".join(zones) + "\n")
        for i, z in enumerate(zones):
            f.write(z + "," + ",".join(f"{v:.1f}" for v in m[i]) + "\n")
    print(f"{args.out}: {m.shape[0]}x{m.shape[1]}, "
          f"mean {np.mean(m) / 60:.1f} min, max {np.max(m) / 60:.1f} min")


if __name__ == "__main__":
    main()
