"""One Overpass client. The public mirrors 429/504 in bursts, so it rotates."""
from __future__ import annotations

import json
import os
import time

import requests

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
# the default python-requests User-Agent is rejected with HTTP 406
USER_AGENT = "traffic-estimate/1.0 (SUMO OD builder)"


class OverpassError(RuntimeError):
    pass


class OverpassClient:
    def __init__(self, mirrors=MIRRORS, tries: int = 4, timeout: int = 300,
                 user_agent: str = USER_AGENT):
        self.mirrors = list(mirrors)
        self.tries = tries
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}

    def query(self, overpass_ql: str) -> dict:
        last = None
        for attempt in range(self.tries):
            for url in self.mirrors:
                try:
                    response = requests.post(url, data={"data": overpass_ql},
                                             headers=self.headers,
                                             timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:
                    last = exc
                    print(f"  {url.split('/')[2]} failed ({type(exc).__name__})")
            if attempt < self.tries - 1:
                wait = 15 * 2 ** attempt
                print(f"  all mirrors busy, retrying in {wait}s "
                      f"(attempt {attempt + 2}/{self.tries})")
                time.sleep(wait)
        raise OverpassError(
            f"all Overpass mirrors failed after {self.tries} rounds: {last}\n"
            "Try again later, or work from a local .osm extract.")

    def centroids(self, body: str, bbox: str) -> list[tuple[float, float]]:
        """Feature centroids as (lon, lat). `body` is a {bbox}-templated union."""
        data = self.query(f"[out:json][timeout:180];({body.format(bbox=bbox)});"
                          "out center;")
        points = []
        for element in data.get("elements", []):
            place = element.get("center", element)
            if "lat" in place and "lon" in place:
                points.append((place["lon"], place["lat"]))
        return points

    def boundaries(self, bbox: tuple[float, float, float, float],
                   levels: set[str], cache: str | None = None,
                   refresh: bool = False) -> dict:
        """Administrative relations with full geometry, cached atomically.

        A truncated reply must never become the cache file, so the JSON is
        validated before it is moved into place.
        """
        if cache and not refresh and os.path.exists(cache):
            try:
                with open(cache, encoding="utf-8") as handle:
                    data = json.load(handle)
                print(f"  using cached {os.path.basename(cache)}")
                return data
            except (ValueError, OSError):
                print("  cached boundary file is corrupt, refetching")
                os.remove(cache)

        pattern = "|".join(sorted(levels))
        print(f"  querying Overpass for admin_level {pattern} ...")
        data = self.query(
            f'[out:json][timeout:180];'
            f'(rel["boundary"="administrative"]["admin_level"~"^({pattern})$"]'
            f'({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););out geom;')
        if "elements" not in data:
            raise OverpassError(f"unexpected Overpass response: {str(data)[:120]}")
        print(f"  got {len(data['elements'])} boundary relations")
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            partial = cache + ".part"
            with open(partial, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            os.replace(partial, cache)
        return data
