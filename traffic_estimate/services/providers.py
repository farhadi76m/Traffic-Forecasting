"""Live traffic providers.

Neshan is the one that works from Tehran: TomTom and HERE do not serve Iran
(the key authenticates, the Traffic calls fail).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests

NESHAN_URL = "https://api.neshan.org/v4/direction"
TOMTOM_URL = ("https://api.tomtom.com/traffic/services/4/flowSegmentData/"
              "absolute/10/json")
HERE_URL = "https://data.traffic.hereapi.com/v7/flow"


class ApiError(RuntimeError):
    """Carries the status so the caller can say what actually broke."""

    def __init__(self, status: int, body: str):
        self.status = status
        super().__init__(f"HTTP {status}: {body[:160]}")


class Provider(ABC):
    name = "provider"
    env_var = ""
    signup = ""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get(self.env_var)
        if not self.api_key:
            raise SystemExit(f"need a {self.name} key: --api-key or "
                             f"${self.env_var}\n  free key: {self.signup}")


class TravelTimeProvider(Provider):
    """Traffic-aware door-to-door driving time."""

    @abstractmethod
    def duration(self, origin: tuple[float, float],
                 destination: tuple[float, float]) -> tuple[float, float] | None:
        """(seconds, metres) for one (lon, lat) pair, or None if no route."""


class NeshanProvider(TravelTimeProvider):
    name = "Neshan"
    env_var = "NESHAN_API_KEY"
    signup = "https://platform.neshan.org"
    diagnosis = ("480 = bad/absent key; 4xx = quota. "
                 "Check the key at platform.neshan.org")

    def duration(self, origin, destination):
        (olon, olat), (dlon, dlat) = origin, destination
        response = requests.get(
            NESHAN_URL, headers={"Api-Key": self.api_key},
            params={"type": "car", "origin": f"{olat:.6f},{olon:.6f}",
                    "destination": f"{dlat:.6f},{dlon:.6f}"}, timeout=30)
        content_type = response.headers.get("content-type", "")
        payload = response.json() if content_type.startswith("application/json") else {}
        if response.status_code != 200 or payload.get("status") == "ERROR":
            raise ApiError(response.status_code,
                           payload.get("message", response.text))
        routes = payload.get("routes") or []
        if not routes:
            return None
        leg = routes[0]["legs"][0]
        return leg["duration"]["value"], leg["distance"]["value"]


class SpeedProvider(Provider):
    """Probe-measured speed on a road segment."""

    @abstractmethod
    def flow(self, lat: float, lon: float) -> tuple[float | None, float | None, float]:
        """(current km/h, free-flow km/h, confidence)."""


class TomTomProvider(SpeedProvider):
    name = "TomTom"
    env_var = "TOMTOM_API_KEY"
    signup = "https://developer.tomtom.com/user/register"
    diagnosis = ("403 -> key lacks the Traffic API entitlement, or is over quota\n"
                 "  400 -> point rejected (check lat/lon order above)\n"
                 "  Any -> TomTom may be geo-blocking this IP; try --provider here")

    def flow(self, lat, lon):
        response = requests.get(
            TOMTOM_URL, params={"key": self.api_key, "unit": "KMPH",
                                "point": f"{lat:.6f},{lon:.6f}"}, timeout=30)
        if response.status_code != 200:
            raise ApiError(response.status_code, response.text.strip())
        data = response.json()["flowSegmentData"]
        return data["currentSpeed"], data["freeFlowSpeed"], data.get("confidence", 1.0)


class HereProvider(SpeedProvider):
    name = "HERE"
    env_var = "HERE_API_KEY"
    signup = "https://developer.here.com"
    diagnosis = "HERE does not serve Iran; the key authenticates but returns nothing."

    def flow(self, lat, lon):
        response = requests.get(
            HERE_URL, params={"apiKey": self.api_key,
                              "locationReferencing": "shape",
                              "in": f"circle:{lat:.6f},{lon:.6f};r=50"}, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None, None, 0.0
        flow = results[0]["currentFlow"]
        return flow["speed"] * 3.6, flow["freeFlow"] * 3.6, flow.get("confidence", 1.0)


SPEED_PROVIDERS = {"tomtom": TomTomProvider, "here": HereProvider}
