"""OD generators. Each produces expected trips; the base class samples and writes."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterator, Sequence

import numpy as np

from ..odmatrix import ODMatrix
from ..taz import TazSet
from .gravity import deterrence, furness
from .landuse import LandUseWeights
from .profiles import EVENING, MORNING, OUTBOUND, DayShape


class ODGenerator(ABC):
    """Turns a zoning into a stream of daily OD matrices.

    Subclasses only decide the expected trips per (hour, origin, destination);
    Poisson sampling, hour selection and serialisation live here.
    """

    def __init__(self, taz: TazSet, *, daily_trips: float,
                 hours: Sequence[int] | None = None, seed: int = 42):
        self.taz = taz
        self.zones = taz.ids
        self.daily_trips = daily_trips
        self.hours = np.arange(24) if hours is None else np.asarray(hours, int)
        self.rng = np.random.default_rng(seed)

    @property
    def n_zones(self) -> int:
        return len(self.zones)

    @abstractmethod
    def expected(self, rng: np.random.Generator) -> np.ndarray:
        """(len(hours), n, n) expected trips for one day."""

    def matrix(self) -> ODMatrix:
        return ODMatrix(self.zones, self.rng.poisson(self.expected(self.rng)),
                        self.hours)

    def generate(self, count: int) -> Iterator[ODMatrix]:
        for _ in range(count):
            yield self.matrix()

    def write_scenarios(self, out_dir: str, count: int,
                        pattern: str = "od_{index:02d}.xml") -> list[str]:
        """Write `count` days, one file at a time -- a city day is millions of rows."""
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for index, matrix in enumerate(self.generate(count)):
            path = matrix.write(os.path.join(out_dir, pattern.format(index=index)))
            paths.append(path)
            print(f"{path}: {matrix.total} trips")
        return paths


class TypicalDayGenerator(ODGenerator):
    """Many days of the *same* city: one gravity shape plus day-to-day noise.

    Zones are weighted by their edge count and the largest one plays the CBD:
    trips flow into it in the morning and back out in the evening.
    """

    def __init__(self, taz: TazSet, *, daily_trips: float = 6000,
                 noise: float = 0.08, commute_strength: float = 1.5,
                 hours=None, seed: int = 42):
        super().__init__(taz, daily_trips=daily_trips, hours=hours, seed=seed)
        self.noise = noise
        self.commute_strength = commute_strength
        self.shape = DayShape()
        sizes = taz.sizes
        self.center = int(np.argmax(sizes))
        weights = sizes / sizes.sum()
        self.base = np.outer(weights, weights)

    @property
    def center_zone(self) -> str:
        return self.zones[self.center]

    def expected(self, rng: np.random.Generator) -> np.ndarray:
        day_factor = float(np.clip(rng.normal(1.0, self.noise), 0.75, 1.25))
        weights = _commute_weights(self.base, self.center, self.hours,
                                   self.commute_strength)
        return (weights * self.daily_trips * day_factor
                * self.shape.fractions[self.hours][:, None, None])


class PriorSampler(ODGenerator):
    """Every scenario is a *different* city, drawn from a log-normal prior.

    This is the sampling distribution a SUMO surrogate is trained on, and
    therefore the prior of the inverse problem.
    """

    def __init__(self, taz: TazSet, *, daily_trips: float = 12000,
                 sigma_total: float = 0.35, sigma_zone: float = 0.45,
                 sigma_cell: float = 0.35, sigma_hour: float = 0.30,
                 max_trips: int = 0, hours=None, seed: int = 1):
        super().__init__(taz, daily_trips=daily_trips, hours=hours, seed=seed)
        self.sigma_total = sigma_total
        self.sigma_zone = sigma_zone
        self.sigma_cell = sigma_cell
        self.sigma_hour = sigma_hour
        self.max_trips = max_trips
        self.shape = DayShape()
        sizes = taz.sizes
        self.center = int(np.argmax(sizes))
        self.size_weights = sizes / sizes.sum()

    def expected(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n_zones
        lognormal = lambda sigma, size=None: np.exp(rng.normal(0, sigma, size))

        total = self.daily_trips * lognormal(self.sigma_total)
        if self.max_trips:
            total = min(total, self.max_trips)
        produce = self.size_weights * lognormal(self.sigma_zone, n)
        attract = self.size_weights * lognormal(self.sigma_zone, n)
        base = np.outer(produce, attract) * lognormal(self.sigma_cell, (n, n))
        fractions = self.shape.jittered(rng, self.sigma_hour).fractions

        weights = _commute_weights(base, self.center, self.hours,
                                   rng.uniform(0.0, 3.0))
        return weights * total * fractions[self.hours][:, None, None]


class GravityGenerator(ODGenerator):
    """Real OSM land use + a real travel-time matrix, distributed by Furness.

    The morning matrix runs home -> work; the evening transposes it.
    """

    def __init__(self, taz: TazSet, weights: LandUseWeights, cost_minutes: np.ndarray,
                 *, daily_trips: float = 60000, beta: float = 0.12,
                 noise: float = 0.08, hours=None, seed: int = 42):
        super().__init__(taz, daily_trips=daily_trips, hours=hours, seed=seed)
        self.noise = noise
        self.beta = beta
        self.cost = np.asarray(cost_minutes, float)
        self.shape = DayShape()
        self.base = furness(weights.productions, weights.attractions,
                            deterrence(self.cost, beta))

    @property
    def mean_trip_minutes(self) -> float:
        return float((self.base * self.cost).sum())

    def expected(self, rng: np.random.Generator) -> np.ndarray:
        day_factor = float(np.clip(rng.normal(1.0, self.noise), 0.75, 1.25))
        outbound = OUTBOUND[self.hours][:, None, None]
        weights = outbound * self.base + (1 - outbound) * self.base.T
        return (weights * self.daily_trips * day_factor
                * self.shape.fractions[self.hours][:, None, None])


def _commute_weights(base: np.ndarray, center: int, hours: np.ndarray,
                     strength: float) -> np.ndarray:
    """Tilt a base matrix toward the centre in the morning, away in the evening."""
    weights = np.repeat(base[None], len(hours), axis=0)
    weights[:, :, center] *= 1 + strength * MORNING[hours][:, None]
    weights[:, center, :] *= 1 + strength * EVENING[hours][:, None]
    return weights / weights.sum(axis=(1, 2), keepdims=True)
