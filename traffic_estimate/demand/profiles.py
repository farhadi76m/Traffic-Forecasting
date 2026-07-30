"""The 24-hour demand shape. One definition, used by every OD generator."""
from __future__ import annotations

import numpy as np

HOURS = 24

# relative demand weight per hour: night trough, 07-08 commute peak,
# 13:00 midday bump, 17-19 evening peak, wind-down
HOUR_PROFILE = np.array([
    0.5, 0.3, 0.25, 0.25, 0.4, 1.0,
    2.5, 4.5, 4.2, 2.8, 2.2, 2.3,
    2.7, 3.1, 2.7, 2.8, 3.5, 4.4,
    4.6, 3.5, 2.5, 1.8, 1.2, 0.8,
])

# how strongly each hour pulls trips toward / away from the central zone
MORNING = np.zeros(HOURS)
MORNING[[6, 7, 8, 9]] = [0.5, 1.0, 0.9, 0.4]
EVENING = np.zeros(HOURS)
EVENING[[13, 16, 17, 18, 19]] = [0.3, 0.5, 0.9, 1.0, 0.5]

# share of an hour's trips running home -> work; the rest run the other way
OUTBOUND = np.array([
    .5, .5, .5, .5, .6, .75,
    .85, .9, .88, .8, .6, .5,
    .5, .45, .45, .4, .3, .2,
    .15, .2, .3, .4, .45, .5,
])


class DayShape:
    """Hourly demand weights, optionally jittered to make each day differ."""

    def __init__(self, weights: np.ndarray = HOUR_PROFILE):
        self.weights = np.asarray(weights, float)

    @property
    def fractions(self) -> np.ndarray:
        """Share of the daily total falling in each hour; sums to 1."""
        return self.weights / self.weights.sum()

    def jittered(self, rng: np.random.Generator, sigma: float) -> "DayShape":
        return DayShape(self.weights * np.exp(rng.normal(0, sigma, HOURS)))

    def share_of(self, hours) -> float:
        return float(self.fractions[list(hours)].sum())


def parse_hours(spec: str | None) -> np.ndarray:
    """'7,8' -> [7 8]; None -> the whole day."""
    if not spec:
        return np.arange(HOURS)
    return np.array(sorted({int(h) for h in spec.split(",")}), dtype=int)
