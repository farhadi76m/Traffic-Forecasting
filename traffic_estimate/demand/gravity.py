"""Doubly-constrained gravity distribution."""
from __future__ import annotations

import numpy as np


def deterrence(cost_minutes: np.ndarray, beta: float) -> np.ndarray:
    """exp(-beta * cost). Higher beta keeps more trips local."""
    return np.exp(-beta * np.asarray(cost_minutes, float))


def furness(productions: np.ndarray, attractions: np.ndarray,
            impedance: np.ndarray, iterations: int = 50,
            tol: float = 1e-4) -> np.ndarray:
    """Scale rows and columns until both margins match; returns a unit-sum matrix.

        trips[i,j] = a_i * P_i * b_j * A_j * impedance[i,j]
    """
    productions = np.asarray(productions, float)
    attractions = np.asarray(attractions, float)
    attractions = attractions * (productions.sum() / attractions.sum())
    a = np.ones_like(productions)
    b = np.ones_like(attractions)
    trips = np.outer(productions, attractions) * impedance
    for _ in range(iterations):
        a = productions / np.maximum((impedance * (b * attractions)).sum(1), 1e-9)
        b = attractions / np.maximum((impedance.T * (a * productions)).sum(1), 1e-9)
        trips = np.outer(a * productions, b * attractions) * impedance
        if np.abs(trips.sum(1) - productions).max() < tol * productions.sum():
            break
    return trips / trips.sum()
