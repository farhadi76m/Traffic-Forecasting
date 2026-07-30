"""Scoring shared by the forecaster, the surrogate and the inverse problem."""
from __future__ import annotations

import numpy as np

ABS_TOL = 3.0
REL_TOL = 0.2


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    residual = ((y_true - y_pred) ** 2).sum()
    total = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - residual / total)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       abs_tol: float = ABS_TOL,
                       rel_tol: float = REL_TOL) -> dict[str, float]:
    """A prediction counts as accurate within rel_tol, or abs_tol on quiet edges."""
    error = np.abs(np.asarray(y_pred, float) - y_true)
    within = error <= np.maximum(abs_tol, rel_tol * y_true)
    return {"accuracy": float(within.mean()),
            "r2": r2(y_true, y_pred),
            "mae": float(error.mean()),
            "rmse": float(np.sqrt((error ** 2).mean()))}


def format_metrics(name: str, values: dict[str, float]) -> str:
    return (f"{name:18s} accuracy {values['accuracy'] * 100:6.2f}%  "
            f"R2 {values['r2']:.4f}  MAE {values['mae']:.2f}  "
            f"RMSE {values['rmse']:.2f}")
