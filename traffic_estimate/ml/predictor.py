"""Inference: a trained model directory answering (edge, hour) -> vehicles/hour."""
from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np

from .models import FORECAST_MODELS, HOURS, HistoricalTable


class Predictor:
    """Wraps whichever model was trained behind one hourly-rate interface."""

    def __init__(self, meta: dict, rates: Callable[[np.ndarray], np.ndarray]):
        self.meta = meta
        self._rates = rates

    @property
    def edges(self) -> list[str]:
        return self.meta["edges"]

    @property
    def model_name(self) -> str:
        return self.meta["model"]

    @property
    def metrics(self) -> dict:
        return self.meta.get("metrics", {})

    @classmethod
    def load(cls, model_dir: str) -> "Predictor":
        with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta["model"] == "table":
            table = HistoricalTable(np.load(os.path.join(model_dir, "table.npy")))
            return cls(meta, table.rates)

        import torch
        checkpoint = torch.load(os.path.join(model_dir, "model.pt"),
                                map_location="cpu", weights_only=False)
        model = FORECAST_MODELS[meta["model"]](
            len(meta["edges"]), len(meta["stat_cols"]), meta["emb_dim"],
            meta["hidden"]).load_checkpoint(checkpoint["state_dict"])
        static = torch.tensor(checkpoint["static"])

        def rates(indices: np.ndarray) -> np.ndarray:
            edge = torch.as_tensor(np.atleast_1d(indices), dtype=torch.long)
            with torch.no_grad():
                return torch.exp(model.profile(edge, static[edge])).numpy()

        predictor = cls(meta, rates)
        predictor.model = model
        return predictor

    def index_of(self, edge_id: str) -> int:
        try:
            return self.edges.index(edge_id)
        except ValueError:
            raise SystemExit(
                f"edge '{edge_id}' is not in the network the model was trained "
                f"on ({len(self.edges)} edges)") from None

    def rates(self, indices: np.ndarray) -> np.ndarray:
        """(n, 24) vehicles/hour for the given edge indices."""
        return self._rates(np.atleast_1d(indices))

    def all_rates(self) -> np.ndarray:
        return self.rates(np.arange(len(self.edges)))

    def profile(self, edge_id: str) -> np.ndarray:
        return self.rates(self.index_of(edge_id))[0]

    def at(self, edge_id: str, hour: int) -> float:
        return float(self.profile(edge_id)[hour])

    def format_profile(self, edge_id: str, width: int = 40) -> str:
        rates = self.profile(edge_id)
        peak = int(np.argmax(rates))
        lines = [f"edge {edge_id} - estimated traffic (vehicles/hour)"]
        for hour in range(HOURS):
            bar = "#" * int(round(rates[hour] / max(rates.max(), 1) * width))
            mark = "  <- peak" if hour == peak else ""
            lines.append(f"  {hour:02d}:00  {rates[hour]:7.1f}  {bar}{mark}")
        return "\n".join(lines)
