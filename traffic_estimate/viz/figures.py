"""Figures for the forecasting model."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm

from ..ml.predictor import Predictor
from ..network import RoadNetwork
from . import theme
from .theme import AQUA, BASELINE, BLUE, CMAP, GRID, INK, INK2

QUIET = 0.5  # vehicles/hour below which an edge is drawn as background


class ForecastFigures:
    """The four figures the README shows, all from one model + dataset."""

    def __init__(self, network: RoadNetwork, data, predictor: Predictor,
                 out_dir: str, hours=(7, 13, 18, 23), top_edges: int = 6):
        self.network = network
        self.data = data
        self.predictor = predictor
        self.out_dir = out_dir
        self.hours = list(hours)
        self.top_edges = top_edges
        self.rates = predictor.all_rates()
        self.position = {edge: i for i, edge in enumerate(predictor.edges)}
        os.makedirs(out_dir, exist_ok=True)

    def _save(self, fig, name: str) -> str:
        path = os.path.join(self.out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def render_all(self) -> list[str]:
        paths = [self.network_traffic(), self.edge_profiles(),
                 self.pred_vs_actual(), self.daily_profile()]
        print(f"wrote {len(paths)} figures to {self.out_dir}")
        return paths

    def network_traffic(self) -> str:
        shapes = [self.network.edge(e).getShape() for e in self.predictor.edges]
        norm = PowerNorm(gamma=0.5, vmin=0,
                         vmax=max(np.percentile(self.rates[:, self.hours], 99.5), 1))
        # frame the region that actually carries traffic, not the full bbox
        active = np.flatnonzero(self.rates.max(1) >= QUIET)
        points = np.array([p for i in active for p in shapes[i]])
        (x0, y0), (x1, y1) = points.min(0), points.max(0)
        mx, my = 0.03 * (x1 - x0), 0.03 * (y1 - y0)
        aspect = (y1 - y0 + 2 * my) / (x1 - x0 + 2 * mx)

        ncol = 2
        nrow = int(np.ceil(len(self.hours) / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(11, 5.2 * aspect * nrow + 1.5))
        for ax, hour in zip(np.ravel(axes), self.hours):
            values = self.rates[:, hour]
            quiet = values < QUIET
            ax.add_collection(LineCollection(
                [shapes[i] for i in np.flatnonzero(quiet)],
                colors=GRID, linewidths=0.5))
            busy = np.flatnonzero(~quiet)
            busy = busy[np.argsort(values[busy])]  # heaviest drawn last, on top
            ax.add_collection(LineCollection(
                [shapes[i] for i in busy], colors=CMAP(norm(values[busy])),
                linewidths=0.6 + 1.9 * norm(values[busy])))
            ax.set_xlim(x0 - mx, x1 + mx)
            ax.set_ylim(y0 - my, y1 + my)
            ax.set_aspect("equal")
            theme.bare(ax)
            theme.title(ax, f"{hour:02d}:00", size=10)
        for ax in np.ravel(axes)[len(self.hours):]:
            ax.set_visible(False)

        fig.suptitle("Estimated traffic across the network", x=0.06, y=0.99,
                     ha="left", fontsize=13, fontweight="bold", color=INK)
        bar = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm),
                           ax=np.ravel(axes).tolist(), orientation="horizontal",
                           fraction=0.03, pad=0.03, aspect=45)
        bar.set_label("vehicles / hour", color=INK2)
        bar.outline.set_visible(False)
        return self._save(fig, "network_traffic.png")

    def edge_profiles(self) -> str:
        busiest = (self.data.groupby("edge")["count"].sum()
                   .sort_values(ascending=False).head(self.top_edges).index.tolist())
        ncol, nrow = 3, int(np.ceil(self.top_edges / 3))
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.1 * nrow),
                                 sharex=True, sharey=True)
        hours = np.arange(24)
        for n, (ax, edge) in enumerate(zip(np.ravel(axes), busiest)):
            days = (self.data[self.data["edge"] == edge]
                    .pivot(index="scenario", columns="hour", values="count")
                    .to_numpy())
            low, median, high = np.percentile(days, [10, 50, 90], axis=0)
            ax.fill_between(hours, low, high, color=AQUA, alpha=0.16, linewidth=0)
            ax.plot(hours, median, color=AQUA, lw=2,
                    label="simulated (median, 10-90%)")
            ax.plot(hours, self.rates[self.position[edge]], color=BLUE, lw=2,
                    label="predicted")
            ax.set_title(edge, loc="left", fontsize=9, color=INK2)
            ax.set_xticks([0, 6, 12, 18, 23])
            if n == 0:
                ax.text(0.03, 0.9, "predicted", color=BLUE, fontsize=9,
                        fontweight="bold", transform=ax.transAxes)
                ax.text(0.03, 0.78, "simulated", color=AQUA, fontsize=9,
                        fontweight="bold", transform=ax.transAxes)
        handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False,
                   labelcolor=INK2, bbox_to_anchor=(0.99, 1.0))
        fig.suptitle("Busiest edges: 24h traffic profile", x=0.06, y=1.0,
                     ha="left", fontsize=13, fontweight="bold", color=INK)
        fig.supxlabel("hour of day", color=INK2, fontsize=10)
        fig.supylabel("vehicles / hour", color=INK2, fontsize=10)
        fig.tight_layout()
        return self._save(fig, "edge_profiles.png")

    def pred_vs_actual(self) -> str:
        typical = self.data.groupby(["edge", "hour"])["count"].mean().reset_index()
        observed = typical["count"].to_numpy()
        predicted = self.rates[typical["edge"].map(self.position).to_numpy(),
                               typical["hour"].to_numpy()]
        if len(observed) > 25000:
            sample = np.random.default_rng(0).choice(len(observed), 25000,
                                                     replace=False)
            observed, predicted = observed[sample], predicted[sample]

        fig, ax = plt.subplots(figsize=(6.5, 6.2))
        limit = max(predicted.max(), observed.max()) * 1.2 + 1
        ax.plot([1, limit], [1, limit], ls="--", lw=1, color=BASELINE, zorder=1)
        ax.scatter(predicted + 1, observed + 1, s=6, color=BLUE, alpha=0.2,
                   linewidths=0, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.9, limit)
        ax.set_ylim(0.9, limit)
        ax.set_xlabel("predicted vehicles/hour (+1, log)")
        ax.set_ylabel("simulated typical day (+1, log)")
        theme.title(ax, "Predicted vs typical-day traffic")
        scores = self.predictor.metrics.get("test_typical")
        if scores:
            ax.text(0.03, 0.97,
                    f"held-out days:  accuracy {scores['accuracy'] * 100:.1f}%   "
                    f"R² {scores['r2']:.3f}   MAE {scores['mae']:.2f}",
                    transform=ax.transAxes, va="top", color=INK2, fontsize=9)
        fig.tight_layout()
        return self._save(fig, "pred_vs_actual.png")

    def daily_profile(self) -> str:
        totals = (self.data.groupby(["scenario", "hour"])["count"].sum()
                  .unstack().to_numpy())
        hours = np.arange(24)
        low, high = np.percentile(totals, [10, 90], axis=0)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.fill_between(hours, low, high, color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(hours, totals.mean(0), color=BLUE, lw=2)
        ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 23])
        ax.set_xlabel("hour of day")
        ax.set_ylabel("vehicle entries / hour (all edges)")
        theme.title(ax, "Network-wide daily traffic (mean, 10-90% over days)")
        ax.set_ylim(0, None)
        fig.tight_layout()
        return self._save(fig, "daily_profile.png")
