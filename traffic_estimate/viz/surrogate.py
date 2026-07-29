"""Figures for the surrogate and the inverse experiment."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..ml.datasets import SurrogateDataset
from ..ml.models import ODSurrogate
from .theme import BASELINE, CONGESTED_REGIME, LIGHT_REGIME

REGIMES = {"light": (LIGHT_REGIME, "light (~12k trips/day)"),
           "cong": (CONGESTED_REGIME, "congested (~100k trips/day)")}


class SurrogateFigures:
    """Reads a `--root` holding one folder per demand regime."""

    def __init__(self, root: str, out_dir: str):
        self.root = root
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def dataset(self, regime: str) -> SurrogateDataset:
        return SurrogateDataset.load(self.path(regime, "dataset", "data.npz"))

    def _save(self, fig, name: str) -> str:
        out = os.path.join(self.out_dir, name)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(out)
        return out

    def render_all(self) -> None:
        for figure in (self.congestion, self.surrogate_fit,
                       self.identifiability, self.recovery):
            try:
                figure()
            except Exception as exc:  # a missing stage must not kill the rest
                print(f"skip {figure.__name__}: {exc}")

    def congestion(self) -> str:
        """Why the inverse problem is solvable in one regime and not the other."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for regime, (colour, label) in REGIMES.items():
            data = self.dataset(regime)
            ratio = data.congestion_ratio
            peak = ratio[:, :, 7:10].ravel()
            axes[0].hist(np.clip(peak, 1, 4), bins=80, alpha=0.6, color=colour,
                         label=f"{label}\nmean {peak.mean():.2f}x",
                         density=True, log=True)
            axes[1].scatter(data.od.sum((1, 2)), ratio[:, :, 7:10].mean((1, 2)),
                            s=12, alpha=0.5, color=colour, label=label)
        axes[0].set_xlabel("travel time / free-flow time (peak 07-09)")
        axes[0].set_ylabel("density (log)")
        axes[0].set_title("Congestion: is travel time even moving?")
        axes[1].set_xlabel("total trips in the OD matrix")
        axes[1].set_ylabel("mean travel time / free-flow, peak")
        axes[1].set_xscale("log")
        axes[1].set_title("Sensitivity of travel time to demand")
        for ax in axes:
            ax.legend(fontsize=8)
        return self._save(fig, "congestion.png")

    def surrogate_fit(self) -> str:
        """Surrogate vs SUMO on held-out OD, against the OD-blind baseline."""
        regimes = [r for r in REGIMES
                   if os.path.exists(self.path(r, "model", "metrics.json"))]
        fig, axes = plt.subplots(1, len(regimes) * 2,
                                 figsize=(5.2 * len(regimes), 4.2))
        axes = np.atleast_1d(axes)
        panel = 0
        for regime in regimes:
            with open(self.path(regime, "model", "metrics.json")) as handle:
                scores = json.load(handle)
            data = self.dataset(regime)
            predicted_counts, predicted_tt = self._predict(regime, data)
            test = self._test_ids(regime)
            blind = np.broadcast_to(data.traveltime[self._train_ids(regime)].mean(0),
                                    data.traveltime[test].shape)

            panels = [(data.counts[test], predicted_counts, None,
                       "vehicle counts (veh/h)", scores["counts"]["surrogate_r2"]),
                      (data.traveltime[test], predicted_tt, blind,
                       "travel time (s)",
                       scores["traveltime_seconds"]["surrogate_r2"])]
            for true, predicted, baseline, label, r2 in panels:
                ax = axes[panel]
                panel += 1
                sample = np.random.default_rng(0).choice(true.size, 4000,
                                                         replace=False)
                x, y = true.ravel()[sample], predicted.ravel()[sample]
                if baseline is not None:
                    ax.scatter(x, baseline.ravel()[sample], s=5, alpha=0.25,
                               color=BASELINE,
                               label=("OD-blind baseline (R2 "
                                      f"{scores['traveltime_seconds']['blind_r2']:.2f})"))
                ax.scatter(x, y, s=5, alpha=0.4, color=REGIMES[regime][0],
                           label=f"surrogate (R2 {r2:.2f})")
                limit = np.percentile(x, 99.5)
                ax.plot([0, limit], [0, limit], "k--", lw=1)
                ax.set_xlim(0, limit)
                ax.set_ylim(0, limit)
                ax.set_xlabel(f"SUMO {label}")
                ax.set_ylabel("surrogate")
                ax.set_title(f"{regime} regime: {label.split(' (')[0]}", fontsize=10)
                ax.legend(fontsize=7, loc="upper left")
        return self._save(fig, "surrogate.png")

    def identifiability(self) -> str:
        """Spectrum of J'J: how many OD directions the observations constrain."""
        styles = {("light", "traveltime"): (LIGHT_REGIME, "-", "light / travel time"),
                  ("cong", "traveltime"): (CONGESTED_REGIME, "-", "congested / travel time"),
                  ("cong", "counts"): (CONGESTED_REGIME, "--", "congested / counts"),
                  ("light", "counts"): (LIGHT_REGIME, "--", "light / counts")}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        unknowns = None
        for (regime, observed), (colour, style, label) in styles.items():
            path = self.path(regime, "model", "inverse", f"spectrum_{observed}.npy")
            if not os.path.exists(path):
                continue
            eigenvalues = np.load(path)
            unknowns = len(eigenvalues)
            ax.semilogy(np.arange(1, unknowns + 1), np.maximum(eigenvalues, 1e-12),
                        color=colour, ls=style, lw=1.8, label=label)
        ax.axhline(1.0, color="k", lw=1)
        ax.text(0.98, 1.0, " data = prior ", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8)
        ax.set_xlabel(f"OD direction (sorted), of {unknowns} unknowns")
        ax.set_ylabel("eigenvalue of J'J  (data / prior information)")
        ax.set_title("Identifiability: how many OD directions the sensors "
                     "constrain\nabove the line the data wins; below it the "
                     "posterior is just the prior", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(1e-8, None)
        return self._save(fig, "identifiability.png")

    def recovery(self) -> str | None:
        """True vs recovered OD, with posterior error bars."""
        cases = [(r, o) for r in REGIMES for o in ("traveltime", "counts")
                 if os.path.exists(self.path(r, "model", "inverse",
                                             f"recovery_{o}.npz"))]
        if not cases:
            return None
        fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.3))
        for ax, (regime, observed) in zip(np.atleast_1d(axes), cases):
            blob = np.load(self.path(regime, "model", "inverse",
                                     f"recovery_{observed}.npz"))
            true = blob["true"].ravel()
            recovered = blob["map"].ravel()
            ax.errorbar(true, recovered, yerr=blob["sd"].ravel(), fmt="o", ms=3,
                        alpha=0.35, color=REGIMES[regime][0],
                        ecolor=REGIMES[regime][0], elinewidth=0.5, capsize=0)
            limit = max(true.max(), recovered.max()) * 1.05
            ax.plot([0, limit], [0, limit], "k--", lw=1)
            r2 = 1 - ((true - recovered) ** 2).sum() / ((true - true.mean()) ** 2).sum()
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.set_xlabel("true OD flow (trips/h)")
            ax.set_ylabel("recovered (posterior mean +/- sd)")
            ax.set_title(f"{regime} / {observed}   R2 {r2:.3f}", fontsize=10)
        return self._save(fig, "recovery.png")

    # --- helpers -----------------------------------------------------------

    def _checkpoint(self, regime: str) -> dict:
        return torch.load(self.path(regime, "model", "surrogate.pt"),
                          map_location="cpu", weights_only=False)

    def _test_ids(self, regime: str) -> np.ndarray:
        return self._checkpoint(regime)["test_s"][:6]

    def _train_ids(self, regime: str) -> np.ndarray:
        return self._checkpoint(regime)["train_s"]

    def _predict(self, regime: str, data: SurrogateDataset):
        checkpoint = self._checkpoint(regime)
        model = ODSurrogate.from_checkpoint(checkpoint)
        test = checkpoint["test_s"][:6]
        normalised = ((np.log1p(data.od[test]) - checkpoint["od_mu"])
                      / checkpoint["od_sd"]).astype(np.float32)
        with torch.no_grad():
            log_counts, log_tt = model(torch.tensor(normalised),
                                       torch.arange(checkpoint["n_edges"]),
                                       torch.tensor(checkpoint["static_norm"]))
        return (torch.exp(log_counts).numpy(),
                np.exp(log_tt.numpy()) * data.free_flow[None, :, None])
