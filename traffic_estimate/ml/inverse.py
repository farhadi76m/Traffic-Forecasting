"""The inverse problem: observed traffic -> a posterior over the OD matrix.

MAP + Laplace rather than NUTS: with ~900 unknowns and a neural forward model,
NUTS needs tens of thousands of surrogate evaluations and mixes badly on the
ridges this problem has. The Gauss-Newton Hessian H = I + J'J gives the same
posterior to second order in one shot, and J -- d(observation)/d(OD) -- is the
classical assignment matrix. Its spectrum answers the question that matters:

    HOW MANY DIRECTIONS OF THE OD MATRIX DOES THE DATA ACTUALLY CONSTRAIN?

An eigenvalue far above 1 means the data pins that direction down much more
tightly than the prior does; near 0 means the posterior is just the prior.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .datasets import SurrogateDataset
from .models import ODSurrogate

JACOBIAN_CHUNK = 32


@dataclass
class InverseCase:
    scenario: int
    od_r2_posterior: float
    od_r2_prior_mean: float
    total_trips_true: float
    total_trips_map: float
    identifiable_directions: int
    eig_max: float
    eig_median: float
    posterior_rel_width: float
    spectrum: np.ndarray = field(repr=False, default=None)
    recovery: dict = field(repr=False, default=None)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if k not in ("spectrum", "recovery")}


class ODInversion:
    """Recovers held-out OD matrices from simulated sensor readings."""

    def __init__(self, checkpoint: str, dataset: SurrogateDataset, *,
                 observe: str = "traveltime", n_sensors: int = 300,
                 sigma_tt: float = 0.0, steps: int = 1500, lr: float = 5e-2,
                 seed: int = 0):
        # cuDNN's fused GRU refuses to backprop in eval mode, and we need
        # gradients w.r.t. the GRU's *input*
        torch.backends.cudnn.enabled = False
        torch.manual_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = dataset
        self.observe = observe
        self.steps, self.lr = steps, lr

        self.ckpt = torch.load(checkpoint, map_location=self.device,
                               weights_only=False)
        self.model = ODSurrogate.from_checkpoint(self.ckpt).to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.n_od = self.ckpt["n_od"]
        self.static = torch.tensor(self.ckpt["static_norm"], device=self.device)
        self.od_mu = torch.tensor(self.ckpt["od_mu"], device=self.device)
        self.od_sd = torch.tensor(self.ckpt["od_sd"], device=self.device)

        # a detector on an empty street measures nothing, so watch busy edges
        busy = dataset.counts.mean((0, 2))
        self.sensors = (np.argsort(-busy)[:n_sensors]
                        if n_sensors and n_sensors < dataset.n_edges
                        else np.arange(dataset.n_edges))
        self.sensor_t = torch.tensor(self.sensors, device=self.device)
        self.sensor_static = self.static[self.sensor_t]
        self.sigma_tt, self.alpha_counts = self._calibrate_likelihood(sigma_tt)
        print(f"surrogate held-out error -> sigma_tt {self.sigma_tt:.4f} "
              f"(log travel-time ratio), relative count error "
              f"{self.alpha_counts:.3f}")

    # --- likelihood --------------------------------------------------------

    def _calibrate_likelihood(self, sigma_tt: float) -> tuple[float, float]:
        """Never claim the sensors are more precise than the surrogate is accurate.

        With a hand-set sigma the posterior fits surrogate bias rather than
        traffic and the OD diverges (measured OD R2 = -169 in the light regime),
        so take sigma from the surrogate's own held-out residuals.
        """
        test = self.ckpt["test_s"]
        normalised = ((np.log1p(self.data.od[test]) - self.ckpt["od_mu"])
                      / self.ckpt["od_sd"]).astype(np.float32)
        with torch.no_grad():
            log_counts, log_tt = self.model(
                torch.tensor(normalised, device=self.device),
                self.sensor_t, self.sensor_static)

        observed_tt = self._log_ratio(self.data.traveltime[test][:, self.sensors])
        sigma = float(sigma_tt) or float((log_tt.cpu().numpy() - observed_tt).std())

        true_counts = self.data.counts[test][:, self.sensors]
        predicted = torch.exp(log_counts).cpu().numpy()
        busy = true_counts > 5
        alpha = (float(np.abs(predicted[busy] - true_counts[busy]).std()
                       / max(true_counts[busy].mean(), 1.0)) if busy.any() else 0.1)
        return sigma, alpha

    def _log_ratio(self, traveltime: np.ndarray) -> np.ndarray:
        free_flow = self.data.free_flow[self.sensors][:, None]
        return np.log(np.maximum(traveltime / free_flow, 1e-3))

    # --- inference ---------------------------------------------------------

    def _od_from_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Undo the training normalisation: latent -> trips."""
        return torch.clamp(torch.expm1(x * self.od_sd + self.od_mu), min=0.0)

    def _residual_fn(self, observed_tt, observed_counts, weights):
        """Standardised residual; stacking these makes 0.5*||r||^2 the NLL."""
        def residual(x):
            log_counts, log_tt = self.model(x[None], self.sensor_t,
                                            self.sensor_static, unfused=True)
            parts = []
            if self.observe in ("traveltime", "both"):
                parts.append(((log_tt[0] - observed_tt) / self.sigma_tt).reshape(-1))
            if self.observe in ("counts", "both"):
                parts.append(((torch.exp(log_counts[0]) - observed_counts)
                              / weights).reshape(-1))
            return torch.cat(parts)
        return residual

    def recover(self, scenario: int) -> InverseCase:
        tensor = lambda a: torch.tensor(a, device=self.device, dtype=torch.float32)
        observed_tt = tensor(self._log_ratio(
            self.data.traveltime[scenario][self.sensors]))
        observed_counts = tensor(self.data.counts[scenario][self.sensors])
        # Poisson counting noise widened by the surrogate's own error; weighting
        # by the *observed* count keeps the residual cleanly differentiable
        weights = torch.sqrt(observed_counts + 1.0
                             + (self.alpha_counts * observed_counts) ** 2)
        residual = self._residual_fn(observed_tt, observed_counts, weights)

        x = torch.zeros(24, self.n_od, device=self.device, requires_grad=True)
        optimiser = torch.optim.Adam([x], lr=self.lr)
        for _ in range(self.steps):
            optimiser.zero_grad()
            loss = 0.5 * (x ** 2).sum() + 0.5 * (residual(x) ** 2).sum()
            loss.backward()
            optimiser.step()
        x_map = x.detach()

        jacobian = self._jacobian(residual, x_map)
        gauss_newton = jacobian.T @ jacobian
        covariance = torch.linalg.inv(
            torch.eye(gauss_newton.shape[0], device=self.device) + gauss_newton)
        latent_sd = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0))
        eigenvalues = torch.linalg.eigvalsh(gauss_newton).clamp(min=0)
        eigenvalues = eigenvalues.cpu().numpy()[::-1]

        od_map = self._od_from_latent(x_map).cpu().numpy()
        od_high = self._od_from_latent(
            x_map + latent_sd.reshape(24, self.n_od)).cpu().numpy()
        od_prior = self._od_from_latent(torch.zeros_like(x_map)).cpu().numpy()
        od_true = self.data.od[scenario]

        def score(prediction):
            residual_ss = ((od_true - prediction) ** 2).sum()
            return float(1 - residual_ss / ((od_true - od_true.mean()) ** 2).sum())

        return InverseCase(
            scenario=int(scenario),
            od_r2_posterior=score(od_map),
            od_r2_prior_mean=score(od_prior),
            total_trips_true=float(od_true.sum()),
            total_trips_map=float(od_map.sum()),
            identifiable_directions=int((eigenvalues > 1.0).sum()),
            eig_max=float(eigenvalues[0]),
            eig_median=float(np.median(eigenvalues)),
            posterior_rel_width=float(np.mean((od_high - od_map) / (od_map + 1.0))),
            spectrum=eigenvalues,
            recovery={"true": od_true, "map": od_map, "sd": od_high - od_map})

    def _jacobian(self, residual, x_map: torch.Tensor) -> torch.Tensor:
        """Forward-mode, one chunk of input directions at a time.

        A single jacfwd over all 24 x n_od inputs allocates several GB.
        """
        size = 24 * self.n_od
        basis = torch.eye(size, device=self.device).reshape(size, 24, self.n_od)
        column = lambda v: torch.func.jvp(residual, (x_map,), (v,))[1]
        chunks = [torch.func.vmap(column)(basis[i:i + JACOBIAN_CHUNK])
                  for i in range(0, size, JACOBIAN_CHUNK)]
        return torch.cat(chunks).T

    # --- driver ------------------------------------------------------------

    def run(self, n_cases: int, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        report = {"observe": self.observe, "n_sensors": int(len(self.sensors)),
                  "n_unknowns": 24 * self.n_od, "sigma_tt": self.sigma_tt,
                  "alpha_counts": self.alpha_counts, "cases": []}

        for position, scenario in enumerate(self.ckpt["test_s"][:n_cases]):
            case = self.recover(scenario)
            report["cases"].append(case.as_dict())
            if position == 0:  # keep the first case's arrays for the figures
                np.save(os.path.join(out_dir, f"spectrum_{self.observe}.npy"),
                        case.spectrum)
                np.savez(os.path.join(out_dir, f"recovery_{self.observe}.npz"),
                         **case.recovery)
            print(f"case {position} (scenario {case.scenario:3d}): "
                  f"OD R2 {case.od_r2_posterior:6.3f} "
                  f"(prior mean alone: {case.od_r2_prior_mean:6.3f})   "
                  f"total trips {case.total_trips_map:8.0f} vs true "
                  f"{case.total_trips_true:8.0f}   "
                  f"identifiable {case.identifiable_directions:3d}/{24 * self.n_od}")

        posterior = [c["od_r2_posterior"] for c in report["cases"]]
        prior = [c["od_r2_prior_mean"] for c in report["cases"]]
        identifiable = [c["identifiable_directions"] for c in report["cases"]]
        report["summary"] = {
            "mean_od_r2_posterior": float(np.mean(posterior)),
            "mean_od_r2_prior_mean": float(np.mean(prior)),
            "mean_identifiable_directions": float(np.mean(identifiable)),
            "n_unknowns": 24 * self.n_od}

        print(f"\n--- {self.observe}, {len(self.sensors)} sensors ---")
        print(f"  OD recovered  R2 {np.mean(posterior):.3f}   "
              f"(returning the prior mean scores {np.mean(prior):.3f})")
        print(f"  data constrains {np.mean(identifiable):.0f} of "
              f"{24 * self.n_od} OD directions")

        path = os.path.join(out_dir, f"inverse_{self.observe}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"  -> {path}")
        return report
