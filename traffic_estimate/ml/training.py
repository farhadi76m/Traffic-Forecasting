"""Training loops. Both trainers share the early-stopping bookkeeping."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from ..network import STATIC_FEATURES
from .datasets import ForecastDataset, SurrogateDataset
from .metrics import ABS_TOL, REL_TOL, format_metrics, r2, regression_metrics
from .models import FORECAST_MODELS, HOURS, HistoricalTable, ODSurrogate

EVAL_CHUNK = 65536


def best_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class EarlyStopping:
    """Keeps the best weights seen and says when to give up."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.best_state: dict | None = None
        self.strikes = 0

    def update(self, loss: float, model: nn.Module) -> bool:
        if loss < self.best - self.min_delta:
            self.best, self.strikes = loss, 0
            self.best_state = {k: v.detach().clone()
                               for k, v in model.state_dict().items()}
            return False
        self.strikes += 1
        return self.strikes >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class ForecastTrainer:
    """Trains an (edge, hour) -> vehicles/hour model on held-out days.

    Counts are counts, so the loss is Poisson NLL on a log rate.
    """

    def __init__(self, dataset: ForecastDataset, *, model: str = "mlp",
                 hidden: int = 128, emb_dim: int = 32, lr: float = 2e-3,
                 epochs: int = 40, batch_size: int = 8192, n_test: int = 4,
                 n_val: int = 2, abs_tol: float = ABS_TOL,
                 rel_tol: float = REL_TOL, seed: int = 0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.dataset = dataset
        self.name = model
        self.hidden, self.emb_dim = hidden, emb_dim
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.abs_tol, self.rel_tol = abs_tol, rel_tol
        self.device = best_device()

        self.edges = dataset.edges
        position = {edge: i for i, edge in enumerate(self.edges)}
        self.edge_index = dataset.frame["edge"].map(position).to_numpy(np.int64)
        self.hour = dataset.frame["hour"].to_numpy(np.int64)
        self.y = dataset.frame["count"].to_numpy(np.float32)
        self.partition = dataset.split(n_test, n_val)

        static = dataset.static[list(STATIC_FEATURES)].to_numpy(np.float32)
        self.static_mean, self.static_std = static.mean(0), static.std(0) + 1e-6
        self.static = (static - self.static_mean) / self.static_std
        self.model: nn.Module | None = None
        self.table: HistoricalTable | None = None

    # --- training ----------------------------------------------------------

    def run(self) -> tuple[np.ndarray, dict]:
        predictions = (self._fit_table() if self.name == "table"
                       else self._fit_network())
        return predictions, self.evaluate(predictions)

    def _fit_table(self) -> np.ndarray:
        train = self.partition == 0
        self.table = HistoricalTable.fit(len(self.edges), self.edge_index[train],
                                         self.hour[train], self.y[train])
        return self.table.table[self.edge_index, self.hour]

    def _fit_network(self) -> np.ndarray:
        device = self.device
        tensor = lambda a: torch.tensor(a, device=device)
        static, edge, hour, y = (tensor(self.static), tensor(self.edge_index),
                                 tensor(self.hour), tensor(self.y))

        self.model = FORECAST_MODELS[self.name](
            len(self.edges), len(STATIC_FEATURES), self.emb_dim, self.hidden
        ).to(device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, factor=0.5, patience=2)
        stopper = EarlyStopping()

        train_rows = np.flatnonzero(self.partition == 0)
        val_rows = tensor(np.flatnonzero(self.partition == 1))
        loss_of = lambda rows, **kw: nn.functional.poisson_nll_loss(
            self.model(edge[rows], hour[rows], static[edge[rows]]), y[rows],
            log_input=True, **kw)

        for epoch in range(self.epochs):
            self.model.train()
            order = tensor(np.random.permutation(train_rows))
            total = 0.0
            for batch in order.split(self.batch_size):
                optimiser.zero_grad()
                loss = loss_of(batch)
                loss.backward()
                optimiser.step()
                total += loss.item() * len(batch)

            self.model.eval()
            with torch.no_grad():
                validation = sum(loss_of(b, reduction="sum").item()
                                 for b in val_rows.split(EVAL_CHUNK)) / len(val_rows)
            schedule.step(validation)
            print(f"epoch {epoch:02d}  train_loss {total / len(train_rows):.4f}  "
                  f"val_loss {validation:.4f}  "
                  f"lr {optimiser.param_groups[0]['lr']:.2e}")
            if stopper.update(validation, self.model):
                print("early stop")
                break

        stopper.restore(self.model)
        self.model.eval()
        with torch.no_grad():
            rows = torch.arange(len(self.y), device=device)
            return np.concatenate([
                torch.exp(self.model(edge[b], hour[b], static[edge[b]])).cpu().numpy()
                for b in rows.split(EVAL_CHUNK)])

    # --- evaluation --------------------------------------------------------

    def evaluate(self, predictions: np.ndarray) -> dict:
        results = {}
        for name, flag in (("train", 0), ("val", 1), ("test", 2)):
            mask = self.partition == flag
            if mask.any():
                results[name] = regression_metrics(
                    self.y[mask], predictions[mask], self.abs_tol, self.rel_tol)
                print(format_metrics(name, results[name]))
        results["test_typical"] = self._typical_day(predictions)
        print(format_metrics("test (typical day)", results["test_typical"]))
        return results

    def _typical_day(self, predictions: np.ndarray) -> dict:
        """Single days are Poisson-noisy; a typical-traffic forecast estimates
        the average of the held-out days, so score that too."""
        test = np.flatnonzero(self.partition == 2)
        key = self.edge_index[test] * HOURS + self.hour[test]
        size = len(self.edges) * HOURS
        observed = np.bincount(key, weights=self.y[test], minlength=size)
        predicted = np.bincount(key, weights=predictions[test], minlength=size)
        seen = np.bincount(key, minlength=size) > 0
        counts = np.bincount(key, minlength=size)[seen]
        return regression_metrics(observed[seen] / counts, predicted[seen] / counts,
                                  self.abs_tol, self.rel_tol)

    # --- persistence -------------------------------------------------------

    def save(self, out_dir: str, results: dict) -> str:
        os.makedirs(out_dir, exist_ok=True)
        meta = {"model": self.name, "edges": self.edges,
                "stat_cols": list(STATIC_FEATURES),
                "stat_mean": self.static_mean.tolist(),
                "stat_std": self.static_std.tolist(),
                "emb_dim": self.emb_dim, "hidden": self.hidden,
                "abs_tol": self.abs_tol, "rel_tol": self.rel_tol,
                "metrics": results}
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        if self.table is not None:
            np.save(os.path.join(out_dir, "table.npy"), self.table.table)
        else:
            torch.save({"state_dict": self.model.state_dict(),
                        "static": self.static},
                       os.path.join(out_dir, "model.pt"))
        print(f"saved model to {out_dir}")
        return out_dir


class SurrogateTrainer:
    """Trains OD -> (counts, travel time), with the right likelihood for each.

    The number to read is the gap over the OD-blind baseline. Any model scores
    a high R2 here just by learning the average rush hour; only the gap shows
    that the surrogate actually learned to read the OD.
    """

    def __init__(self, dataset: SurrogateDataset, *, hidden: int = 128,
                 emb_dim: int = 32, lr: float = 2e-3, epochs: int = 300,
                 scen_batch: int = 8, edge_batch: int = 768,
                 tt_weight: float = 1.0, test_frac: float = 0.15,
                 val_frac: float = 0.15, seed: int = 0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.data = dataset
        self.epochs, self.lr = epochs, lr
        self.scen_batch, self.edge_batch = scen_batch, edge_batch
        self.tt_weight = tt_weight
        self.emb_dim, self.hidden = emb_dim, hidden
        self.device = best_device()

        self.tt_ratio = np.log(np.maximum(
            dataset.congestion_ratio, 1e-3)).astype(np.float32)

        order = self.rng.permutation(dataset.n_scenarios)
        n_test = int(test_frac * dataset.n_scenarios)
        n_val = int(val_frac * dataset.n_scenarios)
        self.test_s = order[:n_test]
        self.val_s = order[n_test:n_test + n_val]
        self.train_s = order[n_test + n_val:]

        # the OD spans orders of magnitude, so log1p then standardise
        od_log = np.log1p(dataset.od).astype(np.float32)
        self.od_mu = od_log[self.train_s].mean((0, 1))
        self.od_sd = od_log[self.train_s].std((0, 1)) + 1e-6
        static = dataset.static
        self.static_norm = ((static - static.mean(0))
                            / (static.std(0) + 1e-6)).astype(np.float32)

        tensor = lambda a: torch.tensor(a, device=self.device)
        self.od_t = tensor((od_log - self.od_mu) / self.od_sd)
        self.counts_t = tensor(dataset.counts)
        self.tt_t = tensor(self.tt_ratio)
        self.static_t = tensor(self.static_norm)
        self.all_edges = torch.arange(dataset.n_edges, device=self.device)

        self.model = ODSurrogate(dataset.n_od, dataset.n_edges,
                                 static.shape[1], emb_dim, hidden).to(self.device)
        print(f"{dataset.n_scenarios} scenarios ({len(self.train_s)} train / "
              f"{len(self.val_s)} val / {len(self.test_s)} test), "
              f"{dataset.n_edges} edges, OD dim {dataset.n_od}")

    def predict(self, scenarios, edge_chunk: int = 1024):
        """Full (S, edges, 24) prediction, chunked over edges to fit the GPU."""
        self.model.eval()
        counts, travel = [], []
        with torch.no_grad():
            for start in range(0, self.data.n_edges, edge_chunk):
                batch = self.all_edges[start:start + edge_chunk]
                log_c, log_t = self.model(self.od_t[scenarios], batch,
                                          self.static_t[batch])
                counts.append(torch.exp(log_c))
                travel.append(log_t)
        return torch.cat(counts, 1), torch.cat(travel, 1)

    def fit(self) -> None:
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, factor=0.5, patience=10)
        stopper = EarlyStopping()
        steps = max(1, len(self.train_s) // self.scen_batch)
        tensor = lambda a: torch.tensor(a, device=self.device)

        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            for _ in range(steps):
                scenarios = tensor(self.rng.choice(self.train_s, self.scen_batch,
                                                   replace=False))
                edges = self.all_edges[tensor(
                    self.rng.choice(self.data.n_edges, self.edge_batch,
                                    replace=False))]
                optimiser.zero_grad()
                log_c, log_t = self.model(self.od_t[scenarios], edges,
                                          self.static_t[edges])
                loss = (nn.functional.poisson_nll_loss(
                            log_c, self.counts_t[scenarios][:, edges],
                            log_input=True)
                        + self.tt_weight * nn.functional.mse_loss(
                            log_t, self.tt_t[scenarios][:, edges]))
                loss.backward()
                optimiser.step()
                total += float(loss)

            if epoch % 5 and epoch != self.epochs - 1:
                continue
            counts, travel = self.predict(tensor(self.val_s))
            score = -r2(self.counts_t[self.val_s].cpu().numpy(),
                        counts.cpu().numpy()) \
                - r2(self.tt_t[self.val_s].cpu().numpy(), travel.cpu().numpy())
            schedule.step(score)
            print(f"ep {epoch:3d}  loss {total / steps:8.4f}   "
                  f"val -R2 sum {score:7.3f}   "
                  f"lr {optimiser.param_groups[0]['lr']:.1e}")
            if stopper.update(score, self.model):
                print("early stop")
                break
        stopper.restore(self.model)

    def evaluate(self) -> dict:
        counts, travel = self.predict(torch.tensor(self.test_s,
                                                   device=self.device))
        counts, travel = counts.cpu().numpy(), travel.cpu().numpy()
        true_counts = self.data.counts[self.test_s]
        true_tt = self.tt_ratio[self.test_s]
        blind_counts = np.broadcast_to(self.data.counts[self.train_s].mean(0),
                                       true_counts.shape)
        blind_tt = np.broadcast_to(self.tt_ratio[self.train_s].mean(0),
                                   true_tt.shape)
        free_flow = self.data.free_flow[None, :, None]

        results = {
            "counts": {"surrogate_r2": r2(true_counts, counts),
                       "blind_r2": r2(true_counts, blind_counts),
                       "surrogate_mae": float(np.abs(true_counts - counts).mean()),
                       "blind_mae": float(np.abs(true_counts - blind_counts).mean())},
            "traveltime_logratio": {
                "surrogate_r2": r2(true_tt, travel),
                "blind_r2": r2(true_tt, blind_tt),
                "surrogate_mae": float(np.abs(true_tt - travel).mean()),
                "blind_mae": float(np.abs(true_tt - blind_tt).mean())},
            "traveltime_seconds": {
                "surrogate_r2": r2(self.data.traveltime[self.test_s],
                                   np.exp(travel) * free_flow),
                "blind_r2": r2(self.data.traveltime[self.test_s],
                               np.exp(blind_tt) * free_flow),
                "surrogate_mae": float(np.abs(
                    self.data.traveltime[self.test_s]
                    - np.exp(travel) * free_flow).mean())},
        }

        print("\n--- test set (unseen OD matrices) ---")
        for name, values in results.items():
            line = f"  {name:22s} R2 {values['surrogate_r2']:7.4f}"
            if "blind_r2" in values:
                gap = values["surrogate_r2"] - values["blind_r2"]
                line += (f"   OD-blind baseline R2 {values['blind_r2']:7.4f}"
                         f"   (gap {gap:+.4f})")
            print(line)
        print("\n  The gap is what reading the OD buys. If it is ~0, the model is")
        print("  ignoring the OD and just predicting the average day.")
        return results

    def save(self, out_dir: str, results: dict) -> str:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(),
                    "od_mu": self.od_mu, "od_sd": self.od_sd,
                    "static_norm": self.static_norm,
                    "free_flow": self.data.free_flow,
                    "n_od": self.data.n_od, "n_edges": self.data.n_edges,
                    "n_static": self.data.static.shape[1],
                    "emb_dim": self.emb_dim, "hidden": self.hidden,
                    "test_s": self.test_s, "train_s": self.train_s},
                   os.path.join(out_dir, "surrogate.pt"))
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved -> {out_dir}")
        return out_dir
