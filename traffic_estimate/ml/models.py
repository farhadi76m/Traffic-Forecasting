"""Neural models. Two families: an edge-profile forecaster and a SUMO surrogate."""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn

HOURS = 24
LOG_RATE_CAP = 12.0


def fourier_basis(harmonics: int = 6) -> torch.Tensor:
    """(24, 2k+1): bias plus k sin/cos harmonics over the day."""
    hour = torch.arange(HOURS, dtype=torch.float32)[:, None]
    frequency = torch.arange(1, harmonics + 1, dtype=torch.float32)[None, :]
    angle = hour * frequency * (2 * torch.pi / HOURS)
    return torch.cat([torch.ones(HOURS, 1), torch.sin(angle), torch.cos(angle)],
                     dim=1)


class Checkpointed(nn.Module):
    """A module that can still load checkpoints written before a rename.

    LEGACY_NAMES maps the old top-level attribute to the current one, so models
    trained by earlier versions keep working.
    """

    LEGACY_NAMES: dict[str, str] = {}

    def load_state_dict(self, state_dict, *args, **kwargs):
        upgraded = {}
        for key, value in state_dict.items():
            head, dot, rest = key.partition(".")
            upgraded[f"{self.LEGACY_NAMES.get(head, head)}{dot}{rest}"] = value
        return super().load_state_dict(upgraded, *args, **kwargs)

    def load_checkpoint(self, state_dict: dict) -> "Checkpointed":
        self.load_state_dict(state_dict)
        self.eval()
        return self


class EdgeProfileModel(Checkpointed):
    """(edge, hour) -> log expected vehicles.

    Deliberately blind to demand: it answers "what does a typical day look like
    on this edge", so it learns one average profile per edge.
    """

    def __init__(self, n_edges: int, n_static: int, emb_dim: int, hidden: int):
        super().__init__()
        self.edge_emb = nn.Embedding(n_edges, emb_dim)
        self.hour_emb = nn.Embedding(HOURS, 8)

    @abstractmethod
    def forward(self, edge: torch.Tensor, hour: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        ...

    def profile(self, edge: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """(B, 24) log rates for whole days."""
        hours = torch.arange(HOURS, device=edge.device).repeat(len(edge))
        return self(edge.repeat_interleave(HOURS), hours,
                    static.repeat_interleave(HOURS, dim=0)).view(-1, HOURS)


class EdgeProfileMLP(EdgeProfileModel):
    """A shared MLP for global time/edge-type patterns, plus per-edge Fourier
    coefficients so every edge owns a smooth 24 h profile."""

    LEGACY_NAMES = {"prof": "own_profile"}

    def __init__(self, n_edges, n_static, emb_dim, hidden):
        super().__init__(n_edges, n_static, emb_dim, hidden)
        self.register_buffer("basis", fourier_basis())
        self.own_profile = nn.Embedding(n_edges, self.basis.shape[1])
        nn.init.zeros_(self.own_profile.weight)
        self.net = nn.Sequential(
            nn.Linear(emb_dim + 8 + 2 + n_static, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, edge, hour, static):
        angle = hour.float() * (2 * torch.pi / HOURS)
        features = torch.cat([self.edge_emb(edge), self.hour_emb(hour),
                              torch.sin(angle)[:, None], torch.cos(angle)[:, None],
                              static], dim=1)
        own = (self.own_profile(edge) * self.basis[hour]).sum(1)
        return (self.net(features).squeeze(1) + own).clamp(max=LOG_RATE_CAP)


class EdgeProfileGRU(EdgeProfileModel):
    """Decodes an edge's whole 24 h profile from its embedding."""

    LEGACY_NAMES = {"init_h": "init_hidden"}

    def __init__(self, n_edges, n_static, emb_dim, hidden):
        super().__init__(n_edges, n_static, emb_dim, hidden)
        self.init_hidden = nn.Linear(emb_dim + n_static, hidden)
        self.gru = nn.GRU(8, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def profile(self, edge, static):
        state = torch.tanh(self.init_hidden(
            torch.cat([self.edge_emb(edge), static], dim=1)))
        steps = self.hour_emb(torch.arange(HOURS, device=edge.device))
        out, _ = self.gru(steps.expand(len(edge), HOURS, 8), state[None])
        return self.head(out).squeeze(2).clamp(max=LOG_RATE_CAP)

    def forward(self, edge, hour, static):
        # decode each distinct edge once, then read off the asked-for hours
        unique, inverse = torch.unique(edge, return_inverse=True)
        representative = torch.empty_like(unique)
        representative.scatter_(0, inverse,
                                torch.arange(len(edge), device=edge.device))
        return self.profile(unique, static[representative])[inverse, hour]


class HistoricalTable:
    """Per-(edge, hour) training average. The baseline the models must beat."""

    def __init__(self, table: np.ndarray):
        self.table = table

    @classmethod
    def fit(cls, n_edges: int, edge_index: np.ndarray, hour: np.ndarray,
            counts: np.ndarray) -> "HistoricalTable":
        total = np.zeros((n_edges, HOURS), np.float32)
        seen = np.zeros((n_edges, HOURS), np.float32)
        np.add.at(total, (edge_index, hour), counts)
        np.add.at(seen, (edge_index, hour), 1)
        return cls(total / np.maximum(seen, 1))

    def rates(self, edge_index: np.ndarray) -> np.ndarray:
        return self.table[np.atleast_1d(edge_index)]


class ODSurrogate(Checkpointed):
    """OD matrix -> per-edge, per-hour counts and travel times.

        OD (24, zones^2) -> GRU over the day -> context c_h
        (edge embedding, static features, c_h) -> MLP -> count, travel time

    The recurrence is what carries congestion built up in earlier hours into
    later ones, which a per-hour feedforward map cannot do.
    """

    LEGACY_NAMES = {"enc": "encoder", "dec": "decoder"}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "ODSurrogate":
        model = cls(checkpoint["n_od"], checkpoint["n_edges"],
                    checkpoint["n_static"], checkpoint["emb_dim"],
                    checkpoint["hidden"])
        return model.load_checkpoint(checkpoint["state_dict"])

    def __init__(self, n_od: int, n_edges: int, n_static: int, emb_dim: int,
                 hidden: int):
        super().__init__()
        self.od_proj = nn.Linear(n_od, hidden)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.edge_emb = nn.Embedding(n_edges, emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim + n_static + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2))

    def context(self, od: torch.Tensor) -> torch.Tensor:
        """Normalised od (B, 24, n_od) -> (B, 24, hidden) demand context."""
        out, _ = self.encoder(torch.relu(self.od_proj(od)))
        return out

    def context_unfused(self, od: torch.Tensor) -> torch.Tensor:
        """The same function, spelled out with primitive ops.

        nn.GRU dispatches to a fused cell PyTorch cannot forward-mode
        differentiate, and the inverse problem needs exactly that: a Jacobian of
        the outputs w.r.t. the OD input. Slower, so only the inverse uses it.
        """
        x = torch.relu(self.od_proj(od))
        w_ih, w_hh = self.encoder.weight_ih_l0, self.encoder.weight_hh_l0
        b_ih, b_hh = self.encoder.bias_ih_l0, self.encoder.bias_hh_l0
        state = torch.zeros(x.shape[0], self.encoder.hidden_size,
                            device=x.device, dtype=x.dtype)
        steps = []
        for t in range(x.shape[1]):
            i_r, i_z, i_n = nn.functional.linear(x[:, t], w_ih, b_ih).chunk(3, -1)
            h_r, h_z, h_n = nn.functional.linear(state, w_hh, b_hh).chunk(3, -1)
            reset = torch.sigmoid(i_r + h_r)
            update = torch.sigmoid(i_z + h_z)
            candidate = torch.tanh(i_n + reset * h_n)
            state = (1 - update) * candidate + update * state
            steps.append(state)
        return torch.stack(steps, dim=1)

    def forward(self, od, edge, static, unfused: bool = False):
        """-> (log count rate, log travel-time ratio), each (B, edges, 24)."""
        context = self.context_unfused(od) if unfused else self.context(od)
        edge_features = torch.cat([self.edge_emb(edge), static], dim=-1)
        batch, n_edges = context.shape[0], edge.shape[0]
        broadcast_context = context[:, None].expand(batch, n_edges, HOURS,
                                                    context.shape[-1])
        broadcast_edges = edge_features[None, :, None].expand(
            batch, n_edges, HOURS, edge_features.shape[-1])
        out = self.decoder(torch.cat([broadcast_edges, broadcast_context], dim=-1))
        return out[..., 0].clamp(max=LOG_RATE_CAP), out[..., 1].clamp(-2.0, 8.0)


FORECAST_MODELS = {"mlp": EdgeProfileMLP, "gru": EdgeProfileGRU}
