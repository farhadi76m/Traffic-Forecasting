#!/usr/bin/env python3
"""Train a model that estimates hourly traffic (vehicles entering an edge).

Input features are only things known at inference time: the edge identity
(learned embedding), the hour of day, and static edge attributes. Scenarios
(simulated days) are split into train/val/test, so evaluation measures how
well the model predicts traffic on *unseen days*.

Models:
  mlp    edge embedding + hour embedding + sin/cos + static features -> MLP
  gru    edge embedding initializes a GRU that decodes the 24-hour profile
  table  historical average per (edge, hour) - baseline

Accuracy: a prediction is "correct" when |pred - true| <= max(abs-tol,
rel-tol * true), i.e. within 20% of the true count or within 3 vehicles
for quiet edges (defaults).

Example:
    python scripts/train.py --data-dir output/dataset --out-dir output/model
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="dir with traffic.csv.gz + edges.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", choices=["mlp", "gru", "table"], default="mlp")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--test-scenarios", type=int, default=4)
    p.add_argument("--val-scenarios", type=int, default=2)
    p.add_argument("--abs-tol", type=float, default=3.0)
    p.add_argument("--rel-tol", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def hour_basis(k=6):
    """(24, 2k+1) Fourier basis over the day: bias + k sin/cos harmonics."""
    h = torch.arange(24, dtype=torch.float32)[:, None]
    freqs = torch.arange(1, k + 1, dtype=torch.float32)[None, :]
    ang = h * freqs * (2 * torch.pi / 24)
    return torch.cat([torch.ones(24, 1), torch.sin(ang), torch.cos(ang)], dim=1)


class MLP(nn.Module):
    """Predicts log(lambda). A shared MLP captures global time/edge-type
    patterns; a per-edge set of Fourier coefficients captures each edge's
    own smooth 24h profile."""

    def __init__(self, n_edges, n_static, emb_dim, hidden):
        super().__init__()
        self.edge_emb = nn.Embedding(n_edges, emb_dim)
        self.hour_emb = nn.Embedding(24, 8)
        self.register_buffer("basis", hour_basis())
        self.prof = nn.Embedding(n_edges, self.basis.shape[1])
        nn.init.zeros_(self.prof.weight)
        self.net = nn.Sequential(
            nn.Linear(emb_dim + 8 + 2 + n_static, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, edge, hour, static):
        ang = hour.float() * (2 * torch.pi / 24)
        x = torch.cat([self.edge_emb(edge), self.hour_emb(hour),
                       torch.sin(ang)[:, None], torch.cos(ang)[:, None], static], dim=1)
        own = (self.prof(edge) * self.basis[hour]).sum(1)
        return (self.net(x).squeeze(1) + own).clamp(max=12.0)


class GRUNet(nn.Module):
    """Decodes an edge's full 24h profile; forward returns value at `hour`."""

    def __init__(self, n_edges, n_static, emb_dim, hidden):
        super().__init__()
        self.edge_emb = nn.Embedding(n_edges, emb_dim)
        self.hour_emb = nn.Embedding(24, 8)
        self.init_h = nn.Linear(emb_dim + n_static, hidden)
        self.gru = nn.GRU(8, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def profile(self, edge, static):
        h0 = torch.tanh(self.init_h(torch.cat([self.edge_emb(edge), static], dim=1)))
        steps = self.hour_emb(torch.arange(24, device=edge.device))
        out, _ = self.gru(steps.expand(len(edge), 24, 8), h0[None])
        return self.head(out).squeeze(2).clamp(max=12.0)  # (B, 24) log-rates

    def forward(self, edge, hour, static):
        return self.profile(edge, static).gather(1, hour[:, None]).squeeze(1)


def metrics(y_true, y_pred, abs_tol, rel_tol):
    err = np.abs(y_pred - y_true)
    ok = err <= np.maximum(abs_tol, rel_tol * y_true)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {"accuracy": float(ok.mean()),
            "r2": float(1 - ss_res / ss_tot),
            "mae": float(err.mean()),
            "rmse": float(np.sqrt((err ** 2).mean()))}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = pd.read_csv(os.path.join(args.data_dir, "traffic.csv.gz"))
    static_df = pd.read_csv(os.path.join(args.data_dir, "edges.csv"))
    edges = static_df["edge"].tolist()
    edge2idx = {e: i for i, e in enumerate(edges)}

    stat_cols = ["length", "speed_limit", "lanes", "priority"]
    stat = static_df[stat_cols].to_numpy(np.float32)
    stat_mean, stat_std = stat.mean(0), stat.std(0) + 1e-6
    stat = (stat - stat_mean) / stat_std

    scenarios = sorted(data["scenario"].unique())
    n_test, n_val = args.test_scenarios, args.val_scenarios
    test_s, val_s = scenarios[-n_test:], scenarios[-n_test - n_val:-n_test]
    train_s = scenarios[:-n_test - n_val]
    print(f"scenarios: train {train_s}, val {val_s}, test {test_s}")

    e_idx = data["edge"].map(edge2idx).to_numpy(np.int64)
    hour = data["hour"].to_numpy(np.int64)
    y = data["count"].to_numpy(np.float32)
    part = np.where(data["scenario"].isin(test_s), 2,
                    np.where(data["scenario"].isin(val_s), 1, 0))

    if args.model == "table":
        tr = part == 0
        table = np.zeros((len(edges), 24), np.float32)
        cnt = np.zeros((len(edges), 24), np.float32)
        np.add.at(table, (e_idx[tr], hour[tr]), y[tr])
        np.add.at(cnt, (e_idx[tr], hour[tr]), 1)
        table /= np.maximum(cnt, 1)
        preds = table[e_idx, hour]
    else:
        Model = MLP if args.model == "mlp" else GRUNet
        model = Model(len(edges), len(stat_cols), args.emb_dim, args.hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        stat_t = torch.tensor(stat, device=device)
        t_edge = torch.tensor(e_idx, device=device)
        t_hour = torch.tensor(hour, device=device)
        # Poisson NLL: the right loss for count data, model outputs log(rate)
        t_y = torch.tensor(y, device=device)

        tr_ids = np.flatnonzero(part == 0)
        val_ids = torch.tensor(np.flatnonzero(part == 1), device=device)
        best_val, best_state, patience = np.inf, None, 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.tensor(np.random.permutation(tr_ids), device=device)
            tot = 0.0
            for b in perm.split(args.batch_size):
                opt.zero_grad()
                pred = model(t_edge[b], t_hour[b], stat_t[t_edge[b]])
                loss = nn.functional.poisson_nll_loss(pred, t_y[b], log_input=True)
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            model.eval()
            with torch.no_grad():
                vl = sum(nn.functional.poisson_nll_loss(
                    model(t_edge[b], t_hour[b], stat_t[t_edge[b]]), t_y[b],
                    log_input=True, reduction="sum").item()
                    for b in val_ids.split(65536)) / len(val_ids)
            sched.step(vl)
            print(f"epoch {epoch:02d}  train_loss {tot/len(tr_ids):.4f}  "
                  f"val_loss {vl:.4f}  lr {opt.param_groups[0]['lr']:.2e}")
            if vl < best_val - 1e-4:
                best_val, patience = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    print("early stop")
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds = np.concatenate([
                torch.exp(model(t_edge[b], t_hour[b], stat_t[t_edge[b]])).cpu().numpy()
                for b in torch.arange(len(y), device=device).split(65536)])

    results = {}
    for name, mask in [("train", part == 0), ("val", part == 1), ("test", part == 2)]:
        if mask.any():
            results[name] = metrics(y[mask], preds[mask], args.abs_tol, args.rel_tol)
            m = results[name]
            print(f"{name:5s}  accuracy {m['accuracy']*100:6.2f}%  R2 {m['r2']:.4f}  "
                  f"MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}")

    # typical-day evaluation: single days are Poisson-noisy, so also score
    # the prediction against the average traffic of the held-out test days
    # (the "typical traffic" a Google-Maps-style forecast estimates)
    te = np.flatnonzero(part == 2)
    key = e_idx[te] * 24 + hour[te]
    n_keys = len(edges) * 24
    y_sum = np.bincount(key, weights=y[te], minlength=n_keys)
    p_sum = np.bincount(key, weights=preds[te], minlength=n_keys)
    n_obs = np.bincount(key, minlength=n_keys)
    seen = n_obs > 0
    results["test_typical"] = metrics(y_sum[seen] / n_obs[seen],
                                      p_sum[seen] / n_obs[seen],
                                      args.abs_tol, args.rel_tol)
    m = results["test_typical"]
    print(f"test (typical day)  accuracy {m['accuracy']*100:6.2f}%  "
          f"R2 {m['r2']:.4f}  MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}")

    meta = {"model": args.model, "edges": edges, "stat_cols": stat_cols,
            "stat_mean": stat_mean.tolist(), "stat_std": stat_std.tolist(),
            "emb_dim": args.emb_dim, "hidden": args.hidden,
            "abs_tol": args.abs_tol, "rel_tol": args.rel_tol, "metrics": results}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    if args.model == "table":
        np.save(os.path.join(args.out_dir, "table.npy"), table)
    else:
        torch.save({"state_dict": model.state_dict(), "static": stat},
                   os.path.join(args.out_dir, "model.pt"))
    print(f"saved model to {args.out_dir}")


if __name__ == "__main__":
    main()
