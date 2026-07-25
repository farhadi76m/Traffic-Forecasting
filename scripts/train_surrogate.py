#!/usr/bin/env python3
"""Train a neural surrogate of SUMO: OD matrix -> per-edge, per-hour traffic.

    OD (24, n_zones^2)  ->  [GRU over the 24 hours]  ->  context c_h (24, H)
    (edge embedding, static features, c_h)  ->  MLP  ->  counts_h, traveltime_h

This is `train.py`'s GRU with one change that matters: the GRU is driven by the
OD, not by an edge embedding alone. The old model's inputs were (edge, hour), so
it could only ever reproduce the average day -- it had no way to react to demand.
Here the hour-to-hour recurrence is what carries congestion built up in earlier
hours into later ones, which is the part a per-hour feedforward map cannot do.

Two heads, each with the right likelihood for its data:
  counts       Poisson NLL on log-rate   (counts are counts)
  travel time  MSE on log(t / freeflow)  (a positive, heavy-tailed ratio)

THE metric to read is `vs OD-blind baseline`. The baseline predicts the training
mean for each (edge, hour) and ignores the OD completely. Any model can score a
high R2 on this data just by learning the average rush hour. Only the gap over
the baseline is evidence that the surrogate actually learned OD -> traffic.

Example:
    python scripts/train_surrogate.py --data output_sur/cong/dataset/data.npz \
        --out-dir output_sur/cong/model
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="data.npz from build_surrogate_dataset.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--scen-batch", type=int, default=8, help="scenarios per step")
    p.add_argument("--edge-batch", type=int, default=768, help="edges per step")
    p.add_argument("--w-tt", type=float, default=1.0, help="weight of the travel-time loss")
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class ODSurrogate(nn.Module):
    def __init__(self, n_od, n_edges, n_static, emb_dim, hidden):
        super().__init__()
        self.od_proj = nn.Linear(n_od, hidden)
        self.enc = nn.GRU(hidden, hidden, batch_first=True)
        self.edge_emb = nn.Embedding(n_edges, emb_dim)
        self.dec = nn.Sequential(
            nn.Linear(emb_dim + n_static + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2))

    def context(self, od):
        """od (B, 24, n_od), already normalized -> (B, 24, H) demand context."""
        c, _ = self.enc(torch.relu(self.od_proj(od)))
        return c

    def context_unfused(self, od):
        """Exactly context(), spelled out with primitive ops.

        nn.GRU dispatches to a fused cell that PyTorch cannot forward-mode
        differentiate, and the inverse problem needs precisely that: a Jacobian
        of the outputs w.r.t. the OD *input*. This loop reads the same weights
        and computes the same function using ops that do support forward AD.
        Slower, so training keeps the fused path and only invert_od.py uses this.
        """
        x = torch.relu(self.od_proj(od))
        W_ih, W_hh = self.enc.weight_ih_l0, self.enc.weight_hh_l0
        b_ih, b_hh = self.enc.bias_ih_l0, self.enc.bias_hh_l0
        h = torch.zeros(x.shape[0], self.enc.hidden_size,
                        device=x.device, dtype=x.dtype)
        out = []
        for t in range(x.shape[1]):
            i_r, i_z, i_n = nn.functional.linear(x[:, t], W_ih, b_ih).chunk(3, -1)
            h_r, h_z, h_n = nn.functional.linear(h, W_hh, b_hh).chunk(3, -1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1 - z) * n + z * h
            out.append(h)
        return torch.stack(out, dim=1)

    def forward(self, od, edge, static, unfused=False):
        """-> (B, n_edge_batch, 24, 2): log count-rate, log travel-time ratio."""
        c = self.context_unfused(od) if unfused else self.context(od)  # (B, 24, H)
        e = torch.cat([self.edge_emb(edge), static], dim=-1)      # (E, D)
        B, E = c.shape[0], edge.shape[0]
        ce = c[:, None].expand(B, E, 24, c.shape[-1])
        ee = e[None, :, None].expand(B, E, 24, e.shape[-1])
        out = self.dec(torch.cat([ee, ce], dim=-1))
        return out[..., 0].clamp(max=12.0), out[..., 1].clamp(-2.0, 8.0)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.data, allow_pickle=True)
    od, counts, tt, ff, static = (d["od"], d["counts"], d["traveltime"],
                                  d["free_flow"], d["static"])
    n_scen, n_edges = counts.shape[0], counts.shape[1]
    n_od = od.shape[-1]

    # targets: counts as-is (Poisson), travel time as log-ratio over free flow
    tt_ratio = np.log(np.maximum(tt / ff[None, :, None], 1e-3)).astype(np.float32)

    # inputs: log1p the OD (it spans orders of magnitude), then standardize
    perm = rng.permutation(n_scen)
    n_test = int(args.test_frac * n_scen)
    n_val = int(args.val_frac * n_scen)
    test_s, val_s, train_s = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    od_log = np.log1p(od).astype(np.float32)
    od_mu = od_log[train_s].mean((0, 1))
    od_sd = od_log[train_s].std((0, 1)) + 1e-6
    od_n = (od_log - od_mu) / od_sd

    st_mu, st_sd = static.mean(0), static.std(0) + 1e-6
    st_n = ((static - st_mu) / st_sd).astype(np.float32)

    print(f"{n_scen} scenarios ({len(train_s)} train / {len(val_s)} val / {len(test_s)} test), "
          f"{n_edges} edges, OD dim {n_od}")

    T = lambda x: torch.tensor(x, device=dev)
    od_t, cnt_t, ttr_t, st_t = T(od_n), T(counts), T(tt_ratio), T(st_n)
    all_edges = torch.arange(n_edges, device=dev)

    model = ODSurrogate(n_od, n_edges, static.shape[1], args.emb_dim, args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

    def predict(scen_ids, edge_chunk=1024):
        """Full (S, n_edges, 24) prediction, chunked over edges to fit the GPU."""
        model.eval()
        cs, ts = [], []
        with torch.no_grad():
            for e0 in range(0, n_edges, edge_chunk):
                eb = all_edges[e0:e0 + edge_chunk]
                lc, lt = model(od_t[scen_ids], eb, st_t[eb])
                cs.append(torch.exp(lc))
                ts.append(lt)
        return torch.cat(cs, 1), torch.cat(ts, 1)

    def evaluate(ids):
        pc, pt = predict(T(ids))
        yc, yt = cnt_t[ids], ttr_t[ids]
        return (r2(yc.cpu().numpy(), pc.cpu().numpy()),
                r2(yt.cpu().numpy(), pt.cpu().numpy()),
                float((pc - yc).abs().mean()),
                float((pt - yt).abs().mean()))

    best, best_state, bad = np.inf, None, 0
    steps = max(1, len(train_s) // args.scen_batch)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for _ in range(steps):
            sb = T(rng.choice(train_s, args.scen_batch, replace=False))
            eb = all_edges[torch.tensor(
                rng.choice(n_edges, args.edge_batch, replace=False), device=dev)]
            opt.zero_grad()
            log_c, log_t = model(od_t[sb], eb, st_t[eb])
            loss_c = nn.functional.poisson_nll_loss(
                log_c, cnt_t[sb][:, eb], log_input=True)
            loss_t = nn.functional.mse_loss(log_t, ttr_t[sb][:, eb])
            loss = loss_c + args.w_tt * loss_t
            loss.backward()
            opt.step()
            tot += float(loss)

        if ep % 5 == 0 or ep == args.epochs - 1:
            vc, vt, mc, mt = evaluate(val_s)
            vloss = -vc - vt
            sched.step(vloss)
            print(f"ep {ep:3d}  loss {tot/steps:8.4f}   val R2  counts {vc:6.3f}  "
                  f"tt {vt:6.3f}   lr {opt.param_groups[0]['lr']:.1e}")
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    print("early stop")
                    break

    model.load_state_dict(best_state)

    # --- the honest evaluation -------------------------------------------------
    # OD-blind baseline: predict the training mean for each (edge, hour).
    base_c = counts[train_s].mean(0)      # (n_edges, 24)
    base_t = tt_ratio[train_s].mean(0)

    pc, pt = predict(T(test_s))
    pc, pt = pc.cpu().numpy(), pt.cpu().numpy()
    yc, yt = counts[test_s], tt_ratio[test_s]
    bc = np.broadcast_to(base_c, yc.shape)
    bt = np.broadcast_to(base_t, yt.shape)

    res = {
        "counts": {"surrogate_r2": r2(yc, pc), "blind_r2": r2(yc, bc),
                   "surrogate_mae": float(np.abs(yc - pc).mean()),
                   "blind_mae": float(np.abs(yc - bc).mean())},
        "traveltime_logratio": {"surrogate_r2": r2(yt, pt), "blind_r2": r2(yt, bt),
                                "surrogate_mae": float(np.abs(yt - pt).mean()),
                                "blind_mae": float(np.abs(yt - bt).mean())},
    }
    # travel time in seconds, the number a user cares about
    tt_true = tt[test_s]
    tt_pred = np.exp(pt) * ff[None, :, None]
    res["traveltime_seconds"] = {
        "surrogate_r2": r2(tt_true, tt_pred),
        "blind_r2": r2(tt_true, np.exp(bt) * ff[None, :, None]),
        "surrogate_mae": float(np.abs(tt_true - tt_pred).mean()),
    }

    print("\n--- test set (unseen OD matrices) ---")
    for k, v in res.items():
        line = f"  {k:22s} R2 {v['surrogate_r2']:7.4f}"
        if "blind_r2" in v:
            line += (f"   OD-blind baseline R2 {v['blind_r2']:7.4f}"
                     f"   (gap {v['surrogate_r2'] - v['blind_r2']:+.4f})")
        print(line)
    print("\n  The gap is what reading the OD buys. If it is ~0, the model is")
    print("  ignoring the OD and just predicting the average day.")

    torch.save({"state_dict": model.state_dict(), "od_mu": od_mu, "od_sd": od_sd,
                "static_norm": st_n, "free_flow": ff,
                "n_od": n_od, "n_edges": n_edges, "n_static": static.shape[1],
                "emb_dim": args.emb_dim, "hidden": args.hidden,
                "test_s": test_s, "train_s": train_s},
               os.path.join(args.out_dir, "surrogate.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
