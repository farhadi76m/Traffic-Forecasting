#!/usr/bin/env python3
"""Predict traffic (vehicles/hour) for an edge at a given time.

Examples:
    python scripts/predict.py --model-dir output/model --edge 150770686#0 --time 8
    python scripts/predict.py --model-dir output/model --edge 150770686#0 --time 17:30
    python scripts/predict.py --model-dir output/model --edge 150770686#0 --profile
"""
import argparse
import json
import os

import numpy as np


def parse_time(t):
    h = int(str(t).split(":")[0])
    if not 0 <= h <= 23:
        raise SystemExit(f"hour must be 0-23, got {t}")
    return h


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--edge", required=True, help="edge id from the network")
    p.add_argument("--time", help="hour of day, e.g. 7 or 07:30")
    p.add_argument("--profile", action="store_true", help="print all 24 hours")
    return p.parse_args()


def load_predictor(model_dir):
    """Returns (meta, fn); fn(edge_indices) -> (n, 24) hourly rates."""
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)
    if meta["model"] == "table":
        table = np.load(os.path.join(model_dir, "table.npy"))
        return meta, lambda idx: table[np.atleast_1d(idx)]

    import torch
    from train import MLP, GRUNet
    ckpt = torch.load(os.path.join(model_dir, "model.pt"),
                      map_location="cpu", weights_only=False)
    Model = MLP if meta["model"] == "mlp" else GRUNet
    model = Model(len(meta["edges"]), len(meta["stat_cols"]),
                  meta["emb_dim"], meta["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    static = torch.tensor(ckpt["static"])

    def fn(idx):
        idx = np.atleast_1d(idx)
        edge = torch.as_tensor(idx, dtype=torch.long).repeat_interleave(24)
        hours = torch.arange(24).repeat(len(idx))
        with torch.no_grad():
            out = torch.exp(model(edge, hours, static[edge]))
        return out.numpy().reshape(len(idx), 24)
    return meta, fn


def main():
    args = parse_args()
    if not args.time and not args.profile:
        raise SystemExit("give --time HH[:MM] and/or --profile")
    meta, predict = load_predictor(args.model_dir)
    try:
        idx = meta["edges"].index(args.edge)
    except ValueError:
        raise SystemExit(f"edge '{args.edge}' not in the network the model was "
                         f"trained on ({len(meta['edges'])} edges)")

    rates = predict(idx)[0]
    if args.profile:
        peak = int(np.argmax(rates))
        print(f"edge {args.edge} - estimated traffic (vehicles/hour)")
        for h in range(24):
            bar = "#" * int(round(rates[h] / max(rates.max(), 1) * 40))
            mark = "  <- peak" if h == peak else ""
            print(f"  {h:02d}:00  {rates[h]:7.1f}  {bar}{mark}")
    if args.time is not None:
        h = parse_time(args.time)
        print(f"edge {args.edge} at {h:02d}:00 -> {rates[h]:.1f} vehicles/hour")


if __name__ == "__main__":
    main()
