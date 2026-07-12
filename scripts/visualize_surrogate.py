#!/usr/bin/env python3
"""Figures for the surrogate + inverse experiment.

  1. congestion.png     travel time vs free flow in each regime -- why the
                        inverse problem is solvable in one and not the other
  2. surrogate.png      surrogate vs SUMO on held-out OD, against the OD-blind
                        baseline that the old (edge, hour) model is equivalent to
  3. identifiability.png spectrum of the assignment matrix J'J: how many OD
                        directions the observations actually constrain
  4. recovery.png       true vs recovered OD with posterior error bars

Example:
    python scripts/visualize_surrogate.py --root output_sur --out-dir docs/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_surrogate import ODSurrogate

C_LIGHT, C_CONG, C_BASE = "#4C78A8", "#E45756", "#9E9E9E"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="output_sur")
    p.add_argument("--out-dir", default="docs/figures")
    return p.parse_args()


def load(root, reg):
    d = np.load(f"{root}/{reg}/dataset/data.npz", allow_pickle=True)
    return d


def fig_congestion(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for reg, col, name in [("light", C_LIGHT, "light (~12k trips/day)"),
                           ("cong", C_CONG, "congested (~100k trips/day)")]:
        d = load(root, reg)
        ratio = d["traveltime"] / d["free_flow"][None, :, None]
        peak = ratio[:, :, 7:10].ravel()
        axes[0].hist(np.clip(peak, 1, 4), bins=80, alpha=0.6, color=col,
                     label=f"{name}\nmean {peak.mean():.2f}x", density=True, log=True)
        # sensitivity: how much does travel time move when demand moves?
        tot = d["od"].sum((1, 2))
        tt_mean = ratio[:, :, 7:10].mean((1, 2))
        axes[1].scatter(tot, tt_mean, s=12, alpha=0.5, color=col, label=name)
    axes[0].set_xlabel("travel time / free-flow time (peak 07-09)")
    axes[0].set_ylabel("density (log)")
    axes[0].set_title("Congestion: is travel time even moving?")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("total trips in the OD matrix")
    axes[1].set_ylabel("mean travel time / free-flow, peak")
    axes[1].set_xscale("log")
    axes[1].set_title("Sensitivity of travel time to demand")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/congestion.png", dpi=130)
    print(f"{out}/congestion.png")


def fig_surrogate(root, out):
    regs = [r for r in ("light", "cong") if os.path.exists(f"{root}/{r}/model/metrics.json")]
    fig, axes = plt.subplots(1, len(regs) * 2, figsize=(5.2 * len(regs), 4.2))
    axes = np.atleast_1d(axes)
    k = 0
    for reg in regs:
        m = json.load(open(f"{root}/{reg}/model/metrics.json"))
        d = load(root, reg)
        ck = torch.load(f"{root}/{reg}/model/surrogate.pt", map_location="cpu",
                        weights_only=False)
        model = ODSurrogate(ck["n_od"], ck["n_edges"], ck["n_static"],
                            ck["emb_dim"], ck["hidden"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        test = ck["test_s"][:6]
        odn = (np.log1p(d["od"][test]) - ck["od_mu"]) / ck["od_sd"]
        st = torch.tensor(ck["static_norm"])
        ff = d["free_flow"]
        with torch.no_grad():
            edges = torch.arange(ck["n_edges"])
            lc, lt = model(torch.tensor(odn, dtype=torch.float32), edges, st)
        pred_c = torch.exp(lc).numpy()
        pred_t = (np.exp(lt.numpy()) * ff[None, :, None])
        true_c, true_t = d["counts"][test], d["traveltime"][test]
        blind_t = np.broadcast_to(d["traveltime"][ck["train_s"]].mean(0), true_t.shape)

        for target, (tr, pr, bl, lab) in enumerate([
                (true_c, pred_c, None, "vehicle counts (veh/h)"),
                (true_t, pred_t, blind_t, "travel time (s)")]):
            ax = axes[k]; k += 1
            sub = np.random.default_rng(0).choice(tr.size, 4000, replace=False)
            t, p = tr.ravel()[sub], pr.ravel()[sub]
            if bl is not None:
                ax.scatter(t, bl.ravel()[sub], s=5, alpha=0.25, color=C_BASE,
                           label=f"OD-blind baseline (R2 {m['traveltime_seconds']['blind_r2']:.2f})")
            ax.scatter(t, p, s=5, alpha=0.4,
                       color=C_LIGHT if reg == "light" else C_CONG,
                       label=("surrogate (R2 "
                              + (f"{m['counts']['surrogate_r2']:.2f}" if target == 0
                                 else f"{m['traveltime_seconds']['surrogate_r2']:.2f}") + ")"))
            lim = np.percentile(t, 99.5)
            ax.plot([0, lim], [0, lim], "k--", lw=1)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_xlabel(f"SUMO {lab}")
            ax.set_ylabel("surrogate")
            ax.set_title(f"{reg} regime: {lab.split(' (')[0]}", fontsize=10)
            ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/surrogate.png", dpi=130)
    print(f"{out}/surrogate.png")


def fig_identifiability(root, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {("light", "traveltime"): (C_LIGHT, "-", "light / travel time"),
              ("cong", "traveltime"): (C_CONG, "-", "congested / travel time"),
              ("cong", "counts"): (C_CONG, "--", "congested / counts"),
              ("light", "counts"): (C_LIGHT, "--", "light / counts")}
    n_unknown = None
    for (reg, obs), (col, ls, lab) in styles.items():
        f = f"{root}/{reg}/model/inverse/spectrum_{obs}.npy"
        if not os.path.exists(f):
            continue
        eig = np.load(f)
        n_unknown = len(eig)
        ax.semilogy(np.arange(1, len(eig) + 1), np.maximum(eig, 1e-12),
                    color=col, ls=ls, lw=1.8, label=lab)
    ax.axhline(1.0, color="k", lw=1)
    ax.text(0.98, 1.0, " data = prior ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8)
    ax.set_xlabel(f"OD direction (sorted), of {n_unknown} unknowns")
    ax.set_ylabel("eigenvalue of J'J  (data / prior information)")
    ax.set_title("Identifiability: how many OD directions the sensors constrain\n"
                 "above the line the data wins; below it the posterior is just the prior",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(1e-8, None)
    fig.tight_layout()
    fig.savefig(f"{out}/identifiability.png", dpi=130)
    print(f"{out}/identifiability.png")


def fig_recovery(root, out):
    files = [(r, o) for r in ("light", "cong") for o in ("traveltime", "counts")
             if os.path.exists(f"{root}/{r}/model/inverse/recovery_{o}.npz")]
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(4.6 * len(files), 4.3))
    for ax, (reg, obs) in zip(np.atleast_1d(axes), files):
        z = np.load(f"{root}/{reg}/model/inverse/recovery_{obs}.npz")
        true, mapv, sd = z["true"].ravel(), z["map"].ravel(), z["sd"].ravel()
        col = C_LIGHT if reg == "light" else C_CONG
        ax.errorbar(true, mapv, yerr=sd, fmt="o", ms=3, alpha=0.35, color=col,
                    ecolor=col, elinewidth=0.5, capsize=0)
        lim = max(true.max(), mapv.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ss = 1 - ((true - mapv) ** 2).sum() / ((true - true.mean()) ** 2).sum()
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("true OD flow (trips/h)")
        ax.set_ylabel("recovered (posterior mean +/- sd)")
        ax.set_title(f"{reg} / {obs}   R2 {ss:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/recovery.png", dpi=130)
    print(f"{out}/recovery.png")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for fn in (fig_congestion, fig_surrogate, fig_identifiability, fig_recovery):
        try:
            fn(args.root, args.out_dir)
        except Exception as e:  # noqa: BLE001 - a missing stage should not kill the rest
            print(f"skip {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
