#!/usr/bin/env python3
"""The inverse problem: observed travel times -> posterior over the OD matrix.

This is `bayesian scenario.py` with the toy BPR link-cost function replaced by
the trained SUMO surrogate. The structure is identical:

    bayesian scenario.py          this script
    --------------------          -----------
    od_demand ~ HalfNormal        x ~ Normal(0, 1)  (in the surrogate's own
                                    normalized input space, so the prior is
                                    exactly the OD distribution the surrogate
                                    was trained on)
    BPR(od) -> travel time        surrogate(x) -> travel time, counts
    Normal(mu=t, sigma) observed  same
    NUTS                          MAP + Laplace (the surrogate is differentiable,
                                    so we get gradients and a Hessian for free)

Why Laplace instead of NUTS: with 864 unknowns and a neural forward model, NUTS
needs tens of thousands of surrogate evaluations and mixes badly on the ridges
this problem has. The Gauss-Newton Hessian H = I + J'J gives the same posterior
to second order in one shot, and J -- the Jacobian d(travel time)/d(OD) -- is
the classical *assignment matrix*, whose spectrum answers the question that
actually matters:

    HOW MANY DIRECTIONS OF THE OD MATRIX DOES THE DATA ACTUALLY CONSTRAIN?

An eigenvalue of J'J far above 1 means the data pins that OD direction down far
more tightly than the prior does. An eigenvalue near 0 means the observations say
nothing and the posterior just returns the prior. Reporting the count of the
former is the honest way to say whether travel times identify the OD at all.

Example:
    python scripts/invert_od.py --model output_sur/cong/model/surrogate.pt \
        --data output_sur/cong/dataset/data.npz --observe traveltime
"""
import argparse
import json
import os

import numpy as np
import torch

from train_surrogate import ODSurrogate

# cuDNN's fused GRU refuses to backprop in eval mode. We need gradients (and a
# Jacobian) w.r.t. the GRU's *input*, so use the native kernel; the surrogate's
# context_unfused() then handles the forward-mode AD the Jacobian needs.
torch.backends.cudnn.enabled = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="surrogate.pt")
    p.add_argument("--data", required=True, help="data.npz (for the held-out truth)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--observe", choices=["traveltime", "counts", "both"],
                   default="traveltime",
                   help="what the sensors report. 'traveltime' is the deployment case.")
    p.add_argument("--n-sensors", type=int, default=300,
                   help="how many edges are observed (0 = every edge)")
    p.add_argument("--sigma-tt", type=float, default=0.0,
                   help="noise on log(travel time / free flow). 0 = calibrate it "
                        "from the surrogate's own held-out error (recommended)")
    p.add_argument("--n-cases", type=int, default=5, help="held-out ODs to recover")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.model), "inverse")
    os.makedirs(out_dir, exist_ok=True)

    ck = torch.load(args.model, map_location=dev, weights_only=False)
    model = ODSurrogate(ck["n_od"], ck["n_edges"], ck["n_static"],
                        ck["emb_dim"], ck["hidden"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    d = np.load(args.data, allow_pickle=True)
    od_true_all, counts_all, tt_all, ff = (d["od"], d["counts"], d["traveltime"],
                                           d["free_flow"])
    od_mu, od_sd = ck["od_mu"], ck["od_sd"]
    st = torch.tensor(ck["static_norm"], device=dev)
    n_od, n_edges = ck["n_od"], ck["n_edges"]

    # sensors: which edges we get to observe. Prefer edges that actually carry
    # traffic -- a detector on an empty street measures nothing.
    busy = counts_all.mean((0, 2))
    if args.n_sensors and args.n_sensors < n_edges:
        sensors = np.argsort(-busy)[:args.n_sensors]
    else:
        sensors = np.arange(n_edges)
    sensors_t = torch.tensor(sensors, device=dev)
    st_s = st[sensors_t]
    ff_s = torch.tensor(ff[sensors], device=dev)

    # recover held-out scenarios the surrogate never trained on
    cases = ck["test_s"][:args.n_cases]
    mu_t = torch.tensor(od_mu, device=dev)
    sd_t = torch.tensor(od_sd, device=dev)

    # ---- calibrate the likelihood to the surrogate's own accuracy -------------
    # The likelihood must not claim the sensors are more precise than the
    # surrogate is accurate. If it does, the posterior fits surrogate bias rather
    # than traffic and the OD diverges (we measured OD R2 = -169 with a hand-set
    # sigma of 0.05 in the light regime). So take sigma from the surrogate's
    # held-out residuals, and widen the count weights by its relative error too.
    test_all = ck["test_s"]
    odn_test = torch.tensor(
        ((np.log1p(od_true_all[test_all]) - od_mu) / od_sd).astype(np.float32), device=dev)
    with torch.no_grad():
        lc, lt = model(odn_test, sensors_t, st_s)
    tt_true_test = np.log(np.maximum(
        tt_all[test_all][:, sensors] / ff[sensors][:, None], 1e-3))
    resid_tt = lt.cpu().numpy() - tt_true_test
    sigma_tt = float(args.sigma_tt) or float(resid_tt.std())

    c_true_test = counts_all[test_all][:, sensors]
    c_pred_test = torch.exp(lc).cpu().numpy()
    # relative error of the surrogate on the busy edges that carry the signal
    busy_m = c_true_test > 5
    alpha_c = float(np.abs(c_pred_test[busy_m] - c_true_test[busy_m]).std()
                    / max(c_true_test[busy_m].mean(), 1.0)) if busy_m.any() else 0.1

    print(f"surrogate held-out error -> sigma_tt {sigma_tt:.4f} "
          f"(log travel-time ratio), relative count error {alpha_c:.3f}")

    def od_from_x(x):
        """latent (24, n_od) -> OD trips. Inverse of the training normalization."""
        return torch.clamp(torch.expm1(x * sd_t + mu_t), min=0.0)

    def forward(x):
        """latent -> (log tt-ratio, counts) at the sensor edges."""
        log_c, log_t = model(x[None], sensors_t, st_s, unfused=True)
        return log_t[0], torch.exp(log_c[0])          # (S, 24), (S, 24)

    report = {"observe": args.observe, "n_sensors": int(len(sensors)),
              "n_unknowns": int(n_od * 24), "sigma_tt": sigma_tt,
              "alpha_counts": alpha_c, "cases": []}

    for ci, s in enumerate(cases):
        tt_obs = torch.tensor(
            np.log(np.maximum(tt_all[s][sensors] / ff[sensors][:, None], 1e-3)),
            device=dev, dtype=torch.float32)
        c_obs = torch.tensor(counts_all[s][sensors], device=dev, dtype=torch.float32)

        # Poisson counting noise, widened by the surrogate's own error. Weighting
        # by the *observed* count keeps the weight constant and the residual
        # cleanly differentiable.
        c_w = torch.sqrt(c_obs + 1.0 + (alpha_c * c_obs) ** 2)

        def residual(x):
            """Standardized residual: stacking these makes 0.5*||r||^2 the NLL."""
            log_t, c = forward(x)
            parts = []
            if args.observe in ("traveltime", "both"):
                parts.append(((log_t - tt_obs) / sigma_tt).reshape(-1))
            if args.observe in ("counts", "both"):
                parts.append(((c - c_obs) / c_w).reshape(-1))
            return torch.cat(parts)

        # ---- MAP: minimize 0.5||x||^2 (prior) + 0.5||r(x)||^2 (likelihood) ----
        x = torch.zeros(24, n_od, device=dev, requires_grad=True)  # start at prior mean
        opt = torch.optim.Adam([x], lr=args.lr)
        for i in range(args.steps):
            opt.zero_grad()
            r = residual(x)
            loss = 0.5 * (x ** 2).sum() + 0.5 * (r ** 2).sum()
            loss.backward()
            opt.step()

        x_map = x.detach()

        # ---- Laplace: H = I + J'J, with J the assignment matrix ----
        # Built one chunk of input directions at a time: a full jacfwd over all
        # 864 inputs at once would allocate several GB on the GPU.
        D = 24 * n_od
        basis = torch.eye(D, device=dev).reshape(D, 24, n_od)
        col = lambda v: torch.func.jvp(residual, (x_map,), (v,))[1]
        J = torch.cat([torch.func.vmap(col)(basis[i:i + 32])
                       for i in range(0, D, 32)]).T          # (n_residual, D)
        JtJ = J.T @ J
        H = torch.eye(24 * n_od, device=dev) + JtJ
        cov = torch.linalg.inv(H)
        x_sd = torch.sqrt(torch.clamp(torch.diagonal(cov), min=0))

        eig = torch.linalg.eigvalsh(JtJ).clamp(min=0).cpu().numpy()[::-1]
        n_ident = int((eig > 1.0).sum())    # data beats the prior in this direction

        # ---- how well did we actually recover the OD? ----
        od_map = od_from_x(x_map).cpu().numpy()
        od_true = od_true_all[s]
        od_prior = od_from_x(torch.zeros_like(x_map)).cpu().numpy()  # prior mean

        def score(pred):
            ss = ((od_true - pred) ** 2).sum()
            return float(1 - ss / ((od_true - od_true.mean()) ** 2).sum())

        # posterior uncertainty translated back into trips
        od_hi = od_from_x(x_map + x_sd.reshape(24, n_od)).cpu().numpy()
        rel_width = float(np.mean((od_hi - od_map) / (od_map + 1.0)))

        case = {
            "scenario": int(s),
            "od_r2_posterior": score(od_map),
            "od_r2_prior_mean": score(od_prior),
            "total_trips_true": float(od_true.sum()),
            "total_trips_map": float(od_map.sum()),
            "identifiable_directions": n_ident,
            "eig_max": float(eig[0]),
            "eig_median": float(np.median(eig)),
            "posterior_rel_width": rel_width,
        }
        report["cases"].append(case)
        if ci == 0:  # keep the arrays of the first case for the figures
            np.save(os.path.join(out_dir, f"spectrum_{args.observe}.npy"), eig)
            od_sd_trips = (od_hi - od_map)
            np.savez(os.path.join(out_dir, f"recovery_{args.observe}.npz"),
                     true=od_true, map=od_map, sd=od_sd_trips)
        print(f"case {ci} (scenario {s:3d}): "
              f"OD R2 {case['od_r2_posterior']:6.3f} "
              f"(prior mean alone: {case['od_r2_prior_mean']:6.3f})   "
              f"total trips {case['total_trips_map']:8.0f} vs true {case['total_trips_true']:8.0f}   "
              f"identifiable {n_ident:3d}/{24*n_od}")

    ident = [c["identifiable_directions"] for c in report["cases"]]
    r2p = [c["od_r2_posterior"] for c in report["cases"]]
    r2b = [c["od_r2_prior_mean"] for c in report["cases"]]
    report["summary"] = {
        "mean_od_r2_posterior": float(np.mean(r2p)),
        "mean_od_r2_prior_mean": float(np.mean(r2b)),
        "mean_identifiable_directions": float(np.mean(ident)),
        "n_unknowns": 24 * n_od,
    }
    print(f"\n--- {args.observe}, {len(sensors)} sensors ---")
    print(f"  OD recovered  R2 {np.mean(r2p):.3f}   "
          f"(doing nothing and returning the prior mean scores {np.mean(r2b):.3f})")
    print(f"  data constrains {np.mean(ident):.0f} of {24*n_od} OD directions")

    with open(os.path.join(out_dir, f"inverse_{args.observe}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  -> {out_dir}/inverse_{args.observe}.json")


if __name__ == "__main__":
    main()
