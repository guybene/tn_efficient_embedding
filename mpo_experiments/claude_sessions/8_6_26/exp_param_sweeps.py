# %% [markdown]
# PARAM SWEEPS -- alg:mpop on the XX MPO: error and cost vs ONE parameter at a time
# ===================================================================================
r"""
Our sketch only (alg:mpop, XX_REGISTRY["ours"]) on rho_A of the XX ground state.
Every sweep moves ONE parameter away from the base point (CONFIG "base") and keeps
all the others fixed.  One RUN = M single estimates of Tr(rho_A^K) averaged into one
estimate; every point is run `reps` times independently (own seed per run) and the
figures show the mean of the runs with a +- one standard deviation band.

Each swept parameter is run in up to two MODES:
  fixed_m     m stays at the base value            -> the error grows with the parameter
  theorem_m   m moves with the parameter by the asymptotic rule of cor:k-gaussian /
              lem:mpop-correct (theorem_m below)   -> the error stays about the same
                                                      and the cost grows polynomially
The rule, from the base m (only the swept parameter enters, no constants):
  sites   m ~ N^2          (N_E log d_Pi = (N - 1) N log d)
  K       m ~ K^2
  eta     m ~ eta^2        (eta = Tr(A) / ||A||_K, Schatten-K; eta = 1 is rank one)
  eps     m ~ 1 / eps^2
  delta   m ~ log(1 / delta)
eps and delta are not inputs of the algorithm, they act only through m, so they have
the theorem_m mode only:
  eps     y = the relative error of the run's estimate (it should fall with eps);
  delta   y = the fraction of estimates whose relative error exceeds the base eps;
          here one estimate = the mean of delta_block single estimates and a run
          draws delta_estimates of them (a failure FRACTION needs many estimates).
D is not swept: the sketch size of the theorem does not depend on D.

THE ETA SWEEP (CONFIG "eta") leaves the XX point: eta of rho_A is moved by the XXZ
anisotropy Jz inside one magnetization sector (n_up up spins), at its own sites and
chi_max -- the settings scanned on 2026-11-22 to give the widest spread of eta.  The
x value of a point is the eta of the operator actually built.  build_xxz_mps below is
build_xx_mps of sketches.py with these two model parameters added.

Everything else is imported: the operator class and the sketch (sketches.py), the
closed-form cost (asserted == the exact _ec tally at every unit), the estimator loop,
pickles and seeds (exp_xx.py).

FOLDER  results/param_sweeps/<tag>/
  vs_<param>/<mode>/<label>/m=<m>/rep=<i>.pkl     one pickle per run (skipped if done)
  plots/<param>_<mode>.png, plots/summary.json, config.json
"""

# %% [1] imports -------------------------------------------------------------------------
import os
import json
import math
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

from sketches import XX_REGISTRY, XX_MPS, build_xx_mps, estimate_cost, DTYPE
from exp_xx import (cost_formula, single_estimates, seed_for, load_pkl, save_pkl,
                    load_units, subdirs, _lp, peak_rss_mb, DEVICE)

# %% [2] CONFIG --------------------------------------------------------------------------
CONFIG = {
    "dist": "rademacher",
    "seed": 0,

    # the base point: every sweep moves one parameter from here
    "base": {"sites": 8, "K": 4, "chi_max": 3,      # MPO bond D = chi_max^2 = 9
             "m": 16, "eps": 0.3, "delta": 0.2},
    "M": 1000,                  # single estimates averaged into ONE estimate = one run
    "reps": 10,                 # independent runs per point

    "sweeps": ["K", "eta", "sites"],   # run order: sites (largest m) last.  "eps" and
                                       # "delta" are NOT run (2026-11-23): they need a
                                       # proper (eps, delta) experiment first
    "modes": {"sites": ["fixed_m", "theorem_m"], "K": ["fixed_m", "theorem_m"],
              "eta": ["fixed_m", "theorem_m"], "eps": ["theorem_m"],
              "delta": ["theorem_m"]},
    "sites_list": [5, 10, 15, 20, 25, 30, 35, 40],     # N_A (L = 2 N_A)
    "K_list": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "eps_list": [0.5, 0.4, 0.3, 0.25, 0.2],
    "delta_list": [0.4, 0.3, 0.2, 0.1],

    # eta sweep: XXZ, one sector, Jz moves eta (K = base K).  Measured 2026-11-22 at
    # these settings: Jz 4 -> eta 1.08, -1 -> 1.33, -1.75 -> 1.61, -1.9 -> 1.80,
    # -2 -> 2.04
    "eta": {"sites": 5, "chi_max": 5, "n_up": 4,      # D = 25; 4 up spins of L = 10
            "Jz_list": [4.0, -1.0, -1.75, -1.9, -2.0]},

    # delta sweep: one estimate = mean of delta_block singles; estimates per run
    "delta_block": 100,
    "delta_estimates": 100,

    "chunk_budget": 1_000_000,  # samples per batch = budget // (m^2 chi), >= 4
    "out_root": os.path.join("results", "param_sweeps"),
}


# %% [3] the sketch size rule ------------------------------------------------------------
def theorem_m(cfg, param, x):
    """The base m moved by the asymptotic dependence of the theorem's sketch size on
    the swept parameter alone (cor:k-gaussian, lem:mpop-correct); x is the parameter's
    value at the point (for eta: the operator's eta, with eta = 1 as the reference)."""
    b = cfg["base"]
    if param == "sites":
        factor = (x / b["sites"]) ** 2
    elif param == "K":
        factor = (x / b["K"]) ** 2
    elif param == "eta":
        factor = x ** 2
    elif param == "eps":
        factor = (b["eps"] / x) ** 2
    elif param == "delta":
        factor = math.log(1.0 / x) / math.log(1.0 / b["delta"])
    else:
        raise ValueError(param)
    return max(2, round(b["m"] * factor))


# %% [4] the operators -------------------------------------------------------------------
def build_xxz_mps(NA, chi_max, Jz, n_up, device="cpu"):
    """build_xx_mps of sketches.py with two model parameters added: the anisotropy Jz
    (Jz = 0 is the XX model) and the number n_up of up spins among the L = 2 NA sites.
    The model conserves total Sz, so the ground state is sought inside the sector of
    the initial product state; choosing n_up is what a magnetic field would do.  Same
    DMRG options, same bond cap, same check of the spectrum against TeNPy's."""
    L = 2 * NA
    model = XXZChain({"L": L, "Jxx": 2.0, "Jz": Jz, "hz": 0.0, "bc_MPS": "finite"})
    init = ["down"] * L
    for i in np.linspace(0, L - 1, n_up).round().astype(int):   # the ups spread evenly
        init[i] = "up"
    psi = MPS.from_product_state(model.lat.mps_sites(), init, bc="finite")
    dmrg.run(psi, model, {"trunc_params": {"svd_min": 1e-10, "chi_max": chi_max},
                          "max_E_err": 1e-12, "mixer": True,
                          "max_trunc_err": None})      # the cap IS a truncation: no check
    psi.canonical_form()                                # so get_SL is the Schmidt spectrum
    op = XX_MPS([psi.get_B(i, form="B").to_ndarray() for i in range(L)], NA, device)
    lam = torch.as_tensor(psi.get_SL(NA), dtype=DTYPE) ** 2   # rho_A's eigenvalues
    assert torch.allclose(op.spectrum.sort().values[-len(lam):], lam.sort().values,
                          atol=1e-8), "XX_MPS spectrum != TeNPy Schmidt spectrum"
    return op


def get_op(key, _cache={}):
    """rho_A as an MPO, built once per key.  key = (sites, chi_max): the XX builder of
    sketches.py (every sweep but eta); key = (sites, chi_max, Jz, n_up): the XXZ one."""
    if key not in _cache:
        op = (build_xx_mps(key[0], key[1], device=DEVICE) if len(key) == 2
              else build_xxz_mps(*key, device=DEVICE))
        bonds = [c.shape[0] for c in op.mpo_cores] + [op.mpo_cores[-1].shape[3]]
        print(f"  [op {key}] chi={op.chi}  bonds={bonds}  "
              f"spectrum={[round(v, 4) for v in op.spectrum.tolist()]}")
        _cache[key] = (op, bonds)
    return _cache[key]


def eta_of(op, K):
    """eta = Tr(A) / ||A||_K (Schatten-K), from the exact spectrum."""
    return float(op.spectrum.sum() / (op.spectrum ** K).sum() ** (1.0 / K))


# %% [5] one unit = one run at one point -------------------------------------------------
def run_unit(cfg, path, op, bonds, K, m, n, context):
    """n single estimates at sketch size m; est = their mean (one run's estimate)."""
    rec = load_pkl(path)
    if rec is not None and rec["done"]:
        return rec
    t0 = time.time()
    seed_for(cfg, path)
    sk = XX_REGISTRY["ours"](op.n_sites, op.chi, m, cfg["dist"], DEVICE)
    c = max(4, cfg["chunk_budget"] // (m ** 2 * op.chi))
    sk.chunk = lambda K, c=c: c
    tv = op.true_pk(K)
    pool = single_estimates(sk, op, K, n)
    est = pool.mean().item()
    per_sample = cost_formula("ours", bonds, op.d, m, 1, K)
    tally = estimate_cost(sk, op, K)
    assert per_sample == tally, f"closed form {per_sample} != tally {tally}"
    rec = {**context, "sites": op.n_sites, "K": K, "m": m, "bonds": bonds,
           "eta": eta_of(op, K), "spectrum": op.spectrum.cpu().numpy(), "true": tv,
           "n_estimates": n, "pool": pool.numpy(), "est": est,
           "rel_err": abs(est - tv) / abs(tv), "cost_per_sample": per_sample,
           "seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb(), "done": True}
    save_pkl(path, rec)
    return rec


def run(cfg, folder):
    b = cfg["base"]
    for param in cfg["sweeps"]:
        points = []                                   # (label, x, op key, K)
        if param == "eta":
            e = cfg["eta"]
            for Jz in e["Jz_list"]:
                key = (e["sites"], e["chi_max"], Jz, e["n_up"])
                points.append((f"Jz={Jz}", eta_of(get_op(key)[0], b["K"]), key, b["K"]))
        else:
            for x in cfg[f"{param}_list"]:
                sites = x if param == "sites" else b["sites"]
                K = x if param == "K" else b["K"]
                points.append((f"{param}={x}", x, (sites, b["chi_max"]), K))
        n = (cfg["delta_block"] * cfg["delta_estimates"] if param == "delta"
             else cfg["M"])
        for mode in cfg["modes"][param]:
            for label, x, key, K in points:
                op, bonds = get_op(key)
                m = b["m"] if mode == "fixed_m" else theorem_m(cfg, param, x)
                for rep in range(cfg["reps"]):
                    path = os.path.join(folder, f"vs_{param}", mode, label, f"m={m}",
                                        f"rep={rep}.pkl")
                    rec = run_unit(cfg, path, op, bonds, K, m, n,
                                   {"sweep": param, "mode": mode, "label": label,
                                    "x": x, "rep": rep})
                    print(f"  [vs {param} {mode}] {label:12s} m={m:3d} rep={rep}  "
                          f"rel_err={rec['rel_err']:.3f}  cost/sample="
                          f"{rec['cost_per_sample']:.2e}  ({rec['seconds']:.0f}s)")


# %% [6] plot: mean of the runs with a +- std band ----------------------------------------
def plot(cfg, folder):
    plots = os.path.join(folder, "plots")
    os.makedirs(_lp(plots), exist_ok=True)
    b = cfg["base"]
    xlabel = {"sites": "sites N_A", "K": "K", "eta": "eta = Tr(A) / ||A||_K",
              "eps": "eps", "delta": "delta"}
    summary = {}
    for sub in subdirs(folder):
        if not sub.startswith("vs_"):
            continue
        param = sub[3:]
        for mode in subdirs(os.path.join(folder, sub)):
            recs = load_units(os.path.join(folder, sub, mode))
            if not recs:
                continue
            runs = {}                                 # x -> the runs at x
            for r in recs:
                runs.setdefault(r["x"], []).append(r)
            xs = sorted(runs)
            if param == "delta":                      # fraction of block means off by > eps
                ys = [[float(np.mean(np.abs(
                    r["pool"].reshape(-1, cfg["delta_block"]).mean(axis=1) / r["true"]
                    - 1.0) > b["eps"])) for r in runs[x]] for x in xs]
                ylabel = (f"fraction of estimates with relative error > {b['eps']}\n"
                          f"(one estimate = mean of {cfg['delta_block']} singles)")
            else:
                ys = [[r["rel_err"] for r in runs[x]] for x in xs]
                ylabel = f"relative error of the mean of M={cfg['M']} estimates"
            mean = np.array([np.mean(y) for y in ys])
            std = np.array([np.std(y, ddof=1) if len(y) > 1 else 0.0 for y in ys])
            ms = [runs[x][0]["m"] for x in xs]
            costs = [runs[x][0]["cost_per_sample"] for x in xs]
            summary[f"vs_{param}/{mode}"] = [
                {"x": x, "m": m, "runs": len(y), "mean": float(mu), "std": float(sd),
                 "values": y, "cost_per_sample": c}
                for x, m, y, mu, sd, c in zip(xs, ms, ys, mean, std, costs)]

            n_reps = max(len(y) for y in ys)
            if mode == "fixed_m":
                fig, ax_e = plt.subplots(figsize=(8, 5.5))
            else:
                fig, (ax_e, ax_c) = plt.subplots(2, 1, figsize=(8, 9.5))
            ax_e.plot(xs, mean, marker="o", ms=7, lw=2, color="tab:blue",
                      label=f"mean of {n_reps} runs")
            ax_e.fill_between(xs, mean - std, mean + std, color="tab:blue", alpha=0.2,
                              label="+- one standard deviation")
            if param == "delta":
                ax_e.plot(xs, xs, "--", color="gray", label="delta")
            ax_e.set_ylim(bottom=0)
            ax_e.set_xlabel(xlabel[param], fontsize=12)
            ax_e.set_ylabel(ylabel, fontsize=11)
            ax_e.grid(True, alpha=0.3); ax_e.legend(fontsize=10)
            r0 = recs[0]
            fixed = (f"sites={r0['sites']}" if param != "sites" else "") + \
                    (f"  K={r0['K']}" if param != "K" else "")
            m_txt = f"m={b['m']}" if mode == "fixed_m" else "m by the theorem's rule"
            ax_e.set_title(f"ours  |  vs {param}  |  {m_txt}  |  {fixed.strip()}  |  "
                           f"D={max(r0['bonds'])}  |  {cfg['dist']}", fontsize=11)
            if mode == "theorem_m":
                ax_c.plot(xs, costs, marker="o", ms=7, lw=2, color="tab:blue")
                for x, c, m in zip(xs, costs, ms):
                    ax_c.annotate(f"m={m}", (x, c), textcoords="offset points",
                                  xytext=(0, 8), ha="center", fontsize=9)
                ax_c.set_xscale("log"); ax_c.set_yscale("log")
                ax_c.set_xlabel(xlabel[param] + "  (log)", fontsize=12)
                ax_c.set_ylabel("contraction cost of one sample E_K (log)", fontsize=11)
                ax_c.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            path = os.path.join(plots, f"{param}_{mode}.png")
            fig.savefig(_lp(path), dpi=130); plt.close(fig)
            print(f"  saved {path}")
    with open(_lp(os.path.join(plots, "summary.json")), "w") as f:
        json.dump(summary, f, indent=2, default=str)


# %% [7] main ------------------------------------------------------------------------------
def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    b = cfg["base"]
    tag = (f"xx_chi{b['chi_max']}_N{b['sites']}_K{b['K']}_m{b['m']}_M{cfg['M']}"
           f"_reps{cfg['reps']}_{cfg['dist']}_seed{cfg['seed']}")
    folder = os.path.join(cfg["out_root"], tag)
    os.makedirs(_lp(folder), exist_ok=True)
    json.dump(cfg, open(_lp(os.path.join(folder, "config.json")), "w"), indent=2, default=str)
    t0 = time.time()
    run(cfg, folder)
    plot(cfg, folder)
    print(f"\ndone in {time.time() - t0:.0f}s  ->  {os.path.abspath(folder)}")


if __name__ == "__main__":
    main()
