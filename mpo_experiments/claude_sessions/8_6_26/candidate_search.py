# %% [markdown]
# CANDIDATE SEARCH -- ours / solomonik / open_tt on the XX MPO, error and cost vs sites (or K)
# ===========================================================================================
r"""
The three sketching tensor networks of tab:mpop-cost, all at the same m, raced on
rho_A of the XX ground state (build_xx_mps of exp_xx: TeNPy DMRG, MPS bond capped at
chi_max, MPO bond chi^2, d = 2; the spectrum is the capped Schmidt spectrum).  The
sketch size follows the theory's m = O(N): m = m_factor * sites at every point.  Every
candidate averages M single estimates of Tr(rho_A^K) into ONE estimate per x; the
figure has two panels with a shared x-axis, the relative error of that estimate and
its total contraction cost (closed form x M).  No cost matching: M is the same for
every curve.

Everything is imported from sketches.py / exp_xx.py: the operator, the sketches and
their optimal sweeps (XX_REGISTRY), the estimator, the closed-form cost (asserted ==
the exact _ec tally at every unit), pickles, seeds.

FOLDER  results/candidate_search/<tag>/
  vs_<param>/<param>=<x>/<algo>/m=<m>.pkl     one pickle per unit (skipped if done)
  plots/candidate_search_vs_<param>.png, plots/summary.json, config.json
"""

# %% [1] imports -------------------------------------------------------------------------
import os
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketches import XX_REGISTRY, build_xx_mps, estimate_cost
from exp_xx import (cost_formula, single_estimates, seed_for, load_pkl, save_pkl,
                    load_units, subdirs, _lp, peak_rss_mb, DEVICE)

# %% [2] CONFIG --------------------------------------------------------------------------
CONFIG = {
    "chi_max": 3,               # MPS bond cap: rho_A rank <= 3, MPO bond D = chi^2 = 9
    "dist": "rademacher",
    "seed": 0,

    "base": {"sites": 8, "K": 4},     # the fixed parameter of each sweep
    "race_params": ["sites"],         # ["sites", "K"]
    "sites_list": list(range(3, 41)),  # N_A (L = 2 N_A)
    "K_list": [1, 2, 3, 4, 5, 6, 7],
    "algos": ["ours", "solomonik", "open_tt"],
    "m_factor": 1,              # m = m_factor * sites at every point (m = O(N))
    "M": 5000,                  # single estimates averaged into ONE estimate per point
    "chunk_budget": 1_000_000,  # samples per batch = budget // (m^2 chi), >= 4

    "out_root": os.path.join("results", "candidate_search"),
}


# %% [3] the operator + sketch factory ------------------------------------------------------
class Handle:
    """rho_A of the XX ground state at N_A = sites, as an MPO (built once, cached)."""

    def __init__(self, cfg, sites):
        self.cfg, self.sites = cfg, sites
        self.op = build_xx_mps(sites, cfg["chi_max"], device=DEVICE)
        self.bonds = [c.shape[0] for c in self.op.mpo_cores] + [self.op.mpo_cores[-1].shape[3]]
        print(f"  [xx NA={sites}] chi={self.op.chi}  bonds={self.bonds}  "
              f"rho={self.op.intrinsic_dim():.2f}  p2={self.op.true_pk(2):.5f}")

    def make(self, name, m):
        sk = XX_REGISTRY[name](self.sites, self.op.chi, m, self.cfg["dist"], DEVICE)
        c = max(4, self.cfg["chunk_budget"] // (m ** 2 * self.op.chi))
        sk.chunk = lambda K, c=c: c
        return sk


def get_handle(cfg, sites, _cache={}):
    if sites not in _cache:
        _cache[sites] = Handle(cfg, sites)
    return _cache[sites]


# %% [4] one unit = (param, x, algo, m) --------------------------------------------------
def run_unit(cfg, path, h, K, algo, m, context):
    rec = load_pkl(path)
    if rec is not None and rec["done"]:
        return rec
    t0 = time.time()
    seed_for(cfg, path)
    sk = h.make(algo, m)
    tv = h.op.true_pk(K)
    pool = single_estimates(sk, h.op, K, cfg["M"])
    est = pool.mean().item()
    b1 = getattr(sk, "b1", 1)
    per_sample = cost_formula(algo, h.bonds, h.op.d, m, 1, K, b1)
    tally = estimate_cost(sk, h.op, K)
    assert per_sample == tally, f"{algo}: closed form {per_sample} != tally {tally}"
    rec = {**context, "K": K, "algo": algo, "m": m, "b1": b1, "bonds": h.bonds,
           "true": tv, "n_estimates": cfg["M"], "pool": pool.numpy(),
           "est": est, "rel_err": abs(est - tv) / abs(tv),
           "cost_per_sample": per_sample, "cost": cfg["M"] * per_sample,
           "seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb(), "done": True}
    save_pkl(path, rec)
    return rec


def run(cfg, folder):
    base = cfg["base"]
    for param in cfg["race_params"]:
        for x in cfg[f"{param}_list"]:
            sites = x if param == "sites" else base["sites"]
            K = x if param == "K" else base["K"]
            h = get_handle(cfg, sites)
            m = cfg["m_factor"] * sites
            for algo in cfg["algos"]:
                path = os.path.join(folder, f"vs_{param}", f"{param}={x}", algo,
                                    f"m={m}.pkl")
                rec = run_unit(cfg, path, h, K, algo, m,
                               {"sweep": param, "x": x, "sites": sites})
                print(f"  [vs {param}={x}] {algo:10s} m={m:3d}  rel_err="
                      f"{rec['rel_err']:.3f}  cost={rec['cost']:.2e}  "
                      f"({rec['seconds']:.0f}s)")


# %% [5] plot: two panels, shared x -------------------------------------------------------
def plot(cfg, folder):
    plots = os.path.join(folder, "plots")
    os.makedirs(_lp(plots), exist_ok=True)
    look = {"ours": ("tab:blue", "o"), "solomonik": ("tab:green", "s"),
            "open_tt": ("tab:red", "^")}
    summary = {}
    for sub in subdirs(folder):
        if not sub.startswith("vs_"):
            continue
        recs = load_units(os.path.join(folder, sub))
        if not recs:
            continue
        param = recs[0]["sweep"]
        fig, (ax_e, ax_c) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
        for algo in cfg["algos"]:
            pts = sorted((r["x"], r["rel_err"], r["cost"], r["m"]) for r in recs
                         if r["algo"] == algo)
            if not pts:
                continue
            summary[f"vs_{param}/{algo}"] = [
                {"x": p[0], "m": p[3], "rel_err": p[1], "cost": p[2]} for p in pts]
            xs = [p[0] for p in pts]
            col, mk = look[algo]
            ax_e.plot(xs, [p[1] for p in pts], marker=mk, ms=7, lw=2, color=col, label=algo)
            ax_c.plot(xs, [p[2] for p in pts], marker=mk, ms=7, lw=2, color=col, label=algo)
        ax_e.set_yscale("log"); ax_c.set_yscale("log")
        ax_e.set_ylabel(f"relative error of the mean of M={cfg['M']} estimates", fontsize=11)
        ax_c.set_ylabel("total contraction cost", fontsize=11)
        ax_c.set_xlabel("sites N_A" if param == "sites" else "K", fontsize=12)
        base = cfg["base"]
        fixed = f"K={base['K']}" if param == "sites" else f"sites={base['sites']}"
        mf = cfg["m_factor"]
        ax_e.set_title(f"xx  |  chi={cfg['chi_max']}  |  m = {mf if mf != 1 else ''}N_A  |  "
                       f"{fixed}  |  {cfg['dist']}", fontsize=11)
        for ax in (ax_e, ax_c):
            ax.grid(True, which="both", alpha=0.3)
        ax_e.legend(fontsize=10)
        fig.tight_layout()
        path = os.path.join(plots, f"candidate_search_vs_{param}.png")
        fig.savefig(_lp(path), dpi=130); plt.close(fig)
        print(f"  saved {path}")
    with open(_lp(os.path.join(plots, "summary.json")), "w") as f:
        json.dump(summary, f, indent=2, default=str)


# %% [6] main ------------------------------------------------------------------------------
def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    tag = (f"xx_chi{cfg['chi_max']}_m{cfg['m_factor']}N_K{cfg['base']['K']}_M{cfg['M']}"
           f"_{cfg['dist']}_seed{cfg['seed']}")
    folder = os.path.join(cfg["out_root"], tag)
    os.makedirs(_lp(folder), exist_ok=True)
    json.dump(cfg, open(_lp(os.path.join(folder, "config.json")), "w"), indent=2, default=str)
    t0 = time.time()
    run(cfg, folder)
    plot(cfg, folder)
    print(f"\ndone in {time.time() - t0:.0f}s  ->  {os.path.abspath(folder)}")


if __name__ == "__main__":
    main()
