# %% [markdown]
# EXP XX -- Renyi moments of the XX ground state: m* theorem part + method comparison
# =====================================================================================
r"""
EXPERIMENT 2: Renyi moments p_k = Tr(rho_A^k), k >= 2, of the XX ground state.

OPERATOR (XX only -- the identity/all_ones synthetic operators were parked 2026-08-12
to focus on the physical case; restore from archive/retired_2026_08_12_idones.py).
  * xx        rho_A of the XX-chain ground state (L = 2*NA sites, open chain,
              H = J sum (s+ s- + h.c.)): TeNPy DMRG ground state (MPS) -> XX_MPS,
              the operator AS AN MPO: mpo_cores = per-site Kronecker MPS.MPS^*
              cores (bond chi^2, cut closed at build).  Never expanded.  Ground
              truth is exact (chi x chi Gram of the factor W) and cross-checked
              at build time against TeNPy's own Schmidt spectrum at the cut
              (asserted inside build_xx_mps).

THE EXPERIMENT: the MPO is the input; every block S_r A S_{r+1}^T is one L->R
sweep over the MPO cores, fresh per block.  (The MPS experiment was REMOVED
2026-08-28: given the factor W, Tr((W W^H)^k) is exactly computable from the
chi x chi Gram W^H W, so estimating it is unnecessary -- and ma, its sketch,
went with it.)

METHODS (CONFIG algos; matrix sketches reach (eps,delta) single-shot via m*,
scalar probes via the smallest Monte-Carlo averaging batch M -- same protocol
as exp 1):
  ours        OUR sketch (alg:mpop): TT cores over the sites, dangling size-m output
  tt_capped   rMPS probe (the k>1 extension of the Camano/Kressner chain probes), bond
              capped at min(R, sites) -- the Schmidt limit; R from tt_R_list, cheapest
              passing kept
  kron        Rademacher product probe (rows = 1)
  kron_pauli  Noa's partial-basis probe in her XX best-case basis (phi, theta) =
              (pi/2, pi/2) (her Sec. IV C, Table I; fixed in sketches._pauli_table)

OUTPUTS per spectrum: m* vs sites for the matrix sketches (theorem shape ~sites^2
overlaid), cost-to-(eps,delta) vs sites and vs k for all methods (log y), an optional
running-mean error-vs-cost curve panel, config.json + results.json.
COLAB: upload sketches.py next to this file, `pip install physics-tenpy` (the XX
ground state comes from TeNPy DMRG), run the `# %%` cells in order; DEVICE is
auto-detected; heavier Colab values sit in comments next to each CONFIG entry.
"""

# %% [1] imports & device -------------------------------------------------------------
import os
import json
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketches import (XX_REGISTRY, estimate_operator, estimate_cost,
                      build_xx_mps, DTYPE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% [2] CONFIG -----------------------------------------------------------------------
CONFIG = {
    "d": 2,
    "dist": "rademacher",
    "seed": 0,

    "eps": 0.3,
    "delta": 0.2,
    "groups": 500,             # estimators per single-shot pass-rate   (Colab: 2000)
    "m_ceiling": 192,          # m* binary-search cap                   (Colab: 512)
    "pool": 300_000,           # Monte-Carlo pool for the scalar batch search (Colab: 5e6)
    "min_groups": 40,          # censoring floor: batch search stops at pool/min_groups

    "spectra": ["xx"],         # identity/all_ones parked 2026-08-12 (see archive)
    "k_list": [2, 3, 4],
    "base_K": 2,
    "NA_list": [3, 4, 5],      # xx subsystem sizes (L = 2*NA)          (Colab: [4,6,8,10])
    "base_sites": {"xx": 4},

    "algos": ["ours", "tt_capped", "kron", "kron_pauli"],
    "matrix_algos": ["ours"],
    "tt_R_list": [2, 4],       # rMPS probe bonds tried (per point the cheapest passing)


    # optional running-mean error-vs-cost curves (one panel; leading-order batches)
    "curves": {"enabled": True, "spectrum": "xx", "sites": 4, "K": 2, "m": 12,
               "points": 40, "base_batch": 1, "burn": 0.3,
               "batch_pow": {"ours": 0, "tt_capped": 3,
                             "kron": 3, "kron_pauli": 3}},

    "out_root": os.path.join("results", "exp_xx"),
}


# %% [3] operator handles (one interface over the two backends) -----------------------
class XXHandle:
    """rho_A of the XX ground state, as an MPO.  Cost = measured _ec count."""

    def __init__(self, cfg, sites):
        self.cfg, self.sites = cfg, sites
        self.op = build_xx_mps(sites, device=DEVICE)
        pks = "  ".join(f"p{k}={self.op.true_pk(k):.5f}" for k in cfg["k_list"])
        print(f"  [xx NA={sites}] chi={self.op.chi}  rho={self.op.intrinsic_dim():.2f}  {pks}  "
              f"(TeNPy Schmidt cross-check inside build OK)")

    def true(self, K):
        return self.op.true_pk(K)

    def make(self, name, m):
        return XX_REGISTRY[name](self.sites, self.op.chi, m, self.cfg["dist"], DEVICE)

    def estimate(self, sk, facs):
        return estimate_operator(sk, facs, self.op)

    def cost(self, sk, K):
        return estimate_cost(sk, self.op, K)


# (MpoHandle -- the identity/all_ones bond-1 MPO backend -- and the algos_for filter
#  were parked 2026-08-12 with the synthetic spectra: archive/retired_2026_08_12_idones.py)
def get_handle(cfg, spectrum, sites, _cache={}):
    assert spectrum == "xx", "only the xx spectrum is active (id/all_ones parked)"
    if ("xx", sites) not in _cache:
        _cache[("xx", sites)] = XXHandle(cfg, sites)
    return _cache[("xx", sites)]


# %% [4] shared (eps,delta) utilities -------------------------------------------------
@torch.no_grad()
def pass_rate(h, sk, K, eps, M, groups):
    """Fraction of `groups` estimators (each the mean of M single estimates) within eps
    of the truth; the (eps,delta) property holds iff this >= 1 - delta."""
    tv, need, chunk = h.true(K), groups * M, max(sk.chunk(K), M)
    ests, done = [], 0
    while done < need:
        c = min(chunk, need - done)
        ests.append(h.estimate(sk, [sk.draw(c) for _ in range(K)]))
        done += c
    r = torch.cat(ests)[:need].reshape(groups, M).mean(dim=1) / tv
    return ((r - 1.0).abs() <= eps).double().mean().item()


def m_star(cfg, h, name, K, eps, delta):
    """Single-shot m*: smallest m whose ONE estimate is (eps,delta); None = censored."""
    groups, ceil, cache = cfg["groups"], cfg["m_ceiling"], {}

    def pr(m):
        if m not in cache:
            cache[m] = pass_rate(h, h.make(name, m), K, eps, 1, groups)
        return cache[m]

    if pr(ceil) < 1.0 - delta:
        return None
    lo, hi = 1, ceil
    while lo < hi:
        mid = (lo + hi) // 2
        if pr(mid) >= 1.0 - delta:
            hi = mid
        else:
            lo = mid + 1
    return lo


@torch.no_grad()
def sample_pool(h, sk, K, P):
    chunk, ests, done = sk.chunk(K), [], 0
    while done < P:
        c = min(chunk, P - done)
        ests.append(h.estimate(sk, [sk.draw(c) for _ in range(K)]))
        done += c
    return torch.cat(ests)[:P]


def min_batch(cfg, h, sk, K, eps, delta):
    """Smallest averaging batch M whose M-sample mean is (eps,delta), measured on one
    Monte-Carlo pool (pass rate at M = fraction of the pool//M disjoint M-means within
    eps).  Censored (None) beyond M = pool/min_groups."""
    r = sample_pool(h, sk, K, cfg["pool"]) / h.true(K)
    lo_groups = cfg["min_groups"]

    def ok(M):
        gs = len(r) // M
        if gs < lo_groups:
            return False
        means = r[: gs * M].reshape(gs, M).mean(dim=1)
        return ((means - 1.0).abs() <= eps).double().mean().item() >= 1.0 - delta

    ceil = len(r) // lo_groups
    if not ok(ceil):
        return None
    lo, hi = 1, ceil
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


# %% [5] m* part (matrix sketches; the theorem statement) -----------------------------
def mstar_part(cfg, folder):
    colors = {"ours": "tab:blue"}
    res = {}
    for spectrum in cfg["spectra"]:
        sites_list = cfg["NA_list"]
        res[spectrum] = {}
        for K in cfg["k_list"]:
            rows = {a: [] for a in cfg["matrix_algos"]}
            print(f"\n[m* | {spectrum} | K={K}]  eps={cfg['eps']} delta={cfg['delta']}")
            for sites in sites_list:
                h = get_handle(cfg, spectrum, sites)
                for a in cfg["matrix_algos"]:
                    ms = m_star(cfg, h, a, K, cfg["eps"], cfg["delta"])
                    rows[a].append(ms)
                msg = "  ".join(f"{a}={'cens' if rows[a][-1] is None else rows[a][-1]}"
                                for a in cfg["matrix_algos"])
                print(f"   sites={sites}: {msg}")
            res[spectrum][f"K{K}"] = {"sites": sites_list, **rows}
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for a in cfg["matrix_algos"]:
                gx = [s for s, v in zip(sites_list, rows[a]) if v is not None]
                gy = [v for v in rows[a] if v is not None]
                if gx:
                    ax.plot(gx, gy, marker="o", ms=8, lw=2, color=colors[a], label=a)
                    ref = [gy[0] * (x / gx[0]) ** 2 for x in sites_list]
                    ax.plot(sites_list, ref, "--", color=colors[a], alpha=0.5,
                            label=f"{a} theorem shape ~sites^2")
                for s, v in zip(sites_list, rows[a]):
                    if v is None:
                        ax.scatter([s], [cfg["m_ceiling"]], marker="x", s=90,
                                   color=colors[a])
            ax.set_xlabel("sites (N_A for xx, N otherwise)", fontsize=12)
            ax.set_ylabel("m*  (single-shot (eps,delta))", fontsize=12)
            ax.set_title(f"{spectrum}  |  m* vs sites  |  K={K}", fontsize=12)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
            fig.tight_layout()
            path = os.path.join(folder, f"mstar_vs_sites_{spectrum}_K{K}.png")
            fig.savefig(path, dpi=130); plt.close(fig)
            print(f"   saved {os.path.abspath(path)}")
    return res


# %% [6] cost-to-(eps,delta) comparison (all methods) ---------------------------------
def compare_point(cfg, h, K, algos):
    eps, delta, out = cfg["eps"], cfg["delta"], {}
    for name in algos:
        if name in cfg["matrix_algos"]:
            ms = m_star(cfg, h, name, K, eps, delta)
            out[name] = (None if ms is None else
                         {"m": ms, "cost": h.cost(h.make(name, ms), K)})
        elif name == "tt_capped":
            best = None
            for R in cfg["tt_R_list"]:
                sk = h.make(name, R)
                M = min_batch(cfg, h, sk, K, eps, delta)
                if M is not None:
                    cost = M * h.cost(sk, K)
                    if best is None or cost < best["cost"]:
                        best = {"R": R, "M": M, "cost": cost}
            out[name] = best
        else:                                        # kron / kron_pauli (xx only)
            sk = h.make(name, 1)
            M = min_batch(cfg, h, sk, K, eps, delta)
            out[name] = (None if M is None else
                         {"M": M, "cost": M * h.cost(sk, K)})
    return out


def comparison(cfg, folder):
    colors = {"ours": "tab:blue", "tt_capped": "tab:orange",
              "kron": "tab:green", "kron_pauli": "tab:red"}
    res = {}

    def plot(xs, xlabel, rows, title, fname, algos):
        fig, ax = plt.subplots(figsize=(9, 6))
        for a in algos:
            gx = [x for x, r in zip(xs, rows) if r[a] is not None]
            gy = [r[a]["cost"] for r in rows if r[a] is not None]
            if gx:
                ax.plot(gx, gy, marker="o", ms=8, lw=2, color=colors[a], label=a)
            for x, r in zip(xs, rows):
                if r[a] is None:
                    ax.scatter([x], [1], marker="x", s=90, color=colors[a])
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("contraction cost to reach (eps,delta)", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=10)
        fig.tight_layout()
        path = os.path.join(folder, fname)
        fig.savefig(path, dpi=130); plt.close(fig)
        print(f"   saved {os.path.abspath(path)}")

    for spectrum in cfg["spectra"]:
        sites_list = cfg["NA_list"]
        algos = cfg["algos"]
        res[spectrum] = {}
        # ---- cost vs sites at base K ----
        K, rows = cfg["base_K"], []
        print(f"\n[compare | {spectrum} | vs sites | K={K}]  algos={algos}")
        for sites in sites_list:
            h = get_handle(cfg, spectrum, sites)
            r = compare_point(cfg, h, K, algos)
            rows.append(r)
            print(f"   sites={sites}: " + "  ".join(
                f"{a}=" + ("cens" if r[a] is None else f"{r[a]['cost']:.2e}")
                for a in algos))
        res[spectrum]["vs_sites"] = {"sites": sites_list, "rows": rows}
        plot(sites_list, "sites", rows,
             f"{spectrum}  |  cost to (eps,delta)  |  K={K}",
             f"cost_vs_sites_{spectrum}_K{K}.png", algos)
        # ---- cost vs K at base sites ----
        sites = cfg["base_sites"][spectrum]
        h, rows = get_handle(cfg, spectrum, sites), []
        print(f"[compare | {spectrum} | vs K | sites={sites}]")
        for K in cfg["k_list"]:
            r = compare_point(cfg, h, K, algos)
            rows.append(r)
            print(f"   K={K}: " + "  ".join(
                f"{a}=" + ("cens" if r[a] is None else f"{r[a]['cost']:.2e}")
                for a in algos))
        res[spectrum]["vs_K"] = {"K": cfg["k_list"], "rows": rows}
        plot(cfg["k_list"], "K (trace power)", rows,
             f"{spectrum}  |  cost to (eps,delta)  |  sites={sites}",
             f"cost_vs_K_{spectrum}.png", algos)
    return res


# %% [7] optional running-mean error-vs-cost curves -----------------------------------
@torch.no_grad()
def curves(cfg, folder):
    """One panel: relative error of the running mean vs cumulative contraction cost.
    Per step each method draws batch = base * m^batch_pow (LEADING-ORDER rule, never
    the exact cost ratio); every method stops at the same cost budget."""
    cc = cfg["curves"]
    if not cc["enabled"]:
        return None
    spectrum, sites, K, m = cc["spectrum"], cc["sites"], cc["K"], cc["m"]
    algos = cfg["algos"]
    h = get_handle(cfg, spectrum, sites)
    tv, res = h.true(K), {}
    sks = {a: h.make(a, m) for a in algos}
    costs = {a: h.cost(sk, K) for a, sk in sks.items()}
    budget = cc["points"] * max(costs.values())
    fig, ax = plt.subplots(figsize=(12, 7))
    print(f"\n[curves | {spectrum} sites={sites} K={K} m={m}]  budget={budget:.2e}")
    for a in algos:
        sk, cost = sks[a], costs[a]
        batch = max(1, cc["base_batch"] * m ** cc["batch_pow"][a])
        preds, xs, cum, chunk = [], [], 0.0, sk.chunk(K)
        while cum < budget:
            drawn = 0
            while drawn < batch:
                c = min(chunk, batch - drawn)
                preds.append(h.estimate(sk, [sk.draw(c) for _ in range(K)]))
                drawn += c
            cum += batch * cost
            xs.append(cum)
        preds = torch.cat(preds)
        counts = torch.arange(1, len(xs) + 1) * batch
        means = preds.cumsum(0)[counts - 1] / counts
        rel = (means - tv).abs() / abs(tv)
        b0 = int(len(rel) * cc["burn"])
        ax.plot(xs[b0:], rel[b0:], lw=2, marker="o", ms=3,
                label=f"{a}  err@end={rel[-1].item():.2e}")
        res[a] = {"cost_per": cost, "batch": int(batch),
                  "rel_end": float(rel[-1].item())}
        print(f"   [{a:10s}] cost/sample={cost:9d} batch={batch:6d} "
              f"steps={len(xs):4d} rel@end={rel[-1]:.3e}")
    ax.set_xlabel("cumulative contraction cost", fontsize=12)
    ax.set_ylabel("rel. error of running mean", fontsize=12)
    ax.set_title(f"{spectrum}  |  sites={sites} K={K} m={m}  (burn={cc['burn']})",
                 fontsize=12)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(folder, f"curves_{spectrum}_sites{sites}_K{K}_m{m}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"   saved {os.path.abspath(path)}")
    return res


# %% [8] main -------------------------------------------------------------------------
def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    torch.manual_seed(cfg["seed"])
    tag = f"eps{cfg['eps']}_delta{cfg['delta']}_d{cfg['d']}"
    folder = os.path.join(cfg["out_root"], tag)
    os.makedirs(folder, exist_ok=True)
    json.dump(cfg, open(os.path.join(folder, "config.json"), "w"), indent=2)
    print(f"device={DEVICE}  dtype={str(DTYPE).split('.')[-1]}")
    t0 = time.time()
    res = {"mstar": mstar_part(cfg, folder),
           "compare": comparison(cfg, folder),
           "curves": curves(cfg, folder)}
    json.dump(res, open(os.path.join(folder, "results.json"), "w"), indent=2,
              default=str)
    print(f"\ndone in {time.time() - t0:.0f}s  ->  {os.path.abspath(folder)}")
    return res


if __name__ == "__main__":
    main()
