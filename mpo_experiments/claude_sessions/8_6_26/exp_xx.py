# %% [markdown]
# EXP XX -- Renyi moments of the XX ground state: the champion race (+ m* theorem part)
# =====================================================================================
r"""
EXPERIMENT 2: Renyi moments p_k = Tr(rho_A^k) of the XX ground state.

OPERATOR: rho_A of the XX-chain ground state (L = 2*NA sites, open chain,
H = J sum (s+ s- + h.c.)): TeNPy DMRG ground state (MPS) with the bond CAPPED at
chi_max (CONFIG; uncapped chi = 2^NA and the MPO cores explode with NA) -> XX_MPS,
the operator AS AN MPO: mpo_cores = per-site Kronecker MPS.MPS^* cores (bond
chi^2, cut closed at build).  Never expanded.  Ground truth is exact (chi x chi Gram
of the factor W) and cross-checked at build against TeNPy's Schmidt spectrum
(asserted in build_xx_mps).  The physical dimension is d = 2 (spin-1/2 chain).

THE ESTIMATOR: the MPO is the input; every block S_r A S_{r+1}^T is one L->R sweep
over the MPO cores, fresh per block, in the cost-optimal pairwise order of each
method (exhaustive search over all orders of one sweep step, 2026-10-20).

THE RACE -- choosing the sketch for k >= 2 (CONFIG algos), all at the same m:
  ours        OUR sketch (alg:mpop): one order-3 core per site, output edge m
  solomonik   masolomonik2022 Alg. 1: Kronecker leaf d -> m per site + split gadget
  kressner    stacked TT probes (bujanovic2026tt): ell independent TT probes of bond
              m per copy, stacked into an ell x ell block; ell from kressner_ell_list
Each race point = one method at one m (and ell) at one x: `trials` single estimates
(kressner: trials // ell^2, since one of its samples costs ell^2 of ours) averaged
into ONE estimate; the point reports its relative error and its cost = per-sample
cost x number of estimates.  The per-sample cost is the CLOSED FORM of the method
(cost_formula: one term per contraction of the sweep, in the MPO's actual bonds),
asserted equal to the exact _ec tally at every unit, so every point is exactly
accounted for.  Two figures per m: cost and error vs sites, and vs K.

TWO STEPS, two functions called from main (comment either out):
  run(cfg, folder)   computes and SAVES one pickle per unit; a finished file is
                     skipped on rerun.
  plot(cfg, folder)  reads ONLY the pickles and draws the figures + summary.json.

FOLDER HIERARCHY  results/exp_xx/<tag>/
  race/vs_<param>/<param>=<value>/<method>/m=<m>_ell=<ell>.pkl
  partA_theorem/<thm1|thm2>/vs_sites/sites=<N>/ours.pkl      (not run by default)
  plots/*.png, plots/summary.json, config.json

PART A -- THEOREM (ours only, kept, NOT called from run()): smallest m whose SINGLE
  estimate is (eps,delta)-accurate, vs sites, thm1 (K=1) and thm2 (K=mstar_K).
"""

# %% [1] imports & device -------------------------------------------------------------
import os
import json
import math
import time
import zlib
import pickle

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketches import (XX_REGISTRY, estimate_operator, estimate_cost,
                      build_xx_mps, DTYPE)

try:
    import resource                      # peak RSS (Linux server); absent on Windows
except ImportError:
    resource = None

DEVICE = "cpu"      # server: 2GB/CUDA-11.4 GPU is unusable; 125GB RAM CPU is the backend

# %% [2] CONFIG -----------------------------------------------------------------------
CONFIG = {
    "d": 2,                    # fixed by the XX model (spin-1/2); tag only
    "chi_max": 4,              # MPS bond cap in DMRG: rho_A rank <= 4, MPO bond 16,
                               # flat in sites (uncapped chi = 2^sites explodes)
    "dist": "rademacher",
    "seed": 0,

    # THE RACE
    "base": {"sites": 4, "K": 4},   # every sweep moves one parameter from here
    "race_params": ["sites", "K"],
    "sites_list": [2, 4, 6, 8, 12, 16, 20, 24, 28, 30],   # N_A (L = 2*N_A)
    "K_list": [1, 2, 3, 4, 5, 6, 7],
    "algos": ["ours", "solomonik", "kressner"],
    "m_list": [2, 4, 8, 16, 32, 64, 128],   # one pair of figures per m; every method at that m
    "kressner_ell_list": [1, 2, 4, 8],      # stacked probes per copy (ell=1: one TT probe)
    "trials": 100,                 # single estimates averaged into ONE estimate per point;
                                   # kressner averages trials // ell^2

    # PART A (theorem, ours only; not called from run())
    "eps": 0.3,
    "delta": 0.2,
    "groups": 50,              # single estimates per m in the m* search; pass iff
                               # >= (1-delta)*groups of them are within eps
    "m_ceiling": 1024,         # m* search cap
    "mstar_K": 2,
    "mstar_sites_list": [2, 3, 4, 5, 6, 7, 8],

    "out_root": os.path.join("results", "exp_xx"),
}


# %% [3] operator handle + sketch builder ---------------------------------------------
class XXHandle:
    """rho_A of the XX ground state at N_A = sites, as an MPO (built once, cached)."""

    def __init__(self, cfg, sites):
        self.cfg, self.sites = cfg, sites
        self.op = build_xx_mps(sites, cfg["chi_max"], device=DEVICE)
        self.bonds = [c.shape[0] for c in self.op.mpo_cores] + [self.op.mpo_cores[-1].shape[3]]
        print(f"  [xx NA={sites}] chi={self.op.chi}  bonds={self.bonds}  "
              f"rho={self.op.intrinsic_dim():.2f}  p2={self.op.true_pk(2):.5f}")

    def make(self, name, m, ell=1):
        if name == "kressner":   # ell stacked TT probes of bond m: rows = ell
            return XX_REGISTRY[name](self.sites, self.op.chi, ell, self.cfg["dist"], DEVICE,
                                     bond=m)
        return XX_REGISTRY[name](self.sites, self.op.chi, m, self.cfg["dist"], DEVICE)


def get_handle(cfg, sites, _cache={}):
    if sites not in _cache:
        _cache[sites] = XXHandle(cfg, sites)
    return _cache[sites]


def ells_for(cfg, name):
    return cfg["kressner_ell_list"] if name == "kressner" else [1]


# %% [4] closed-form per-sample cost ---------------------------------------------------
def cost_formula(name, bonds, d, m, ell, K, b1=1):
    """EXACT per-sample op count of one E_K estimate as a closed form: one term per
    contraction of the method's sweep (sketches.py), in the MPO's actual bonds
    bonds = [x_0, ..., x_N] (x_n = bond left of site n; x_0 = x_N = 1), N sites.
    Asserted equal to the measured _ec tally at every race unit.
      ours       site 1+2: d^3 m x_1 + d^3 m x_1 x_2 + d^2 m^2 x_2
                 site n>=3 (bonds w, q): d m^3 (w + q) + d^2 m^2 w q
                 chain: (K-1) m^3 + m
      solomonik  site 1: d^2 m x_1 + d m^2 x_1
                 site n>=2: 2 (d m^3 + d b1 m^3) [folds] + d m^3 (w + q) + d^2 m^2 w q
                 chain: (K-1) m^3 + m
      kressner   site 1: ell d^2 m x_1 + ell^2 m^2 d x_1
                 site 2..N-1: ell^2 (m^3 d (w + q) + m^2 d^2 w q)
                 site N (bond w): ell^2 (m^2 d w + m d^2 w + m d)
                 chain: (K-1) ell^3 + ell"""
    N = len(bonds) - 1
    if name == "ours":
        block = d ** 3 * m * bonds[1] + d ** 3 * m * bonds[1] * bonds[2] + d ** 2 * m ** 2 * bonds[2]
        for n in range(2, N):
            w, q = bonds[n], bonds[n + 1]
            block += d * m ** 3 * (w + q) + d ** 2 * m ** 2 * w * q
        return K * block + (K - 1) * m ** 3 + m
    if name == "solomonik":
        block = d ** 2 * m * bonds[1] + d * m ** 2 * bonds[1]
        for n in range(1, N):
            w, q = bonds[n], bonds[n + 1]
            block += 2 * (d * m ** 3 + d * b1 * m ** 3) + d * m ** 3 * (w + q) + d ** 2 * m ** 2 * w * q
        return K * block + (K - 1) * m ** 3 + m
    if name == "kressner":
        block = ell * d ** 2 * m * bonds[1] + ell ** 2 * m ** 2 * d * bonds[1]
        for n in range(1, N - 1):
            w, q = bonds[n], bonds[n + 1]
            block += ell ** 2 * (m ** 3 * d * (w + q) + m ** 2 * d ** 2 * w * q)
        w = bonds[N - 1]
        block += ell ** 2 * (m ** 2 * d * w + m * d ** 2 * w + m * d)
        return K * block + (K - 1) * ell ** 3 + ell
    raise ValueError(name)


# %% [5] bookkeeping: pickles, seeds, memory ------------------------------------------
def _lp(path):
    """Windows extended-length path form: lifts the 260-character limit that
    the deep results hierarchy can hit; a no-op elsewhere."""
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + path
    return path


def load_pkl(path):
    if not os.path.exists(_lp(path)):
        return None
    with open(_lp(path), "rb") as f:
        return pickle.load(f)


def save_pkl(path, rec):
    os.makedirs(_lp(os.path.dirname(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(_lp(tmp), "wb") as f:             # write-then-rename: a crash never
        pickle.dump(rec, f)                     # leaves a half-written pickle
    os.replace(_lp(tmp), _lp(path))


def subdirs(path):
    """Immediate subdirectory names of a results folder (long-path safe)."""
    p = _lp(path)
    if not os.path.isdir(p):
        return []
    return sorted(e.name for e in os.scandir(p) if e.is_dir())


def load_units(root):
    """Every finished pickle under root, recursively (long-path safe)."""
    out = []
    for dirpath, _, files in os.walk(_lp(root)):
        for name in files:
            if name.endswith(".pkl"):
                with open(os.path.join(dirpath, name), "rb") as f:
                    rec = pickle.load(f)
                if rec["done"]:
                    out.append(rec)
    return out


def seed_for(cfg, *keys):
    """Deterministic per-unit seed: the same unit redraws the same samples on a
    rerun, independent of what ran before it."""
    torch.manual_seed(cfg["seed"] ^ zlib.crc32("|".join(str(k) for k in keys).encode()))


def peak_rss_mb():
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # Linux: KB


# %% [6] single estimates --------------------------------------------------------------
@torch.no_grad()
def single_estimates(sk, op, K, n):
    """n single Tr(A^K) estimates (one per sample), chunked by the sketch's chunk."""
    chunk, ests, done = sk.chunk(K), [], 0
    while done < n:
        c = min(chunk, n - done)
        ests.append(estimate_operator(sk, [sk.draw(c) for _ in range(K)], op))
        done += c
    return torch.cat(ests)[:n].real.cpu()


# %% [7] THE RACE: one (parameter, value, method, m, ell) per pickle -----------------
def run_race_unit(cfg, path, h, K, algo, m, ell, context):
    """One method at one (m, ell) at one point: n = trials (kressner: trials // ell^2)
    single estimates averaged into one estimate; relative error of that estimate;
    cost = closed-form per-sample cost (asserted == exact tally) x n."""
    rec = load_pkl(path)
    if rec is not None and rec["done"]:
        return rec
    t0 = time.time()
    seed_for(cfg, path)
    sk = h.make(algo, m, ell)
    tv = h.op.true_pk(K)
    n = max(1, cfg["trials"] // ell ** 2)
    pool = single_estimates(sk, h.op, K, n)
    est = pool.mean().item()
    b1 = getattr(sk, "b1", 1)
    per_sample = cost_formula(algo, h.bonds, h.op.d, m, ell, K, b1)
    tally = estimate_cost(sk, h.op, K)
    assert per_sample == tally, f"{algo}: closed form {per_sample} != tally {tally}"
    rec = {**context, "K": K, "algo": algo, "m": m, "ell": ell, "rows": sk.rows(),
           "b1": b1, "b2": getattr(sk, "b2", None), "bonds": h.bonds, "true": tv,
           "n_estimates": n, "pool": pool.numpy(), "est": est,
           "rel_err": abs(est - tv) / abs(tv),
           "cost_per_sample": per_sample, "cost": n * per_sample,
           "seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb(), "done": True}
    save_pkl(path, rec)
    return rec


def run_race(cfg, folder):
    base = cfg["base"]
    for param in cfg["race_params"]:
        for x in cfg[f"{param}_list"]:
            sites = x if param == "sites" else base["sites"]
            K = x if param == "K" else base["K"]
            h = get_handle(cfg, sites)
            for algo in cfg["algos"]:
                for m in cfg["m_list"]:
                    for ell in ells_for(cfg, algo):
                        path = os.path.join(folder, "race", f"vs_{param}", f"{param}={x}",
                                            algo, f"m={m}_ell={ell}.pkl")
                        rec = run_race_unit(cfg, path, h, K, algo, m, ell,
                                            {"sweep": param, "x": x, "sites": sites})
                        print(f"  [vs {param}={x}] {algo:10s} m={m:3d} ell={ell}  n="
                              f"{rec['n_estimates']:3d}  rel_err={rec['rel_err']:.3f}  "
                              f"cost={rec['cost']:.2e}  ({rec['seconds']:.0f}s)")


# %% [8] PART A units (kept; not called from run()) -----------------------------------
def _suff(K, eps, delta, sites, spectrum):
    """The sufficiency-bound shape (cor:k1-gaussian for K=1, cor:k-gaussian for K>=2)
    up to its constant: N_E = sites - 1 cores, d_Pi = 2^sites, and
    rho = Tr(A)/Tr(A^K)^(1/K) from the exact spectrum."""
    logs = sites * math.log(2) + math.log(max(K, 1) / delta)
    if K == 1:
        return (sites - 1) * logs / eps ** 2
    rho = float(spectrum.sum() / (spectrum ** K).sum() ** (1.0 / K))
    return (sites - 1) * K ** 2 * max(1.0, rho ** 2 * eps) * logs / eps ** 2


def run_mstar_unit(cfg, path, h, K, eps, delta, context):
    """Single-shot m* of `ours` at one point: smallest m in [1, m_ceiling] whose ONE
    estimate is (eps,delta).  Doubling search up from m=1 (cheap points never
    evaluate the ceiling), then binary search in the last interval; None if
    censored.  EVERY pass-rate evaluation is written to the pickle before the next
    one starts, so a rerun resumes the search."""
    rec = load_pkl(path)
    if rec is not None and rec["done"]:
        return rec
    op = h.op
    if rec is None:
        rec = {**context, "K": K, "eps": eps, "delta": delta,
               "true": op.true_pk(K), "spectrum": op.spectrum.cpu().numpy(),
               "suff": _suff(K, eps, delta, h.sites, op.spectrum),
               "evals": {}, "m_star": None, "seconds": 0.0, "peak_rss_mb": None,
               "done": False}
    groups, ceil = cfg["groups"], cfg["m_ceiling"]

    def pr(m):
        if m not in rec["evals"]:
            t0 = time.time()
            seed_for(cfg, path, m)
            est = single_estimates(h.make("ours", m), op, K, groups)
            rate = ((est / rec["true"] - 1.0).abs() <= eps).double().mean().item()
            rec["evals"][m] = {"est": est.numpy(), "pass_rate": rate}
            rec["seconds"] += time.time() - t0
            rec["peak_rss_mb"] = peak_rss_mb()
            save_pkl(path, rec)
        return rec["evals"][m]["pass_rate"]

    m = 1
    while m < ceil and pr(m) < 1.0 - delta:
        m = min(2 * m, ceil)
    if pr(m) >= 1.0 - delta:
        lo, hi = m // 2 + 1, m
        while lo < hi:
            mid = (lo + hi) // 2
            if pr(mid) >= 1.0 - delta:
                hi = mid
            else:
                lo = mid + 1
        rec["m_star"] = lo
    rec["done"] = True
    save_pkl(path, rec)
    return rec


def run_part_a(cfg, folder):
    for tag in ("thm1", "thm2"):
        K = 1 if tag == "thm1" else cfg["mstar_K"]
        for sites in cfg["mstar_sites_list"]:
            path = os.path.join(folder, "partA_theorem", tag, "vs_sites",
                                f"sites={sites}", "ours.pkl")
            h = get_handle(cfg, sites)
            rec = run_mstar_unit(cfg, path, h, K, cfg["eps"], cfg["delta"],
                                 {"part": "A", "theorem": tag, "sweep": "sites",
                                  "x": sites, "sites": sites})
            print(f"  [A {tag} vs sites={sites}] K={K} m*="
                  f"{'CENSORED' if rec['m_star'] is None else rec['m_star']}  "
                  f"({rec['seconds']:.0f}s)")


# %% [9] STEP 1: run ------------------------------------------------------------------
def run(cfg, folder):
    print(f"device={DEVICE}  dtype={str(DTYPE).split('.')[-1]}")
    print("\n[RACE] ours / solomonik / kressner: error and cost vs sites, vs K")
    run_race(cfg, folder)
    # run_part_a(cfg, folder)      # the m* theorem part (ours only), on demand


# %% [10] STEP 2: plot (reads only the pickles) --------------------------------------
def plot(cfg, folder):
    plots = os.path.join(folder, "plots")
    os.makedirs(_lp(plots), exist_ok=True)
    summary = {"race": {}, "partA": {}}
    colors = {"ours": "tab:blue", "solomonik": "tab:green", "kressner": "tab:purple"}
    styles = {1: "-", 2: "--", 4: ":", 8: "-."}       # kressner ell
    # ---- RACE: per parameter and per m, two panels (cost, error) vs x ----
    for sub in subdirs(os.path.join(folder, "race")):
        recs = load_units(os.path.join(folder, "race", sub))
        if not recs:
            continue
        param = recs[0]["sweep"]
        for m in cfg["m_list"]:
            rm = [r for r in recs if r["m"] == m]
            if not rm:
                continue
            curves = sorted({(r["algo"], r["ell"]) for r in rm})
            summary["race"][f"vs_{param}_m{m}"] = {
                f"{a}_ell{e}": sorted(
                    [{"x": r["x"], "n": r["n_estimates"], "rel_err": r["rel_err"],
                      "cost_per_sample": r["cost_per_sample"], "cost": r["cost"],
                      "est": r["est"], "true": r["true"]}
                     for r in rm if (r["algo"], r["ell"]) == (a, e)], key=lambda t: t["x"])
                for a, e in curves}
            fig, (ax_c, ax_e) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
            for a, e in curves:
                pts = sorted((r["x"], r["cost"], r["rel_err"]) for r in rm
                             if (r["algo"], r["ell"]) == (a, e))
                xs = [p[0] for p in pts]
                lab = a if a != "kressner" else f"kressner ell={e}"
                ax_c.plot(xs, [p[1] for p in pts], marker="o", ms=6, lw=2,
                          color=colors[a], ls=styles.get(e, "-"), label=lab)
                ax_e.plot(xs, [p[2] for p in pts], marker="o", ms=6, lw=2,
                          color=colors[a], ls=styles.get(e, "-"), label=lab)
            ax_c.set_yscale("log"); ax_e.set_yscale("log")
            ax_c.set_ylabel("contraction cost of the estimate", fontsize=12)
            ax_e.set_ylabel("relative error of the estimate", fontsize=12)
            ax_e.set_xlabel(param, fontsize=12)
            base = cfg["base"]
            fixed = f"K={base['K']}" if param == "sites" else f"sites={base['sites']}"
            ax_c.set_title(f"xx  |  m={m}  |  vs {param}  |  {fixed}  |  "
                           f"{cfg['trials']} estimates averaged (kressner: /ell^2)", fontsize=11)
            for ax in (ax_c, ax_e):
                ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
            fig.tight_layout()
            path = os.path.join(plots, f"race_vs_{param}_m{m}.png")
            fig.savefig(_lp(path), dpi=130); plt.close(fig)
            print(f"  saved {path}")
    # ---- PART A (if any pickles exist): m* vs sites, sufficiency shape ----
    for tag in ("thm1", "thm2"):
        recs = sorted(load_units(os.path.join(folder, "partA_theorem", tag)),
                      key=lambda r: r["x"])
        if not recs:
            continue
        summary["partA"][f"{tag}_vs_sites"] = [
            {"x": r["x"], "m_star": r["m_star"], "suff": r["suff"], "seconds": r["seconds"]}
            for r in recs]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        gx = [r["x"] for r in recs if r["m_star"] is not None]
        gm = [r["m_star"] for r in recs if r["m_star"] is not None]
        if gx:
            ax.plot(gx, gm, marker="o", ms=8, lw=2, color="tab:blue", label="measured m*")
            s0 = next(r["suff"] for r in recs if r["m_star"] is not None)
            ax.plot([r["x"] for r in recs], [gm[0] * r["suff"] / s0 for r in recs], "--",
                    color="tab:blue", alpha=0.5, label="sufficiency shape")
        for r in recs:
            if r["m_star"] is None:
                ax.scatter([r["x"]], [cfg["m_ceiling"]], marker="x", s=90, color="tab:blue")
        ax.set_xlabel("sites (N_A)", fontsize=12)
        ax.set_ylabel("m*  (single-shot (eps,delta))", fontsize=12)
        ax.set_title(f"xx  |  {tag}  |  m* vs sites  |  K={recs[0]['K']}", fontsize=12)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
        fig.tight_layout()
        path = os.path.join(plots, f"mstar_{tag}_vs_sites.png")
        fig.savefig(_lp(path), dpi=130); plt.close(fig)
        print(f"  saved {path}")
    with open(_lp(os.path.join(plots, "summary.json")), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  saved {os.path.join(plots, 'summary.json')}")


# %% [11] main ------------------------------------------------------------------------
def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    tag = f"race_d{cfg['d']}_chi{cfg['chi_max']}_trials{cfg['trials']}"
    folder = os.path.join(cfg["out_root"], tag)
    os.makedirs(_lp(folder), exist_ok=True)
    json.dump(cfg, open(_lp(os.path.join(folder, "config.json")), "w"), indent=2)
    t0 = time.time()
    run(cfg, folder)
    plot(cfg, folder)
    print(f"\ndone in {time.time() - t0:.0f}s  ->  {os.path.abspath(folder)}")


if __name__ == "__main__":
    main()
