# %% [markdown]
# EXP XX -- Renyi moments of the XX ground state: m* theorem part + race
# =====================================================================================
r"""
EXPERIMENT 2: Renyi moments p_k = Tr(rho_A^k), k >= 2, of the XX ground state.

OPERATOR: rho_A of the XX-chain ground state (L = 2*NA sites, open chain,
H = J sum (s+ s- + h.c.)): TeNPy DMRG ground state (MPS) -> XX_MPS, the operator AS
AN MPO: mpo_cores = per-site Kronecker MPS.MPS^* cores (bond chi^2, cut closed at
build).  Never expanded.  Ground truth is exact (chi x chi Gram of the factor W) and
cross-checked at build against TeNPy's Schmidt spectrum (asserted in build_xx_mps).
The physical dimension is d = 2 (spin-1/2 chain): d is NOT a parameter here, it is
raced on the general tree operator (exp_general_tn).

THE ESTIMATOR: the MPO is the input; every block S_r A S_{r+1}^T is one L->R sweep
over the MPO cores, fresh per block.

METHODS (CONFIG algos):
  ours        OUR sketch (alg:mpop): TT cores over the sites, dangling size-m output
  tt_capped   rMPS probe (the k>1 extension of the Camano/Kressner chain probes), bond
              capped at min(R, sites) -- the Schmidt limit; R from tt_R_list
  kron        Rademacher product probe (rows = 1)
  kron_pauli  Noa's partial-basis probe in her XX best-case basis (phi, theta) =
              (pi/2, pi/2) (her Sec. IV C, Table I; fixed in sketches._pauli_table)
  kressner    bujanovic2026tt Thm 9: ell independent rMPS probes per copy stacked into
              an ell x ell block (rows = ell), probe rank N_A p with
              p = 2 ceil(log(1/delta)/2); ell from kressner_m_list

TWO STEPS, two functions called from main (comment either out):
  run(cfg, folder)   computes and SAVES one pickle per smallest unit of work; a
                     finished file is skipped on rerun, so a crash costs only the
                     unit that crashed.
  plot(cfg, folder)  reads ONLY the pickles and draws the figures + summary.json.

FOLDER HIERARCHY  results/exp_xx/<tag>/
  partA_theorem/<thm1|thm2>/vs_sites/sites=<N>/ours.pkl
  partB_race/vs_<param>/<param>=<value>/<method>/size=<s>.pkl
  plots/*.png, plots/summary.json, config.json

PART A -- THEOREM (ours only): smallest m whose SINGLE estimate is (eps,delta)-
  accurate, vs sites (the N parameter), thm1 (K=1) and thm2 (K=base).  The pickle
  is updated after EVERY pass-rate evaluation of the m* search (resumable).

PART B -- RACE (all methods): contraction cost (exact op count) to reach (eps,delta)
  vs sites, K, eps, delta.  Every method runs a fixed list of sizes (ours: m,
  kressner: ell, tt_capped: bond R, kron/kron_pauli: rank-1); each size gets its
  smallest averaging batch M from one Monte-Carlo pool; cost = M * per-sample cost;
  the plot keeps the cheapest size per method.  One pickle per
  (parameter, value, method, size).

Every pickle is a dict: the point's full context, the raw estimates as numpy arrays
("evals"[m]["est"] for part A, "pool" for part B), the derived numbers, seconds,
peak RSS (Linux only).
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
    "dist": "rademacher",
    "seed": 0,

    "eps": 0.3,
    "delta": 0.2,
    "groups": 500,             # estimators per single-shot pass-rate
    "m_ceiling": 192,          # m* search cap
    "pool": 300_000,           # Monte-Carlo pool for the batch search
    "min_groups": 40,          # censoring floor: batch search stops at pool/min_groups

    "base": {"sites": 4, "K": 2},   # the base point; every sweep moves one parameter

    # PART A (theorem, ours only): m* vs sites, thm1 (K=1) and thm2 (K=base)
    "sites_list": [3, 4, 5],   # xx subsystem sizes N_A (L = 2*N_A)

    # PART B (race, all methods): cost to (eps,delta) vs each parameter
    "race_params": ["sites", "K", "eps", "delta"],
    "K_list": [2, 3, 4],
    "eps_list": [0.5, 0.4, 0.3, 0.25, 0.2],
    "delta_list": [0.4, 0.3, 0.2, 0.1],
    "algos": ["ours", "tt_capped", "kron", "kron_pauli", "kressner"],
    "ours_m_list": [2, 4, 8, 16, 32],      # our sizes tried, each with its own min batch M
    "kressner_m_list": [2, 4, 8],  # ell stacked probes per copy, same (ell, M) search;
                               # probe rank fixed by Thm 9 (bond = sites p, see make)
    "tt_R_list": [2, 4],       # rMPS probe bonds tried

    "out_root": os.path.join("results", "exp_xx"),
}


# %% [3] operator handle + sketch builder ---------------------------------------------
class XXHandle:
    """rho_A of the XX ground state at N_A = sites, as an MPO (built once, cached)."""

    def __init__(self, cfg, sites):
        self.cfg, self.sites = cfg, sites
        self.op = build_xx_mps(sites, device=DEVICE)
        print(f"  [xx NA={sites}] chi={self.op.chi}  rho={self.op.intrinsic_dim():.2f}  "
              f"p2={self.op.true_pk(2):.5f}  (TeNPy Schmidt cross-check inside build OK)")

    def make(self, name, m, delta):
        if name == "kressner":   # bujanovic2026tt Thm 9 rank: N p, p = 2 ceil(log(1/delta)/2)
            p = 2 * math.ceil(0.5 * math.log(1.0 / delta))
            return XX_REGISTRY[name](self.sites, self.op.chi, m, self.cfg["dist"], DEVICE,
                                     bond=self.sites * p)
        return XX_REGISTRY[name](self.sites, self.op.chi, m, self.cfg["dist"], DEVICE)


def get_handle(cfg, sites, _cache={}):
    if sites not in _cache:
        _cache[sites] = XXHandle(cfg, sites)
    return _cache[sites]


def sizes_for(cfg, name):
    """The sizes a method runs in the race (ours: m, kressner: ell, tt_capped: R,
    kron/kron_pauli: rank-1, size ignored)."""
    if name == "ours":
        return cfg["ours_m_list"]
    if name == "kressner":
        return cfg["kressner_m_list"]
    if name == "tt_capped":
        return cfg["tt_R_list"]
    return [1]


# %% [4] bookkeeping: pickles, seeds, memory ------------------------------------------
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


# %% [5] the two primitives: single estimates, batch search ---------------------------
@torch.no_grad()
def single_estimates(sk, op, K, n):
    """n single Tr(A^K) estimates (one per sample), chunked by the sketch's chunk."""
    chunk, ests, done = sk.chunk(K), [], 0
    while done < n:
        c = min(chunk, n - done)
        ests.append(estimate_operator(sk, [sk.draw(c) for _ in range(K)], op))
        done += c
    return torch.cat(ests)[:n].real.cpu()


def min_batch(r, eps, delta, min_groups):
    """Smallest averaging batch M whose M-sample mean is (eps,delta), on one pool r of
    single estimates divided by the truth: the pass rate at M is the fraction of the
    pool//M disjoint M-means within eps.  Binary search on M.  None (censored) once
    fewer than min_groups disjoint means remain, i.e. beyond M = pool/min_groups."""
    def ok(M):
        gs = len(r) // M
        if gs < min_groups:
            return False
        means = r[: gs * M].reshape(gs, M).mean(dim=1)
        return ((means - 1.0).abs() <= eps).double().mean().item() >= 1.0 - delta

    ceil = len(r) // min_groups
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


# %% [6] PART A units: one m* search per pickle, resumable per evaluation -------------
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
            est = single_estimates(h.make("ours", m, delta), op, K, groups)
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
        K = 1 if tag == "thm1" else cfg["base"]["K"]
        for sites in cfg["sites_list"]:
            path = os.path.join(folder, "partA_theorem", tag, "vs_sites",
                                f"sites={sites}", "ours.pkl")
            h = get_handle(cfg, sites)
            rec = run_mstar_unit(cfg, path, h, K, cfg["eps"], cfg["delta"],
                                 {"part": "A", "theorem": tag, "sweep": "sites",
                                  "x": sites, "sites": sites})
            print(f"  [A {tag} vs sites={sites}] K={K} m*="
                  f"{'CENSORED' if rec['m_star'] is None else rec['m_star']}  "
                  f"({rec['seconds']:.0f}s)")


# %% [7] PART B units: one (parameter, value, method, size) per pickle ----------------
def run_race_unit(cfg, path, h, K, eps, delta, algo, size, context):
    """One method at one size at one point: draw the Monte-Carlo pool, find the
    smallest averaging batch M, cost = M * per-sample op count.  Written once,
    complete; a crash mid-pool leaves nothing and the unit reruns alone."""
    rec = load_pkl(path)
    if rec is not None and rec["done"]:
        return rec
    t0 = time.time()
    seed_for(cfg, path)
    sk = h.make(algo, size, delta)
    tv = h.op.true_pk(K)
    pool = single_estimates(sk, h.op, K, cfg["pool"])
    M = min_batch(pool / tv, eps, delta, cfg["min_groups"])
    cost_per = estimate_cost(sk, h.op, K)
    rec = {**context, "K": K, "eps": eps, "delta": delta, "algo": algo, "size": size,
           "rows": sk.rows(), "bond": getattr(sk, "bond", None), "true": tv,
           "pool": pool.numpy(), "M": M, "cost_per_sample": cost_per,
           "cost": None if M is None else M * cost_per,
           "seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb(), "done": True}
    save_pkl(path, rec)
    return rec


def run_part_b(cfg, folder):
    base = cfg["base"]
    for param in cfg["race_params"]:
        for x in cfg[f"{param}_list"]:
            sites = x if param == "sites" else base["sites"]
            K = x if param == "K" else base["K"]
            eps = x if param == "eps" else cfg["eps"]
            delta = x if param == "delta" else cfg["delta"]
            h = get_handle(cfg, sites)
            for algo in cfg["algos"]:
                for size in sizes_for(cfg, algo):
                    path = os.path.join(folder, "partB_race", f"vs_{param}",
                                        f"{param}={x}", algo, f"size={size}.pkl")
                    rec = run_race_unit(cfg, path, h, K, eps, delta, algo, size,
                                        {"part": "B", "sweep": param, "x": x,
                                         "sites": sites})
                    print(f"  [B vs {param}={x}] {algo:10s} size={size:3d}  M="
                          f"{'cens' if rec['M'] is None else rec['M']:>6}  cost="
                          f"{'cens' if rec['cost'] is None else format(rec['cost'], '.2e')}"
                          f"  ({rec['seconds']:.0f}s)")


# %% [8] STEP 1: run ------------------------------------------------------------------
def run(cfg, folder):
    print(f"device={DEVICE}  dtype={str(DTYPE).split('.')[-1]}")
    print("\n[PART A] theorem: m* of ours vs sites")
    run_part_a(cfg, folder)
    print("\n[PART B] race: cost to (eps,delta), all methods")
    run_part_b(cfg, folder)


# %% [9] STEP 2: plot (reads only the pickles) ---------------------------------------
def plot(cfg, folder):
    plots = os.path.join(folder, "plots")
    os.makedirs(_lp(plots), exist_ok=True)
    summary = {"partA": {}, "partB": {}}
    # ---- PART A: m* vs sites, sufficiency shape scaled to the first point ----
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
    # ---- PART B: cheapest certified size per method, cost vs parameter, log y ----
    colors = {"ours": "tab:blue", "tt_capped": "tab:orange", "kron": "tab:green",
              "kron_pauli": "tab:red", "kressner": "tab:purple"}
    for sub in subdirs(os.path.join(folder, "partB_race")):
        recs = load_units(os.path.join(folder, "partB_race", sub))
        if not recs:
            continue
        param = recs[0]["sweep"]
        xs = sorted({r["x"] for r in recs})
        best = {}                                   # (x, algo) -> cheapest passing unit
        for r in recs:
            if r["cost"] is None:
                continue
            key = (r["x"], r["algo"])
            if key not in best or r["cost"] < best[key]["cost"]:
                best[key] = r
        summary["partB"][f"vs_{param}"] = {
            str(x): {a: (None if (x, a) not in best else
                         {"size": best[(x, a)]["size"], "M": best[(x, a)]["M"],
                          "cost": best[(x, a)]["cost"]})
                     for a in cfg["algos"]} for x in xs}
        fig, ax = plt.subplots(figsize=(9, 6))
        for a in cfg["algos"]:
            gx = [x for x in xs if (x, a) in best]
            gy = [best[(x, a)]["cost"] for x in gx]
            if gx:
                ax.plot(gx, gy, marker="o", ms=8, lw=2, color=colors[a], label=a)
            for x in xs:
                if (x, a) not in best:
                    ax.scatter([x], [1], marker="x", s=90, color=colors[a])
        ax.set_yscale("log")
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel("contraction cost to reach (eps,delta)", fontsize=12)
        base = cfg["base"]
        ax.set_title(f"xx  |  cost vs {param}  |  base sites={base['sites']} "
                     f"K={base['K']} eps={cfg['eps']} delta={cfg['delta']}", fontsize=12)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=10)
        fig.tight_layout()
        path = os.path.join(plots, f"cost_vs_{param}.png")
        fig.savefig(_lp(path), dpi=130); plt.close(fig)
        print(f"  saved {path}")
    with open(_lp(os.path.join(plots, "summary.json")), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  saved {os.path.join(plots, 'summary.json')}")


# %% [10] main ------------------------------------------------------------------------
def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    tag = f"eps{cfg['eps']}_delta{cfg['delta']}_d{cfg['d']}"
    folder = os.path.join(cfg["out_root"], tag)
    os.makedirs(_lp(folder), exist_ok=True)
    json.dump(cfg, open(_lp(os.path.join(folder, "config.json")), "w"), indent=2)
    t0 = time.time()
    run(cfg, folder)
    plot(cfg, folder)
    print(f"\ndone in {time.time() - t0:.0f}s  ->  {os.path.abspath(folder)}")


if __name__ == "__main__":
    main()
