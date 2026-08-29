# %% [markdown]
# EXP GENERAL TN -- the general-framework experiment (theorem validation + comparison)
# =====================================================================================
r"""
EXPERIMENT 1: a hand-crafted GENERAL symmetric tensor network (not an MPO).

THE OPERATOR.  A = C C^H (PSD, complex), where C = mat(T) and T is the fixed 5-node TREE

        e1   e2
         \   /             dangling (physical) legs: e1, e2 on T1 (one node, TWO legs),
        T1[d,d,x]                                    e3 on T2,  e4 on T3
           |x
        I1[x,y,z] --y-- T2[d,y]     I1, I2 : INTERNAL nodes (no dangling legs)
           |z
        I2[z,w,g] --w-- T3[d,w]     g : the GLUE leg, contracted with the conjugate
           |g                           copy T*.  rank(A) <= g, so the glue dimension
         (glue)                         g is the SPECTRUM KNOB: g=1 rank-one (rho=1),
                                        larger g -> larger intrinsic dim rho=Tr/||.||.

All bonds have size chi, physical legs size d; cores are random ISOMETRIES (seeded), so
rho really tracks g (raw Gaussian cores were dropped: their spectrum is dominated by one
eigenvalue for every g).  Ground truth Tr(A^k) is exact (eigenvalues of the g x g Gram).

WHAT IS MEASURED.
 1. m* VALIDATION (ours = `tree`, the plain order-3-core sketch of alg:trace, one core
    per dimension-tree node): smallest m whose SINGLE estimate is (eps,delta)-accurate.
    thm1 (K=1): swept over eps, delta, d.  thm2 (base K=2): swept over eps, delta, d, K,
    plus a flat-vs-spiked spectrum comparison at the base point (rho moves, rank fixed).
    Plots are linear/linear with the sufficiency-bound shape overlaid (scaled to the
    first measured point); rho = Tr(A)/Tr(A^K)^(1/K), computed from the input spectrum.
 2. COMPARISON, cost-to-(eps,delta): every method pays until one (averaged) estimate is
    (eps,delta)-accurate -- matrix sketch `tree` via its single-shot m*, the scalar
    probes via the smallest averaging batch M (Monte Carlo):
      kron        one Rademacher d-vector PER DANGLING EDGE, then contract (rows=1)
      tt_capped   TT probe over the legs in natural order e1..e4, bond capped at
                  min(R, 4) (the literature defines TT probes for chains only -- this
                  is the mechanical extension, no general-topology version exists)
    (kron_pauli is NOT run here: Noa's partial basis is XX-only, where it has its
    physics meaning -- see exp_xx.py.)
    x-axis cost = EXACT contraction op count (the _ec counter), y log.

COLAB: upload sketches.py next to this file and run the cells in order (the `# %%`
markers split them).  DEVICE is auto-detected.  Outputs land in
results/exp_general_tn/<tag>/ as PNGs + config.json + results.json.
"""

# %% [1] imports & device -------------------------------------------------------------
import os
import json
import math
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketches import (TreeOp, TREE_REGISTRY, estimate_operator, estimate_cost, DTYPE)

DEVICE = "cpu"      # server: 2GB/CUDA-11.4 GPU is unusable for the m^4 intermediates; 125GB RAM CPU is the backend

# %% [2] CONFIG -----------------------------------------------------------------------
CONFIG = {
    "d": 3,                    # isometric cores are asserted: chi <= d
    "chi": 2,                  # FIXED < min(d_list): the d sweep moves ONLY log d_Pi
                               # (rho, rank frozen); also caps the m^4 chi^2 d memory
    "op_seed": 0,
    "seed": 0,
    "dist": "rademacher",

    "eps": 0.3,
    "delta": 0.2,
    "groups": 600,             # estimators per single-shot pass-rate   (Colab: 2000)
    "m_ceiling": 96,           # m* search cap; the tree block holds an m^4 chi^2 d
                               # complex intermediate PER SAMPLE (~16 GB at m=96), and
                               # only censored points ever evaluate the ceiling
    "pool": 400_000,           # Monte-Carlo pool of single estimates for the scalar
                               # probes' batch search                   (Colab: 5e6)
    "min_groups": 40,          # least M-means needed to certify a pass rate; the batch
                               # search is censored beyond M = pool/min_groups

    # sweeps -- each list is one axis, the others sit at "base"
    "base": {"g": 4, "K": 2},
    "g_list": [1, 2, 4],               # comparison only; g <= chi^2 asserted
    "K_list": [2, 3, 4, 5],
    "eps_list": [0.5, 0.4, 0.3, 0.25, 0.2],
    "delta_list": [0.4, 0.3, 0.2, 0.1],
    "d_list": [2, 3, 4, 5],            # chi = 2 <= min(d_list)

    # comparison
    "algos": ["tree", "kron", "tt_capped"],
    "tt_R_list": [2, 4],       # TT-probe bonds tried (capped at 4); cheapest pass kept

    "out_root": os.path.join("results", "exp_general_tn"),
}


# %% [3] operator build + sanity ------------------------------------------------------
def build_op(cfg, g):
    op = TreeOp([1.0] * g, d=cfg["d"], chi=cfg["chi"], seed=cfg["op_seed"],
                device=DEVICE)
    return op


def sanity(cfg):
    print(f"device={DEVICE}  dtype={str(DTYPE).split('.')[-1]}")
    for g in cfg["g_list"]:
        op = build_op(cfg, g)
        rank = int((op.spectrum > 1e-12 * op.spectrum.max()).sum())
        print(f"  g={g:3d}: rank={rank:2d}  rho={op.intrinsic_dim():6.3f}  "
              f"Tr(A^2)={op.true_pk(2):.4f}")


# %% [4] shared (eps,delta) utilities -------------------------------------------------
def make_sketch(cfg, name, m, op):
    return TREE_REGISTRY[name](m, d=op.d, dist=cfg["dist"], device=DEVICE)


@torch.no_grad()
def pass_rate(sk, op, K, eps, M, groups):
    """Fraction of `groups` estimators (each the mean of M single estimates) within eps
    of Tr(A^K).  The (eps,delta) property holds iff this >= 1 - delta."""
    tv = op.true_pk(K)
    # batch capped so the tree block's largest intermediate (m^4 chi^2 d complex128
    # entries per sample) stays under ~32 GB (server: 125 GB RAM, CPU backend)
    cap = max(1, 32_000_000_000 // (16 * sk.m ** 4 * op.chi ** 2 * op.d))
    need, chunk = groups * M, max(1, min(max(sk.chunk(K), M), cap))
    ests, done = [], 0
    while done < need:
        c = min(chunk, need - done)
        ests.append(estimate_operator(sk, [sk.draw(c) for _ in range(K)], op))
        done += c
    r = torch.cat(ests)[:need].reshape(groups, M).mean(dim=1) / tv
    return ((r - 1.0).abs() <= eps).double().mean().item()


def m_star(cfg, name, op, K, eps, delta):
    """Single-shot m*: smallest m in [1, m_ceiling] whose ONE estimate is (eps,delta).
    Doubling search up from m=1 -- the cost grows like m^4, so cheap points must
    never evaluate the ceiling -- then binary search in the last interval.
    None if censored at the ceiling."""
    groups, ceil = cfg["groups"], cfg["m_ceiling"]
    cache = {}

    def pr(m):
        if m not in cache:
            cache[m] = pass_rate(make_sketch(cfg, name, m, op), op, K, eps, 1, groups)
        return cache[m]

    m = 1
    while m < ceil and pr(m) < 1.0 - delta:
        m = min(2 * m, ceil)
    if pr(m) < 1.0 - delta:
        return None
    lo, hi = m // 2 + 1, m
    while lo < hi:
        mid = (lo + hi) // 2
        if pr(mid) >= 1.0 - delta:
            hi = mid
        else:
            lo = mid + 1
    return lo


@torch.no_grad()
def sample_pool(sk, op, K, P):
    """P single Tr(A^K) estimates, drawn once and reused for every batch size below."""
    chunk, ests, done = sk.chunk(K), [], 0
    while done < P:
        c = min(chunk, P - done)
        ests.append(estimate_operator(sk, [sk.draw(c) for _ in range(K)], op))
        done += c
    return torch.cat(ests)[:P]


def min_batch(cfg, sk, op, K, eps, delta):
    """Smallest averaging batch M whose M-sample mean is (eps,delta), measured on ONE
    Monte-Carlo pool of `pool` single estimates: the pass rate at batch M is the
    fraction of the pool//M disjoint M-means within eps of the truth.  Binary search on
    M (the whole search reuses the one pool, so it costs `pool` samples total).
    Censored (None) once fewer than `min_groups` disjoint means remain, i.e. beyond
    M = pool/min_groups."""
    r = sample_pool(sk, op, K, cfg["pool"]) / op.true_pk(K)
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


# %% [5] theorem-validation m* sweeps + the spectrum comparison ----------------------
def _suff(K, eps, delta, d, spectrum):
    """The sufficiency-bound shape (cor:k1-gaussian for K=1, cor:k-gaussian for K>=2)
    up to its constant and N_E: d_Pi = d^4 for the tree (dangling legs d^2, d, d), and
    rho = Tr(A)/Tr(A^K)^(1/K) computed from the exact input spectrum."""
    logs = 4 * math.log(d) + math.log(max(K, 1) / delta)
    if K == 1:
        return logs / eps ** 2
    rho = float(spectrum.sum() / (spectrum ** K).sum() ** (1.0 / K))
    return K ** 2 * max(1.0, rho ** 2 * eps) * logs / eps ** 2


def mstar_part(cfg, folder):
    """m* of the `tree` sketch: thm1 (K=1) vs eps/delta/d, thm2 (base K) vs
    eps/delta/d/K.  Flat spectrum [1]*g throughout; linear/linear plots with the
    sufficiency shape scaled to the first measured point."""
    base, res = cfg["base"], {}
    plans = [
        ("thm1", "eps", cfg["eps_list"]),
        ("thm1", "delta", cfg["delta_list"]),
        ("thm1", "d", cfg["d_list"]),
        ("thm2", "eps", cfg["eps_list"]),
        ("thm2", "delta", cfg["delta_list"]),
        ("thm2", "d", cfg["d_list"]),
        ("thm2", "K", cfg["K_list"]),
    ]
    for tag, param, xs in plans:
        K0 = 1 if tag == "thm1" else base["K"]
        rows = []
        print(f"\n[m* | {tag} | vs {param}]  (base d={cfg['d']} chi={cfg['chi']} "
              f"g={base['g']} K={K0} eps={cfg['eps']} delta={cfg['delta']})")
        for x in xs:
            d = x if param == "d" else cfg["d"]
            K = x if param == "K" else K0
            eps = x if param == "eps" else cfg["eps"]
            delta = x if param == "delta" else cfg["delta"]
            op = TreeOp([1.0] * base["g"], d=d, chi=cfg["chi"],
                        seed=cfg["op_seed"], device=DEVICE)
            ms = m_star(cfg, "tree", op, K, eps, delta)
            rows.append({"x": x, "m_star": ms,
                         "suff": _suff(K, eps, delta, d, op.spectrum)})
            print(f"   {param}={x}: m*={'CENSORED' if ms is None else ms}")
        res[f"{tag}_{param}"] = rows
        fig, ax = plt.subplots(figsize=(8, 5.5))
        gx = [r["x"] for r in rows if r["m_star"] is not None]
        gm = [r["m_star"] for r in rows if r["m_star"] is not None]
        if gx:
            ax.plot(gx, gm, marker="o", ms=8, lw=2, color="tab:blue",
                    label="measured m*")
            s0 = next(r["suff"] for r in rows if r["m_star"] is not None)
            ax.plot([r["x"] for r in rows],
                    [gm[0] * r["suff"] / s0 for r in rows], "--",
                    color="tab:blue", alpha=0.5, label="sufficiency shape")
        for r in rows:
            if r["m_star"] is None:
                ax.scatter([r["x"]], [cfg["m_ceiling"]], marker="x", s=90,
                           color="tab:blue")
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel("m*  (single-shot (eps,delta))", fontsize=12)
        ax.set_title(f"{tag}  |  m* vs {param}  |  K={K0}"
                     + ("  (swept)" if param == "K" else ""), fontsize=12)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
        fig.tight_layout()
        path = os.path.join(folder, f"mstar_{tag}_vs_{param}.png")
        fig.savefig(path, dpi=130); plt.close(fig)
        print(f"   saved {os.path.abspath(path)}")
    return res


def spectrum_part(cfg, folder):
    """Flat vs spiked spectrum at the base point: same g/rank/d_Pi, different rho --
    the distribution is the only thing that moves."""
    K, res = cfg["base"]["K"], {}
    print(f"\n[m* | spectrum: flat vs spiked]  (K={K} eps={cfg['eps']} "
          f"delta={cfg['delta']})")
    for name, spec in (("flat", [1.0, 1.0, 1.0, 1.0]),
                       ("spiked", [1.0, 0.1, 0.1, 0.1])):
        op = TreeOp(spec, d=cfg["d"], chi=cfg["chi"], seed=cfg["op_seed"],
                    device=DEVICE)
        rho = float(op.spectrum.sum() / (op.spectrum ** K).sum() ** (1.0 / K))
        ms = m_star(cfg, "tree", op, K, cfg["eps"], cfg["delta"])
        res[name] = {"spectrum": spec, "rho": rho, "m_star": ms}
        print(f"   {name}: rho={rho:.2f}  m*={'CENSORED' if ms is None else ms}")
    fig, ax = plt.subplots(figsize=(6, 5))
    names = list(res)
    vals = [res[n]["m_star"] or cfg["m_ceiling"] for n in names]
    ax.bar(names, vals, color="tab:blue", width=0.5)
    for i, n in enumerate(names):
        ax.annotate(f"rho={res[n]['rho']:.2f}", (i, vals[i]),
                    ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("m*  (single-shot (eps,delta))", fontsize=12)
    ax.set_title(f"flat vs spiked spectrum  |  K={K} eps={cfg['eps']}", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(folder, "mstar_spectrum_flat_vs_spiked.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"   saved {os.path.abspath(path)}")
    return res


# %% [6] cost-to-(eps,delta) comparison -----------------------------------------------
def compare_point(cfg, op, K):
    """Cost to reach (eps,delta) for every CONFIG algo at one (op, K).  Matrix sketch
    `tree`: single-shot m* -> cost of ONE m* estimate.  Scalar probes: smallest
    averaging batch M -> M * per-sample cost (tt_capped: cheapest passing bond in
    tt_R_list, each capped at the tree's 4 legs)."""
    eps, delta = cfg["eps"], cfg["delta"]
    out = {}
    for name in cfg["algos"]:
        if name == "tree":
            ms = m_star(cfg, name, op, K, eps, delta)
            out[name] = (None if ms is None else
                         {"m": ms, "cost": estimate_cost(
                             make_sketch(cfg, name, ms, op), op, K)})
        elif name == "tt_capped":
            best = None
            for R in cfg["tt_R_list"]:
                sk = make_sketch(cfg, name, R, op)
                M = min_batch(cfg, sk, op, K, eps, delta)
                if M is not None:
                    c = M * estimate_cost(sk, op, K)
                    if best is None or c < best["cost"]:
                        best = {"R": R, "M": M, "cost": c}
            out[name] = best
        else:                                   # kron (rows = 1)
            sk = make_sketch(cfg, name, 1, op)
            M = min_batch(cfg, sk, op, K, eps, delta)
            out[name] = (None if M is None else
                         {"M": M, "cost": M * estimate_cost(sk, op, K)})
    return out


def comparison(cfg, folder):
    base, res = cfg["base"], {}
    colors = {"tree": "tab:blue", "kron": "tab:green", "tt_capped": "tab:orange"}
    for param, xs in (("g", cfg["g_list"]), ("K", cfg["K_list"])):
        rows = []
        print(f"\n[comparison vs {param}]  cost to (eps={cfg['eps']}, "
              f"delta={cfg['delta']})")
        for x in xs:
            g, K = (x, base["K"]) if param == "g" else (base["g"], x)
            op = build_op(cfg, g)
            r = compare_point(cfg, op, K)
            rows.append({"x": x, **{k: v for k, v in r.items()}})
            msg = "  ".join(
                f"{a}={'cens' if r[a] is None else format(r[a]['cost'], '.2e')}"
                for a in cfg["algos"])
            print(f"   {param}={x}: {msg}")
        res[param] = rows
        fig, ax = plt.subplots(figsize=(9, 6))
        for a in cfg["algos"]:
            gx = [r["x"] for r in rows if r[a] is not None]
            gy = [r[a]["cost"] for r in rows if r[a] is not None]
            if gx:
                ax.plot(gx, gy, marker="o", ms=8, lw=2, color=colors[a], label=a)
            for r in rows:
                if r[a] is None:
                    ax.scatter([r["x"]], [1], marker="x", s=90, color=colors[a])
        ax.set_yscale("log")
        ax.set_xlabel(f"{param}" + ("  (glue = rho knob)" if param == "g" else ""),
                      fontsize=12)
        ax.set_ylabel("contraction cost to reach (eps,delta)", fontsize=12)
        ax.set_title(f"general tree operator  |  cost vs {param}  |  "
                     f"eps={cfg['eps']} delta={cfg['delta']}", fontsize=12)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=11)
        fig.tight_layout()
        path = os.path.join(folder, f"cost_vs_{param}.png")
        fig.savefig(path, dpi=130); plt.close(fig)
        print(f"   saved {os.path.abspath(path)}")
    return res


# %% [7] main -------------------------------------------------------------------------
def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    torch.manual_seed(cfg["seed"])
    tag = f"d{cfg['d']}_chi{cfg['chi']}_eps{cfg['eps']}_delta{cfg['delta']}"
    folder = os.path.join(cfg["out_root"], tag)
    os.makedirs(folder, exist_ok=True)
    json.dump(cfg, open(os.path.join(folder, "config.json"), "w"), indent=2)
    t0 = time.time()
    sanity(cfg)
    res = {"mstar": mstar_part(cfg, folder),
           "spectrum": spectrum_part(cfg, folder),
           "compare": comparison(cfg, folder)}
    json.dump(res, open(os.path.join(folder, "results.json"), "w"), indent=2,
              default=str)
    print(f"\ndone in {time.time() - t0:.0f}s  ->  {os.path.abspath(folder)}")
    return res


if __name__ == "__main__":
    main()
