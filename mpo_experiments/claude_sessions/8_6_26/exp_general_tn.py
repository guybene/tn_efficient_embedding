# %% [markdown]
# EXP GENERAL TN -- the general-framework experiment (theorem validation + race)
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
the input spectrum IS A's spectrum.  Ground truth Tr(A^k) is exact.

TWO STEPS, two functions called from main (comment either out):
  run(cfg, folder)   computes and SAVES one pickle per smallest unit of work; a
                     finished file is skipped on rerun, so a crash costs only the
                     unit that crashed.
  plot(cfg, folder)  reads ONLY the pickles and draws the figures + summary.json.

FOLDER HIERARCHY  results/exp_general_tn/<tag>/
  partA_theorem/<thm1|thm2|spectrum>/vs_<param>/<param>=<value>/tree.pkl
  partB_race/vs_<param>/<param>=<value>/<method>/size=<s>.pkl
  plots/*.png, plots/summary.json, config.json

PART A -- THEOREM (ours only, the `tree` sketch): smallest m whose SINGLE estimate
  is (eps,delta)-accurate.  thm1 (K=1) and thm2 (K=base) vs eps, delta, d, K
  (K only for thm2), flat spectrum; plus flat vs spiked at the base point.
  The pickle is updated after EVERY pass-rate evaluation of the m* search, so a
  crash mid-search resumes from the last m tried.

PART B -- RACE (all methods): contraction cost (exact op count) to reach (eps,delta)
  vs g (the spectrum knob) and vs d (d is NOT raceable on the XX MPO: d=2 there).
  Every method runs a fixed list of sizes (tree: m, kressner: ell, tt_capped: bond R,
  kron: rank-1); each size gets its smallest averaging batch M by an ADAPTIVE search
  (trials M-means per candidate M, samples drawn lazily, doubling then bisection);
  cost = M * per-sample cost; the plot keeps the cheapest size per method.
  One pickle per (parameter, value, method, size): a crash in one method or one
  size loses only that file.

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

from sketches import (TreeOp, TREE_REGISTRY, estimate_operator, estimate_cost, DTYPE)

try:
    import resource                      # peak RSS (Linux server); absent on Windows
except ImportError:
    resource = None

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
    "groups": 600,             # estimators per single-shot pass-rate
    "m_ceiling": 96,           # m* search cap; the tree block holds an m^4 chi^2 d
                               # complex intermediate PER SAMPLE (~16 GB at m=96), and
                               # only censored points ever evaluate the ceiling
    "trials": 100,             # M-means per candidate M in the batch search; pass iff
                               # >= (1-delta)*trials of them are within eps
    "pool": 400_000,           # cap on single estimates per unit: the batch search is
                               # censored beyond M = pool/trials

    "base": {"g": 4, "K": 2},  # the base point; every sweep moves one parameter

    # PART A (theorem, ours only): m* vs each parameter; K is swept for thm2 only
    "theorem_params": ["eps", "delta", "d", "K"],
    "eps_list": [0.5, 0.4, 0.3, 0.25, 0.2],
    "delta_list": [0.4, 0.3, 0.2, 0.1],
    "d_list": [2, 3, 4, 5],            # chi = 2 <= min(d_list)
    "K_list": [2, 3, 4, 5, 6, 7],        # thm2 only; K=1 is thm1
    "spectra": {"flat": [1.0, 1.0, 1.0, 1.0],      # same g/rank/d_Pi, different rho
                "spiked": [1.0, 0.1, 0.1, 0.1]},

    # PART B (race, all methods): cost to (eps,delta) vs each parameter, at base K
    "race_params": ["g", "d"],         # d raced HERE: the XX MPO has d = 2 fixed
    "g_list": [1, 2, 4],               # g <= chi^2 asserted
    "algos": ["tree", "kron", "tt_capped", "kressner"],
    "tree_m_list": [2, 4, 8, 16, 32],  # our sizes tried, each with its own min batch M
    "kressner_m_list": [2, 4, 8],  # ell stacked probes per copy, same (ell, M) search;
                               # probe rank fixed by Thm 9
                               # (bond = n_legs p, p = 2 ceil(log(1/delta)/2), see make_sketch)
    "tt_R_list": [2, 4],       # TT-probe bonds tried (capped at 4)

    "out_root": os.path.join("results", "exp_general_tn"),
}


# %% [3] operator + sketch builders ---------------------------------------------------
def build_op(cfg, g, d=None, spectrum=None):
    return TreeOp([1.0] * g if spectrum is None else spectrum,
                  d=cfg["d"] if d is None else d, chi=cfg["chi"],
                  seed=cfg["op_seed"], device=DEVICE)


def make_sketch(cfg, name, m, op):
    if name == "kressner":     # bujanovic2026tt Thm 9 rank: N p, p = 2 ceil(log(1/delta)/2)
        p = 2 * math.ceil(0.5 * math.log(1.0 / cfg["delta"]))
        return TREE_REGISTRY[name](m, d=op.d, dist=cfg["dist"], device=DEVICE,
                                   bond=op.n_legs * p)
    return TREE_REGISTRY[name](m, d=op.d, dist=cfg["dist"], device=DEVICE)


def sizes_for(cfg, name):
    """The sizes a method runs in the race (tree: m, kressner: ell, tt_capped: R,
    kron: rank-1, size ignored)."""
    if name == "tree":
        return cfg["tree_m_list"]
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
    """n single Tr(A^K) estimates (one per sample), chunked so the tree block's
    m^4 chi^2 d complex intermediate stays under ~32 GB."""
    cap = max(1, 32_000_000_000 // (16 * sk.m ** 4 * op.chi ** 2 * op.d))
    chunk, ests, done = max(1, min(sk.chunk(K), cap)), [], 0
    while done < n:
        c = min(chunk, n - done)
        ests.append(estimate_operator(sk, [sk.draw(c) for _ in range(K)], op))
        done += c
    return torch.cat(ests)[:n].real.cpu()


def min_batch(sk, op, K, tv, eps, delta, trials, pool_cap):
    """Smallest averaging batch M whose M-sample mean is within eps of the truth tv
    with probability >= 1-delta, judged on `trials` disjoint M-means (pass iff at
    least (1-delta)*trials of them are within eps).  ADAPTIVE: candidate M needs
    trials*M single estimates, drawn lazily into one growing pool that every
    candidate reuses; doubling search from M=1, then bisection in the last
    interval.  M never sets a batch size: samples are drawn in memory-capped chunks
    (single_estimates), M only slices the pool.  Censored (None) when trials*M
    would exceed pool_cap.  Returns (M, pool, trace of (M, pass_rate))."""
    pool, trace = torch.empty(0, dtype=torch.float64), []

    def ok(M):
        nonlocal pool
        need = trials * M
        if len(pool) < need:
            pool = torch.cat([pool, single_estimates(sk, op, K, need - len(pool))])
        means = pool[:need].reshape(trials, M).mean(dim=1) / tv
        rate = ((means - 1.0).abs() <= eps).double().mean().item()
        trace.append((M, rate))
        return rate >= 1.0 - delta

    M_max = pool_cap // trials
    M = 1
    while not ok(M):
        if M >= M_max:
            return None, pool, trace
        M = min(2 * M, M_max)
    lo, hi = M // 2 + 1, M
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo, pool, trace


# %% [6] PART A units: one m* search per pickle, resumable per evaluation -------------
def _suff(K, eps, delta, d, spectrum):
    """The sufficiency-bound shape (cor:k1-gaussian for K=1, cor:k-gaussian for K>=2)
    up to its constant and N_E: d_Pi = d^4 for the tree (dangling legs d^2, d, d), and
    rho = Tr(A)/Tr(A^K)^(1/K) computed from the exact input spectrum."""
    logs = 4 * math.log(d) + math.log(max(K, 1) / delta)
    if K == 1:
        return logs / eps ** 2
    rho = float(spectrum.sum() / (spectrum ** K).sum() ** (1.0 / K))
    return K ** 2 * max(1.0, rho ** 2 * eps) * logs / eps ** 2


def run_mstar_unit(cfg, path, op, K, eps, delta, context):
    """Single-shot m* of the `tree` sketch at one point: smallest m in [1, m_ceiling]
    whose ONE estimate is (eps,delta).  Doubling search up from m=1 (the cost grows
    like m^4, so cheap points never evaluate the ceiling), then binary search in the
    last interval; None if censored.  EVERY pass-rate evaluation is written to the
    pickle before the next one starts, so a rerun resumes the search."""
    rec = load_pkl(path)
    if rec is not None and rec["done"]:
        return rec
    if rec is None:
        rec = {**context, "K": K, "eps": eps, "delta": delta,
               "true": op.true_pk(K), "spectrum": op.spectrum.cpu().numpy(),
               "suff": _suff(K, eps, delta, op.d, op.spectrum),
               "evals": {}, "m_star": None, "seconds": 0.0, "peak_rss_mb": None,
               "done": False}
    groups, ceil = cfg["groups"], cfg["m_ceiling"]

    def pr(m):
        if m not in rec["evals"]:
            t0 = time.time()
            seed_for(cfg, path, m)
            est = single_estimates(make_sketch(cfg, "tree", m, op), op, K, groups)
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
    base = cfg["base"]
    for tag in ("thm1", "thm2"):
        K0 = 1 if tag == "thm1" else base["K"]
        for param in cfg["theorem_params"]:
            if param == "K" and tag == "thm1":
                continue
            for x in cfg[f"{param}_list"]:
                d = x if param == "d" else cfg["d"]
                K = x if param == "K" else K0
                eps = x if param == "eps" else cfg["eps"]
                delta = x if param == "delta" else cfg["delta"]
                path = os.path.join(folder, "partA_theorem", tag, f"vs_{param}",
                                    f"{param}={x}", "tree.pkl")
                op = build_op(cfg, base["g"], d=d)
                rec = run_mstar_unit(cfg, path, op, K, eps, delta,
                                     {"part": "A", "theorem": tag, "sweep": param,
                                      "x": x, "d": d, "g": base["g"], "spectrum_name": "flat"})
                print(f"  [A {tag} vs {param}={x}] K={K} m*="
                      f"{'CENSORED' if rec['m_star'] is None else rec['m_star']}  "
                      f"({rec['seconds']:.0f}s)")
    K = base["K"]
    for name, spec in cfg["spectra"].items():
        path = os.path.join(folder, "partA_theorem", "spectrum", "vs_spectrum",
                            f"spectrum={name}", "tree.pkl")
        op = build_op(cfg, len(spec), spectrum=spec)
        rec = run_mstar_unit(cfg, path, op, K, cfg["eps"], cfg["delta"],
                             {"part": "A", "theorem": "spectrum", "sweep": "spectrum",
                              "x": name, "d": cfg["d"], "g": len(spec),
                              "spectrum_name": name})
        print(f"  [A spectrum {name}] K={K} m*="
              f"{'CENSORED' if rec['m_star'] is None else rec['m_star']}  "
              f"({rec['seconds']:.0f}s)")


# %% [7] PART B units: one (parameter, value, method, size) per pickle ----------------
def run_race_unit(cfg, path, op, K, algo, size, context):
    """One method at one size at one point: adaptive search for the smallest
    averaging batch M (min_batch), cost = M * per-sample op count.  Written once,
    complete; a crash mid-search leaves nothing and the unit reruns alone."""
    rec = load_pkl(path)
    if rec is not None and rec["done"]:
        return rec
    t0 = time.time()
    seed_for(cfg, path)
    sk = make_sketch(cfg, algo, size, op)
    tv = op.true_pk(K)
    M, pool, trace = min_batch(sk, op, K, tv, cfg["eps"], cfg["delta"],
                               cfg["trials"], cfg["pool"])
    cost_per = estimate_cost(sk, op, K)
    rec = {**context, "K": K, "eps": cfg["eps"], "delta": cfg["delta"], "algo": algo,
           "size": size, "rows": sk.sketch_size(),
           "bond": getattr(sk, "bond", None), "true": tv, "pool": pool.numpy(),
           "trials": cfg["trials"], "search": trace,
           "M": M, "cost_per_sample": cost_per,
           "cost": None if M is None else M * cost_per,
           "seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb(), "done": True}
    save_pkl(path, rec)
    return rec


def run_part_b(cfg, folder):
    base = cfg["base"]
    for param in cfg["race_params"]:
        for x in cfg[f"{param}_list"]:
            g = x if param == "g" else base["g"]
            d = x if param == "d" else cfg["d"]
            op = build_op(cfg, g, d=d)
            for algo in cfg["algos"]:
                for size in sizes_for(cfg, algo):
                    path = os.path.join(folder, "partB_race", f"vs_{param}",
                                        f"{param}={x}", algo, f"size={size}.pkl")
                    rec = run_race_unit(cfg, path, op, base["K"], algo, size,
                                        {"part": "B", "sweep": param, "x": x,
                                         "d": d, "g": g})
                    print(f"  [B vs {param}={x}] {algo:10s} size={size:3d}  M="
                          f"{'cens' if rec['M'] is None else rec['M']:>6}  cost="
                          f"{'cens' if rec['cost'] is None else format(rec['cost'], '.2e')}"
                          f"  ({rec['seconds']:.0f}s)")


# %% [8] STEP 1: run ------------------------------------------------------------------
def run(cfg, folder):
    print(f"device={DEVICE}  dtype={str(DTYPE).split('.')[-1]}")
    for g in cfg["g_list"]:
        op = build_op(cfg, g)
        rank = int((op.spectrum > 1e-12 * op.spectrum.max()).sum())
        print(f"  g={g:3d}: rank={rank:2d}  rho={op.intrinsic_dim():6.3f}  "
              f"Tr(A^2)={op.true_pk(2):.4f}")
    print("\n[PART A] theorem: m* of ours")
    run_part_a(cfg, folder)
    print("\n[PART B] race: cost to (eps,delta), all methods")
    run_part_b(cfg, folder)


# %% [9] STEP 2: plot (reads only the pickles) ---------------------------------------
def plot(cfg, folder):
    plots = os.path.join(folder, "plots")
    os.makedirs(_lp(plots), exist_ok=True)
    summary = {"partA": {}, "partB": {}}
    # ---- PART A: m* vs parameter, sufficiency shape scaled to the first point ----
    for tag in ("thm1", "thm2"):
        for sub in subdirs(os.path.join(folder, "partA_theorem", tag)):
            recs = sorted(load_units(os.path.join(folder, "partA_theorem", tag, sub)),
                          key=lambda r: r["x"])
            if not recs:
                continue
            param = recs[0]["sweep"]
            summary["partA"][f"{tag}_vs_{param}"] = [
                {"x": r["x"], "m_star": r["m_star"], "suff": r["suff"],
                 "seconds": r["seconds"]} for r in recs]
            fig, ax = plt.subplots(figsize=(8, 5.5))
            gx = [r["x"] for r in recs if r["m_star"] is not None]
            gm = [r["m_star"] for r in recs if r["m_star"] is not None]
            if gx:
                ax.plot(gx, gm, marker="o", ms=8, lw=2, color="tab:blue",
                        label="measured m*")
                s0 = next(r["suff"] for r in recs if r["m_star"] is not None)
                ax.plot([r["x"] for r in recs], [gm[0] * r["suff"] / s0 for r in recs],
                        "--", color="tab:blue", alpha=0.5, label="sufficiency shape")
            for r in recs:
                if r["m_star"] is None:
                    ax.scatter([r["x"]], [cfg["m_ceiling"]], marker="x", s=90,
                               color="tab:blue")
            ax.set_xlabel(param, fontsize=12)
            ax.set_ylabel("m*  (single-shot (eps,delta))", fontsize=12)
            ax.set_title(f"{tag}  |  m* vs {param}  |  "
                         + ("K swept" if param == "K" else f"K={recs[0]['K']}"),
                         fontsize=12)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
            fig.tight_layout()
            path = os.path.join(plots, f"mstar_{tag}_vs_{param}.png")
            fig.savefig(_lp(path), dpi=130); plt.close(fig)
            print(f"  saved {path}")
    recs = {r["x"]: r for r in load_units(os.path.join(folder, "partA_theorem", "spectrum"))}
    rows = [recs[n] for n in cfg["spectra"] if n in recs]
    if rows:
        summary["partA"]["spectrum"] = [{"x": r["x"], "m_star": r["m_star"],
                                         "spectrum": r["spectrum"].tolist()} for r in rows]
        fig, ax = plt.subplots(figsize=(6, 5))
        vals = [r["m_star"] or cfg["m_ceiling"] for r in rows]
        ax.bar([r["x"] for r in rows], vals, color="tab:blue", width=0.5)
        for i, r in enumerate(rows):
            K = r["K"]
            rho = float(r["spectrum"].sum() / (r["spectrum"] ** K).sum() ** (1.0 / K))
            ax.annotate(f"rho={rho:.2f}" + ("" if r["m_star"] else " (cens)"),
                        (i, vals[i]), ha="center", va="bottom", fontsize=11)
        ax.set_ylabel("m*  (single-shot (eps,delta))", fontsize=12)
        ax.set_title(f"flat vs spiked spectrum  |  K={rows[0]['K']} eps={cfg['eps']}",
                     fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plots, "mstar_spectrum_flat_vs_spiked.png")
        fig.savefig(_lp(path), dpi=130); plt.close(fig)
        print(f"  saved {path}")
    # ---- PART B: cheapest certified size per method, cost vs parameter, log y ----
    colors = {"tree": "tab:blue", "kron": "tab:green", "tt_capped": "tab:orange",
              "kressner": "tab:red"}
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
        ax.set_xlabel(param + ("  (glue = rho knob)" if param == "g" else ""), fontsize=12)
        ax.set_ylabel("contraction cost to reach (eps,delta)", fontsize=12)
        ax.set_title(f"general tree operator  |  cost vs {param}  |  K={recs[0]['K']} "
                     f"eps={cfg['eps']} delta={cfg['delta']}", fontsize=12)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=11)
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
    tag = f"d{cfg['d']}_chi{cfg['chi']}_eps{cfg['eps']}_delta{cfg['delta']}"
    folder = os.path.join(cfg["out_root"], tag)
    os.makedirs(_lp(folder), exist_ok=True)
    json.dump(cfg, open(_lp(os.path.join(folder, "config.json")), "w"), indent=2)
    t0 = time.time()
    run(cfg, folder)
    plot(cfg, folder)
    print(f"\ndone in {time.time() - t0:.0f}s  ->  {os.path.abspath(folder)}")


if __name__ == "__main__":
    main()
