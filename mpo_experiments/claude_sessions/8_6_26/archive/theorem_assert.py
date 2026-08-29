"""
EXP 1 -- assert_theorem: validate the (eps,delta) embedding theorem for the TTM sketch ALONE.

First a hard ASSERT that scaling A by a scalar leaves everything unchanged (rho = Tr(A)/||A||_2,
and hence m*, is scale-free -- est/Tr(A^K) cancels the scalar exactly).  Then sweep each parameter
(N, d, K, eps, delta, and the scalar C) one at a time, measure the single-shot m* (smallest m for
which ONE ttm estimate is (eps,delta)-accurate), and overlay the theorem's predicted scaling so we
can see m* "act like the theorem".  identity (rho=d^N) and all_ones (rho=1) are the two rho points.

Run:  python theorem_assert.py   (or import and call assert_theorem()).
Needs sketches.py beside it.  Figures -> results/assert_theorem/.

The (eps,delta) test is DIRECT: draw many estimators, take the fraction within eps of Tr(A^K); it
passes iff that fraction >= 1 - delta.  Here single-shot (M=1).  (No variance, no Chebyshev.)
"""
import os
import json
import math

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketches import REGISTRY, estimate, make_A, true_value, DTYPE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = {
    "d": 2,
    "dist": "rademacher",
    "trials": 3000,            # estimators drawn to measure each single-shot pass rate
    "m_ceiling": 800,          # upper bound on the binary search for m* (raise on GPU; m^3 cost)
    "seed": 0,
    "spectra": ["identity", "all_ones"],

    # sweeps: each varies ONE axis, the others fixed at 'theorem_base'.  base eps/delta kept loose
    # so the single-shot m* stays resolvable below m_ceiling (the pass test is stringent).
    "theorem_base": {"N": 4, "d": 2, "K": 2, "eps": 0.2, "delta": 0.2},
    "theorem_sweeps": {
        "N":     [3, 4, 5, 6],
        "d":     [2, 3, 4],
        "K":     [1, 2, 3],
        "eps":   [0.5, 0.4, 0.3, 0.25],
        "delta": [0.4, 0.3, 0.2, 0.1],
        "C":     [1, 2, 4, 8, 16],     # scalar multiply A->C*A: rho scale-free, so m* MUST be flat
    },
}


# ============================================================================
# (eps,delta) measurement + single-shot m*
# ============================================================================
@torch.no_grad()
def pass_rate(sketch, A, tv, K, eps, M, groups):
    """DIRECT (eps,delta) measurement: draw `groups` estimators, each the average of `M` single
    Tr(A^K) estimates, and return the fraction within eps of the truth, P(|avg/Tr - 1| <= eps).
    The (eps,delta) property holds iff this >= 1 - delta.  Heavy tails self-handle (a mostly-zero
    estimator has |r-1|~1 > eps so it just fails)."""
    need = groups * M
    chunk = max(sketch.chunk(K), M)
    ests, done = [], 0
    while done < need:
        c = min(chunk, need - done)
        facs = [sketch.draw(c) for _ in range(K)]
        ests.append(estimate(sketch, facs, A))
        done += c
    r = torch.cat(ests)[:need].reshape(groups, M).mean(dim=1) / tv
    return ((r - 1.0).abs() <= eps).double().mean().item()


def ttm_m_star(d, N, kind, K, eps, delta, cfg, scale=1.0):
    """Smallest m in [1, m_ceiling] for which a SINGLE ttm estimate is (eps,delta): pass rate
    >= 1 - delta.  Binary search (pass rate increases with m).  Returns (m* or None if censored,
    pass rate at the chosen/ceiling m).  `scale` multiplies the operator A -> C*A: rho is scale-free
    so m* MUST be flat in C -- the control sweep that proves scale-invariance as a graph."""
    A = make_A(kind, d, N, DEVICE)
    tv = true_value(kind, d, N, K)
    if scale != 1.0:
        A = A.clone(); A[0] = A[0] * scale       # scale whole operator by C
        tv = tv * scale ** K                     # Tr((C A)^K) = C^K Tr(A^K)
    ceil, groups = cfg["m_ceiling"], cfg["trials"]
    cache = {}

    def pr(m):
        if m not in cache:
            sk = REGISTRY["ttm"](d, N, m, cfg["dist"], DEVICE)
            cache[m] = pass_rate(sk, A, tv, K, eps, 1, groups)
        return cache[m]

    if pr(ceil) < 1.0 - delta:
        return None, cache[ceil]                 # even the ceiling fails -> censored
    lo, hi = 1, ceil
    while lo < hi:
        mid = (lo + hi) // 2
        if pr(mid) >= 1.0 - delta:
            hi = mid
        else:
            lo = mid + 1
    return lo, cache[lo]


# ============================================================================
# scale-invariance assert
# ============================================================================
@torch.no_grad()
def assert_scale_invariance(cfg):
    """HARD ASSERT: multiplying A by a scalar s leaves est/Tr(A^K) unchanged sample-for-sample
    (est is K-linear in A, so est(sA)=s^K est(A) and Tr((sA)^K)=s^K Tr(A^K) cancel exactly).
    With the same seed the relative estimates are bit-identical across s -> m* is scale-free,
    exactly as rho = Tr(A)/||A||_2 demands.  (The C sweep below is the graphed version of this.)"""
    print("assert_theorem: scale-invariance (rho is scale-free)")
    d, scalars, worst = cfg["d"], [1, 2, 4, 8, 16], 0.0
    for kind in cfg["spectra"]:
        for N, K in [(4, 1), (4, 3), (5, 2)]:
            tv = true_value(kind, d, N, K)
            base = None
            for s in scalars:
                A = make_A(kind, d, N, DEVICE).clone()
                A[0] = A[0] * s                              # scale the whole operator by s
                sk = REGISTRY["ttm"](d, N, 9, cfg["dist"], DEVICE)
                torch.manual_seed(cfg["seed"])               # identical draws across s
                facs = [sk.draw(2000) for _ in range(K)]
                r = estimate(sk, facs, A) / (tv * s ** K)
                if base is None:
                    base = r
                else:
                    worst = max(worst, (r - base).abs().max().item())
    print(f"  worst |r(s)-r(1)| over all draws = {worst:.2e}  -> "
          f"{'SCALE-INVARIANT' if worst < 1e-9 else 'FAIL'}")
    assert worst < 1e-9, f"scale-invariance FAILED (worst={worst:.2e})"


# ============================================================================
# the sweep
# ============================================================================
def assert_theorem(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    torch.manual_seed(cfg["seed"])
    folder = os.path.join("results", "assert_theorem")
    os.makedirs(folder, exist_ok=True)
    print(f"\n=== EXP 1: assert_theorem (device={DEVICE}, dtype={str(DTYPE).split('.')[-1]}) ===")
    assert_scale_invariance(cfg)

    base = cfg["theorem_base"]
    results = {}
    for param, xs in cfg["theorem_sweeps"].items():
        print(f"\n[sweep {param}]  (base {base})")
        results[param] = {"xs": xs}
        for kind in cfg["spectra"]:
            ms, cens = [], []
            for x in xs:
                # "C" scales the operator (A -> C*A) at the base params; every other sweep moves
                # one of the actual theorem parameters and leaves the scalar at 1.
                p = dict(base)
                scale = x if param == "C" else 1.0
                if param != "C":
                    p[param] = x
                mstar, pr = ttm_m_star(p["d"], p["N"], kind, p["K"], p["eps"], p["delta"], cfg, scale=scale)
                ms.append(mstar if mstar is not None else cfg["m_ceiling"])
                cens.append(mstar is None)
                tag = " CENSORED" if mstar is None else ""
                print(f"   {kind:9s} {param}={x}: m*={'>%d' % cfg['m_ceiling'] if mstar is None else mstar}"
                      f"  pass={pr:.3f} (need {1 - p['delta']:.2f}){tag}")
            results[param][kind] = {"m_star": ms, "censored": cens}
        _plot_theorem(param, xs, results[param], cfg, folder)

    json.dump({"config": cfg, "results": results},
              open(os.path.join(folder, "theorem_results.json"), "w"), indent=2)
    print(f"\nsaved -> {os.path.abspath(folder)}")
    return results


# ============================================================================
# plotting
# ============================================================================
def _theory_curve(param, xs):
    """The theorem's predicted m* scaling in each swept parameter (up to a constant we fit):
    Thm1/3:  m ~ (N^2 log d + N log(k/delta)) * k^2 * max(1, rho^2 eps) / eps^2.  This is a
    SUFFICIENCY upper bound, tight only for the adversarial direction (all_ones); for benign
    identity, measured m* may sit well under it and even trend differently -- loose, NOT a violation."""
    f = {
        "N":     lambda x: x ** 2,
        "d":     lambda x: math.log(x),
        "K":     lambda x: x ** 2,
        "eps":   lambda x: 1.0 / x ** 2,
        "delta": lambda x: math.log(1.0 / x),
        "C":     lambda x: 1.0,            # scale-free: m* independent of the scalar -> flat
    }[param]
    return [f(x) for x in xs]


# LOG-LOG only for the power-law exponents we want to read as a slope:
#   N (~N^2), K (~k^2), eps (~1/eps^2) -> slope = exponent.  Everything else is LINEAR axes
#   (no log x-scale): d, delta, and the scalar sweep C (which must be FLAT, scale-invariant).
# SLOPE is the log-log exponent, set ONLY for the log-log params (used for the fitted-slope readout).
_AXES = {"N": ("log", "log"), "eps": ("log", "log"), "K": ("log", "log"),
         "d": ("linear", "linear"), "delta": ("linear", "linear"), "C": ("linear", "linear")}
_THEORY_SLOPE = {"N": 2.0, "eps": -2.0, "K": 2.0, "d": None, "delta": None, "C": None}
_THEORY_SHAPE = {"N": "~N^2", "d": "~log d", "K": "~k^2", "eps": "~1/eps^2",
                 "delta": "~log(1/delta)", "C": "flat (scale-invariant)"}


def _plot_theorem(param, xs, res, cfg, folder):
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"identity": "tab:blue", "all_ones": "tab:orange"}
    slope = _THEORY_SLOPE[param]
    for kind in cfg["spectra"]:
        color = colors.get(kind, "tab:gray")
        ms = res[kind]["m_star"]; cens = res[kind]["censored"]
        good_x = [x for x, c in zip(xs, cens) if not c]
        good_m = [m for m, c in zip(ms, cens) if not c]
        # measured points; for log-log params annotate the FITTED slope to compare to theory
        fit_txt = ""
        if slope is not None and len(good_x) >= 2:
            p = float(np.polyfit(np.log(good_x), np.log(good_m), 1)[0])
            fit_txt = f"  [fit slope {p:+.2f} vs theory {slope:+.0f}]"
        ax.plot(good_x, good_m, marker="o", ms=8, lw=2, color=color,
                label=f"{kind} measured m*{fit_txt}")
        for x, m, c in zip(xs, ms, cens):           # censored -> x at the ceiling
            if c:
                ax.scatter([x], [m], marker="x", s=90, color=color)
        # reference anchored at this operator's first resolved point: slope line (log-log params)
        # or the predicted shape (linear params).
        if good_x:
            ref = ([good_m[0] * (x / good_x[0]) ** slope for x in xs] if slope is not None
                   else [good_m[0] * _theory_curve(param, [x])[0] / _theory_curve(param, [good_x[0]])[0]
                         for x in xs])
            lab = (f"{kind} theorem ref (slope {slope:+.0f})" if slope is not None
                   else f"{kind} theorem shape {_THEORY_SHAPE[param]}")
            ax.plot(xs, ref, "--", color=color, alpha=0.55, label=lab)
    xscale, yscale = _AXES[param]
    ax.set_xscale(xscale); ax.set_yscale(yscale)
    if xscale == "log":
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("C  (operator scaled A -> C*A)" if param == "C" else param, fontsize=12)
    ax.set_ylabel("m*  (single-shot (eps,delta))", fontsize=12)
    ax.set_title(f"TTM m* vs {param}    theorem {_THEORY_SHAPE[param]}    "
                 f"({xscale}-x/{yscale}-y; x = censored > ceiling)", fontsize=13)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(folder, f"mstar_vs_{param}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"   saved {os.path.abspath(path)}")


if __name__ == "__main__":
    assert_theorem()
