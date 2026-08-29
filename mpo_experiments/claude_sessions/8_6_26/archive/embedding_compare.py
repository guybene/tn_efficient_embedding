"""
EXP 2 -- compare_to_tt: cost comparison TTM vs TT vs KRON, ANCHORED on TT.

TT runs first at its maximal useful bond min(m,N)=N, and we find the averaging batch M = n_TT it
needs for (eps,delta).  Then TTM reuses that SAME M and we push its m DOWN from N: the smallest m
that still reaches (eps,delta) with M samples is its operating point.  Since cost ~ M*m^3, a smaller
m undercuts TT -- the goal is to SUPPRESS TT.  KRON is the cheap scalar baseline.  A sketch that
cannot reach (eps,delta) within batch_ceiling averages is CENSORED (this happens to kron on the
rank-1 direction for free -- a mostly-zero estimator's averages stay far from the truth, so it fails).

Run:  python embedding_compare.py   (or import and call compare_to_tt()).
Needs sketches.py beside it.  Figures -> results/compare_embeddings/.

The (eps,delta) test is DIRECT: draw many estimators, take the fraction within eps of Tr(A^K); it
passes iff that fraction >= 1 - delta.  An averaged estimator uses the mean of M single estimates.
"""
import os
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketches import REGISTRY, estimate, make_A, true_value

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = {
    "d": 2,
    "eps": 0.25,
    "delta": 0.1,
    "dist": "rademacher",
    "seed": 0,
    "spectra": ["identity", "all_ones"],

    # averaging search: smallest batch M whose M-sample average is (eps,delta)
    "groups": 600,             # independent M-averages drawn to estimate the pass rate
    "batch_ceiling": 200000,   # cap on M (censor a sketch that needs more than this)

    "compare_K": {"N": 5, "Ks": [1, 2, 3, 4]},     # cost vs K at this N
    "compare_N": {"K": 2, "Ns": [4, 5, 6, 7]},     # cost vs N at this K
}


# ============================================================================
# (eps,delta) measurement + batch search
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


def min_batch(make_sketch, d, N, K, kind, cfg):
    """Smallest averaging batch M in [1, batch_ceiling] whose M-sample average is (eps,delta):
    pass rate >= 1 - delta.  Binary search (the average tightens as M grows, so the pass rate
    increases monotonically).  Returns M, or None if even batch_ceiling fails (censored)."""
    A = make_A(kind, d, N, DEVICE)
    tv = true_value(kind, d, N, K)
    eps, delta, groups, ceil = cfg["eps"], cfg["delta"], cfg["groups"], cfg["batch_ceiling"]

    def ok(M):
        return pass_rate(make_sketch(), A, tv, K, eps, M, groups) >= 1.0 - delta

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


# ============================================================================
# the three sketches' operating points at one (kind, N, K)
# ============================================================================
def _tt_anchor(d, N, K, kind, cfg):
    """TT at its maximal useful bond min(m,N)=N.  M = smallest batch reaching (eps,delta).
    Returns (M, cost_tt, censored)."""
    M = min_batch(lambda: REGISTRY["tt"](d, N, N, cfg["dist"], DEVICE), d, N, K, kind, cfg)
    if M is None:
        return None, None, True
    return M, M * REGISTRY["tt"](d, N, N).cost(K), False


def _ttm_push_down(d, N, K, kind, M, cfg):
    """With the averaging budget FIXED at M (=TT's batch), push ttm's m DOWN from N: the smallest
    m for which the M-sample average of ttm still passes (eps,delta) (pass rate >= 1 - delta).
    The pass rate drops as m shrinks, so valid m form an upper band [m_best, N]; decrease until it
    fails.  Returns (m_best, cost_ttm)."""
    A = make_A(kind, d, N, DEVICE)
    tv = true_value(kind, d, N, K)
    eps, delta, groups = cfg["eps"], cfg["delta"], cfg["groups"]
    best = N                                                 # ttm@N is richer than tt@N, so it passes
    for m in range(N, 0, -1):
        sk = REGISTRY["ttm"](d, N, m, cfg["dist"], DEVICE)
        if pass_rate(sk, A, tv, K, eps, M, groups) >= 1.0 - delta:
            best = m                                         # still valid -> try lower
        else:
            break                                            # first failure: lower m only worse
    return best, M * REGISTRY["ttm"](d, N, best).cost(K)


def _kron_baseline(d, N, K, kind, cfg):
    """KRON cheap scalar baseline: its own smallest batch reaching (eps,delta).  No guard needed --
    if kron is mostly zeros its averages stay far from the truth, so it just fails up to the
    batch_ceiling and is censored."""
    M = min_batch(lambda: REGISTRY["kron"](d, N, 1, cfg["dist"], DEVICE), d, N, K, kind, cfg)
    if M is None:
        return None, True
    return M * REGISTRY["kron"](d, N, 1).cost(K), False


def _compare_point(kind, d, N, K, cfg):
    M, c_tt, tt_cens = _tt_anchor(d, N, K, kind, cfg)
    if tt_cens:
        return {"tt": None, "ttm": None, "kron": None, "m_best": None, "M": None, "tt_censored": True}
    m_best, c_ttm = _ttm_push_down(d, N, K, kind, M, cfg)
    c_kron, kron_cens = _kron_baseline(d, N, K, kind, cfg)
    return {"tt": c_tt, "ttm": c_ttm, "kron": None if kron_cens else c_kron,
            "m_best": m_best, "M": M, "tt_censored": False}


# ============================================================================
# the sweeps
# ============================================================================
def compare_to_tt(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    torch.manual_seed(cfg["seed"])
    folder = os.path.join("results", "compare_embeddings")
    os.makedirs(folder, exist_ok=True)
    print(f"\n=== EXP 2: compare_to_tt (device={DEVICE}) ===")
    print(f"(eps,delta)=({cfg['eps']},{cfg['delta']}); pass = fraction within eps, need >= "
          f"{1 - cfg['delta']:.2f}.  TT anchors M at bond N; TTM pushes m down at the same M; "
          f"a sketch needing > {cfg['batch_ceiling']} averages is censored.")

    out = {}
    for kind in cfg["spectra"]:
        # ---- cost vs K (fixed N) ----
        N = cfg["compare_K"]["N"]
        rows = []
        print(f"\n[{kind}]  vs K  (N={N})")
        for K in cfg["compare_K"]["Ks"]:
            r = _compare_point(kind, cfg["d"], N, K, cfg)
            rows.append(r); _print_cmp(f"K={K}", r)
        out[f"{kind}_vsK"] = rows
        _plot_compare(cfg["compare_K"]["Ks"], "K (trace power)", rows,
                      f"cost to (eps,delta)  |  {kind}  |  d={cfg['d']} N={N}",
                      folder, f"vsK_{kind}.png")

        # ---- cost vs N (fixed K) ----
        K = cfg["compare_N"]["K"]
        rows = []
        print(f"\n[{kind}]  vs N  (K={K})")
        for N in cfg["compare_N"]["Ns"]:
            r = _compare_point(kind, cfg["d"], N, K, cfg)
            rows.append(r); _print_cmp(f"N={N}", r)
        out[f"{kind}_vsN"] = rows
        _plot_compare(cfg["compare_N"]["Ns"], "N (sites)", rows,
                      f"cost to (eps,delta)  |  {kind}  |  d={cfg['d']} K={K}",
                      folder, f"vsN_{kind}.png")

    json.dump({"config": cfg, "out": out},
              open(os.path.join(folder, "compare_results.json"), "w"), indent=2)
    print(f"\nsaved -> {os.path.abspath(folder)}")
    return out


def _print_cmp(label, r):
    if r["tt_censored"]:
        print(f"   {label}: TT CENSORED (unreliable) -> point skipped")
        return
    supp = "SUPPRESSED" if (r["ttm"] is not None and r["ttm"] < r["tt"]) else "no"
    kron = "censored" if r["kron"] is None else f"{r['kron']:.2e}"
    print(f"   {label}: M={r['M']:>7}  m_best={r['m_best']:>3}  "
          f"cost tt={r['tt']:.2e} ttm={r['ttm']:.2e} kron={kron}   tt-suppressed: {supp}")


def _plot_compare(xs, xlabel, rows, title, folder, fname):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for algo, color in (("ttm", "tab:blue"), ("tt", "tab:orange"), ("kron", "tab:green")):
        gx = [x for x, r in zip(xs, rows) if r.get(algo) is not None]
        gy = [r[algo] for r in rows if r.get(algo) is not None]
        if gx:
            ax.plot(gx, gy, marker="o", ms=8, lw=2, color=color, label=algo)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("contraction cost to reach (eps,delta)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=11)
    fig.tight_layout()
    path = os.path.join(folder, fname)
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"   saved {os.path.abspath(path)}")


if __name__ == "__main__":
    compare_to_tt()
