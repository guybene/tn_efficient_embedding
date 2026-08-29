"""
COST/ERROR EXPERIMENT for the trace-estimation sketches.

Algorithms are imported from sketches.py (the SINGLE source validated by test_sketches.py) --
no sketch code is duplicated here.  For each (A_kind, K, m) we plot the relative error of the
running-mean estimate of Tr(A^k) vs. CUMULATIVE CONTRACTION COST, for every algorithm:

  * each algorithm runs until its cumulative cost reaches a budget, so all curves END at ~the
    same x (the budget auto-sizes per panel so even the priciest sketch gets a full curve);
  * per step a sketch draws `batch` estimates, with batch = the LEADING m-order ratio
    (solomonik/dense: B, tt/ttm: B*m, kron: B*m^4) so kron draws m^4x more per step;
  * x = (steps*batch)*cost_per_step, cost_per_step = the validated closed-form Sketch.cost(K).

Output: one folder per (K, d, N) -- compare_K<K>_d<d>_N<N>/ -- with config.json and one large PNG
per (A_kind, m): <A_kind>_m<m>.png.  Figures are saved, not displayed.  The dense d^N matrix is
never formed.  GPU auto-detected; precision is sketches.DTYPE (edit that one line for float32).
"""
import os
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketches import REGISTRY, estimate, true_value, make_A, DTYPE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = {
    "d": 2,
    "N": 5,
    "ms": [100],
    "Ks": [6],
    "A_kinds": ["identity", "all_ones"],
    "algos": [ "ttm"],#, "tt", "kron"],    # "solomonik" "dense" benched (in REGISTRY, not run)
    "dist": "rademacher",
    "base_batch": 1,        # leading-order batch multiplier (kron draws base_batch*m^4 per step)
    "budget": "auto",       # cumulative-cost budget each algo runs to; "auto" = points*max_cost
    "points": 50,          # ~ solomonik steps per curve (sets the auto budget)
    "burn": 0.30,           # fraction of early steps dropped from the plot
    "seed": 0,
    "run_self_test": False,  # set True to verify TN == dense (imported from test_sketches) first
}


def batch_for(algo, m, base):
    """Per-step sample count = the LEADING m-order ratio so each step is comparable compute and
    kron (O(1)) draws m^4x more than solomonik (O(m^4)).  (NOT the exact cost ratio.)"""
    return {
        "solomonik": base, "dense": base,          # O(m^4)
        "tt": base * m, "ttm": base * m,            # O(m^3)
        "kron": base * m ** 4,                      # O(1)
    }[algo]


@torch.no_grad()
def run_curve(sketch, A, K, batch, cost, budget):
    """Draw `batch` estimates per step; stop once cumulative cost (steps*batch*cost) >= budget.
    Collect every prediction, then derive the running-mean curve and the variance from them.
    Returns (running mean of Tr(A^k) per step, cumulative cost per step, variance of the predictions)."""
    chunk = sketch.chunk(K)
    preds, xs, cum = [], [], 0.0
    while cum < budget:
        drawn = 0
        while drawn < batch:
            c = min(chunk, batch - drawn)
            facs = [sketch.draw(c) for _ in range(K)]
            preds.append(estimate(sketch, facs, A))
            drawn += c
        cum += batch * cost
        xs.append(cum)
    preds = torch.cat(preds)                                   # all predictions (each step adds `batch`)
    counts = torch.arange(1, len(xs) + 1, device=preds.device) * batch
    means = preds.cumsum(0)[counts - 1] / counts              # running mean at each step boundary
    return means.double().cpu(), torch.tensor(xs, dtype=torch.float64), preds.var().item()


def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    torch.manual_seed(cfg["seed"])
    print(f"device={DEVICE}  dtype={str(DTYPE).split('.')[-1]}")
    if cfg.get("run_self_test", False):
        from test_sketches import self_test
        assert self_test(), "TN != dense -- aborting"

    d, N, burn, points = cfg["d"], cfg["N"], cfg["burn"], cfg["points"]
    for A_kind in cfg["A_kinds"]:
        for K in cfg["Ks"]:
            A = make_A(A_kind, d, N, DEVICE)        # plain MPO: identity = I_d cores, all_ones = J/d^N
            tv = true_value(A_kind, d, N, K)         # identity -> d^N ; all_ones -> 1
            folder = f"compare_K{K}_d{d}_N{N}"      # one folder per parameter set (K, d, N)
            os.makedirs(folder, exist_ok=True)
            json.dump(cfg, open(os.path.join(folder, "config.json"), "w"), indent=2)
            for m in cfg["ms"]:
                sks = {a: REGISTRY[a](d, N, m, cfg["dist"], DEVICE) for a in cfg["algos"]}
                costs = {a: sk.cost(K) for a, sk in sks.items()}   # validated per-step op counts
                budget = points * max(costs.values()) if cfg["budget"] == "auto" else cfg["budget"]
                fig, ax = plt.subplots(figsize=(14, 8))            # ONE big graph per m
                panel = []
                for algo in cfg["algos"]:
                    sk, cost = sks[algo], costs[algo]
                    batch = batch_for(algo, m, cfg["base_batch"])
                    means, x, var = run_curve(sk, A, K, batch, cost, budget)
                    rel = (means - tv).abs() / abs(tv)
                    relvar = var / tv ** 2                    # relative (scale-free) variance
                    b0 = int(len(rel) * burn)
                    ax.plot(x[b0:], rel[b0:], linewidth=2, marker="o", ms=4,
                            label=f"{algo}  err={rel[-1].item():.2e}  relvar={relvar:.2e}")
                    panel.append(rel[b0:])
                    print(f"  {A_kind:9s} K={K} m={m:3d} [{algo:9s}] "
                          f"cost/step={cost:12d} batch={batch:9d} steps={len(rel):5d} "
                          f"rel@end={rel[-1]:.3e} relvar={relvar:.3e}")
                top = 1.3 * torch.quantile(torch.cat(panel), 0.75).item()
                ax.set_ylim(-0.03 * top, top)        # auto: ceiling tracks 75th-pct rel err (clips tt spikes)
                ax.set_title(f"m={m}   {A_kind}   K={K}   d={d}  N={N}   (true={tv:g})", fontsize=14)
                ax.set_xlabel("cumulative cost  (free*contracted*free)", fontsize=12)
                ax.set_ylabel("rel error of running mean", fontsize=12)
                ax.grid(True, alpha=0.3); ax.legend(fontsize=12)
                fig.tight_layout()
                path = os.path.join(folder, f"{A_kind}_m{m}.png")
                fig.savefig(path, dpi=130); print(f"  saved {os.path.abspath(path)}")
                plt.close(fig)


if __name__ == "__main__":
    main()
