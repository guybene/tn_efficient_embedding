"""Quick probe: is kron's high-K zero just the rank-1 all_ones annihilation (expected),
or does it also misbehave on identity (a real bug)?  B=1-style: draw a big batch of single
estimates and look at zero-fraction + mean vs true."""
import torch
from sketches import Kron, make_A, estimate, true_value

torch.manual_seed(0)
d, N = 2, 4
B = 200_000
m = 1  # kron ignores m

for kind in ("identity", "all_ones"):
    A = make_A(kind, d, N)
    tv = true_value(kind, d, N, 1)  # K filled per-row below
    print(f"\n=== {kind}  (d={d}, N={N}, B={B}) ===")
    print(f"{'K':>3} {'true':>10} {'mean est':>12} {'rel err':>10} {'frac zero':>10} {'max|est|':>12}")
    for K in (1, 2, 3, 4, 6, 8):
        sk = Kron(d, N, m)
        facs = [sk.draw(B) for _ in range(K)]
        est = estimate(sk, facs, A)
        tvK = true_value(kind, d, N, K)
        mean = est.mean().item()
        fz = (est == 0).float().mean().item()
        rel = abs(mean - tvK) / abs(tvK)
        print(f"{K:>3} {tvK:>10.4g} {mean:>12.5g} {rel:>10.4f} {fz:>10.5f} {est.abs().max().item():>12.5g}")
