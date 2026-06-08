"""
Exact construction of the 4-replica matrix
        M = E[S^T S ⊗ S^T S]   ∈  R^{d^{2N} × d^{2N}}
for the sketching tensor network of the thesis.

Strategy:
  * Encode the four 4-replica partitions of {a,b,c,d} as
        A  = (a=b)(c=d)         block map (0,0,1,1)
        U1 = (a=c)(b=d)                  (0,1,0,1)
        U2 = (a=d)(b=c)                  (0,1,1,0)
        4K = abcd                         (0,0,0,0)
  * Implement the partition meet  P1 ∧ P2  combinatorially (returns INVALID
    when the meet has an odd-sized block, which kills the moment).
  * Build μ^X_{P,B}, μ^g_{L,R}, then W^P, u_init^{(P1,P2)}, u_end^{P_N}
    by direct evaluation of formulas (19), (20), (21).
  * Precompute  V[P_seq]  for every P_seq ∈ {A,U1,U2,4K}^N  (only 4^N of them).
  * For each pattern sequence, enumerate the (i^a,i^b,i^c,i^d) tuples that
    realize it, fill M with V[P_seq].  Skip invalid partitions automatically.

Convention for the matrix view of M:
        row index = i^a · d^N + i^c       (matches existing trace code)
        col index = i^b · d^N + i^d
"""
from __future__ import annotations
import itertools
import math
import numpy as np
import torch


PATTERNS = ('A', 'U1', 'U2', '4K')

# Replica-to-block tag for each partition. Replicas (a,b,c,d) -> (0,1,2,3).
PARTITION_MAP = {
    'A':  (0, 0, 1, 1),
    'U1': (0, 1, 0, 1),
    'U2': (0, 1, 1, 0),
    '4K': (0, 0, 0, 0),
}


def closest_dividers(m):
    target = math.sqrt(m)
    best = (1, m, abs(1 - target))
    for m1 in range(1, int(math.sqrt(m)) + 1):
        if m % m1 == 0:
            d_ = abs(m1 - target)
            if d_ < best[2]:
                best = (m1, m // m1, d_)
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Partition meet  P1 ∧ P2.   Two replicas land in the same meet-block iff
# they agree in BOTH P1 and P2.  An odd-sized block kills the moment.
# ---------------------------------------------------------------------------
def joint(p1, p2):
    if p1 == 'INVALID' or p2 == 'INVALID':
        return 'INVALID'
    b1, b2 = PARTITION_MAP[p1], PARTITION_MAP[p2]
    blocks = {}
    for i in range(4):
        blocks.setdefault((b1[i], b2[i]), []).append(i)
    sizes = sorted(len(v) for v in blocks.values())
    if sizes == [4]:
        return '4K'
    if sizes != [2, 2]:
        return 'INVALID'
    block_a = next(v for v in blocks.values() if 0 in v)
    if 1 in block_a: return 'A'
    if 2 in block_a: return 'U1'
    if 3 in block_a: return 'U2'


def pattern_of_4tuple(a, b, c, d_val):
    if a == b == c == d_val:                      return '4K'
    if a == b and c == d_val and a != c:          return 'A'
    if a == c and b == d_val and a != b:          return 'U1'
    if a == d_val and b == c and a != b:          return 'U2'
    return 'INVALID'


def N_pat(p, M):
    return M if p == '4K' else M * (M - 1)


# ---------------------------------------------------------------------------
# Leaf and gadget 4-moments
# ---------------------------------------------------------------------------
def mu_X(P, B, s2, s4):
    """E[ Π_q X_{i^q, k^q} ]  with i-pattern P and k-pattern B."""
    j = joint(P, B)
    if j == 'INVALID':
        return 0.0
    return s4 if j == '4K' else s2 * s2


def mu_g(L, R, s2, s4, m):
    """Gadget moment with left-half outer pattern L, right-half R."""
    total = 0.0
    for s in PATTERNS:                       # internal-l pattern
        ml, mr = joint(L, s), joint(R, s)
        if ml == 'INVALID' or mr == 'INVALID':
            continue
        wL = s4 if ml == '4K' else s2 * s2
        wR = s4 if mr == '4K' else s2 * s2
        total += N_pat(s, m) * wL * wR
    return total


# ---------------------------------------------------------------------------
# W^P, u_init^(P1,P2), u_end^P_N   --- formulas (19), (20), (21)
# Basis: [A, U1, U2, 4K]
# ---------------------------------------------------------------------------
def build_W(P, m, m1, m2, s2, s4):
    W = np.zeros((4, 4))
    for i, Zo in enumerate(PATTERNS):
        for j, Zi in enumerate(PATTERNS):
            inner = 0.0
            for s1 in PATTERNS:
                for s2p in PATTERNS:
                    mux = mu_X(P, joint(s1, s2p), s2, s4)
                    if mux == 0: continue
                    Larg, Rarg = joint(Zi, s1), joint(s2p, Zo)
                    if Larg == 'INVALID' or Rarg == 'INVALID': continue
                    inner += (N_pat(s1, m1) * N_pat(s2p, m2)
                              * mux * mu_g(Larg, Rarg, s2, s4, m))
            W[i, j] = N_pat(Zo, m) * inner
    return W


def build_u_init(P1, P2, m, m1, m2, s2, s4):
    u = np.zeros(4)
    for i, Zo in enumerate(PATTERNS):
        tot = 0.0
        for k1 in PATTERNS:
            mux1 = mu_X(P1, k1, s2, s4)
            if mux1 == 0: continue
            for k1p in PATTERNS:
                for k2p in PATTERNS:
                    mux2 = mu_X(P2, joint(k1p, k2p), s2, s4)
                    if mux2 == 0: continue
                    Larg, Rarg = joint(k1, k1p), joint(k2p, Zo)
                    if Larg == 'INVALID' or Rarg == 'INVALID': continue
                    tot += (N_pat(k1, m) * N_pat(k1p, m1) * N_pat(k2p, m2)
                            * mux1 * mux2 * mu_g(Larg, Rarg, s2, s4, m))
        u[i] = N_pat(Zo, m) * tot
    return u


def build_u_end(PN, m, m1, m2, s2, s4):
    u = np.zeros(4)
    for j, Zi in enumerate(PATTERNS):
        tot = 0.0
        for n_pat in ('A', '4K'):
            for k1 in PATTERNS:
                for k2 in PATTERNS:
                    mux = mu_X(PN, joint(k1, k2), s2, s4)
                    if mux == 0: continue
                    Larg, Rarg = joint(Zi, k1), joint(k2, n_pat)
                    if Larg == 'INVALID' or Rarg == 'INVALID': continue
                    tot += (N_pat(n_pat, m) * N_pat(k1, m1) * N_pat(k2, m2)
                            * mux * mu_g(Larg, Rarg, s2, s4, m))
        u[j] = tot
    return u


# ---------------------------------------------------------------------------
# Full M
# ---------------------------------------------------------------------------
def compute_M(m, N, d,
              distribution='gaussian',
              device='cpu', dtype=torch.float64):
    assert N >= 3, "Construction assumes N >= 3."
    m1, m2 = closest_dividers(m)
    s2 = 1.0 / m
    if distribution == 'rademacher':
        s4 = s2 ** 2
    elif distribution == 'gaussian':
        s4 = 3.0 * s2 ** 2
    else:
        raise ValueError(f"unknown distribution {distribution!r}")

    W      = {p: build_W(p, m, m1, m2, s2, s4) for p in PATTERNS}
    u_init = {(p1, p2): build_u_init(p1, p2, m, m1, m2, s2, s4)
              for p1 in PATTERNS for p2 in PATTERNS}
    u_end  = {p: build_u_end(p, m, m1, m2, s2, s4) for p in PATTERNS}

    # All 4^N chain values
    V = {}
    for P_seq in itertools.product(PATTERNS, repeat=N):
        v = u_init[(P_seq[0], P_seq[1])].copy()
        for l in range(2, N - 1):
            v = W[P_seq[l]] @ v
        V[P_seq] = float(u_end[P_seq[N - 1]] @ v)

    D = d ** N
    M = np.zeros((D * D, D * D), dtype=np.float64)

    # Per-site (a,b,c,d) tuples that realize each pattern.
    site_tuples_for = {
        '4K': lambda d: [(v, v, v, v) for v in range(d)],
        'A':  lambda d: [(v1, v1, v2, v2)
                         for v1 in range(d) for v2 in range(d) if v1 != v2],
        'U1': lambda d: [(v1, v2, v1, v2)
                         for v1 in range(d) for v2 in range(d) if v1 != v2],
        'U2': lambda d: [(v1, v2, v2, v1)
                         for v1 in range(d) for v2 in range(d) if v1 != v2],
    }
    cache = {p: site_tuples_for[p](d) for p in PATTERNS}

    for P_seq in itertools.product(PATTERNS, repeat=N):
        val = V[P_seq]
        if val == 0.0:
            continue
        per_site = [cache[p] for p in P_seq]
        for combo in itertools.product(*per_site):
            ia = ib = ic = id_ = 0
            for s in combo:
                ia = ia * d + s[0]
                ib = ib * d + s[1]
                ic = ic * d + s[2]
                id_ = id_ * d + s[3]
            M[ia * D + ic, ib * D + id_] = val

    return torch.from_numpy(M).to(dtype=dtype, device=device)

import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

def explosion(d=2, Ns=(3,4,5,6), Ls=(1,2,3), distribution='gaussian'):
    lam = {L: [] for L in Ls}        # lambda_max(M)^L / d^{2N}
    for N in Ns:
        print(f"Working on {N}")
        M = compute_M(m=N, N=N, d=d, distribution=distribution).numpy()

        D2 = float(d**(2*N))
        lmax = float(eigsh(M, k=1, which='LM', return_eigenvectors=False)[0])
        for L in Ls:
            lam[L].append(lmax**L / D2)
    return lam

def plot_explosion(lam, d, Ns, Ls):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for L in Ls:
        ax[0].plot(Ns, lam[L], 'o-', label=f'L={L}')
    # reference slope d^{(L-2)N} for L=3
    ref = [d**((3-2)*N) for N in Ns]
    ax[0].plot(Ns, ref, 'k--', alpha=0.4, label=r'$d^{N}$ slope')
    for a, t in zip(ax, [r'$\lambda_{\max}^L / d^{2N}$']):
        a.set_yscale('log'); a.set_xlabel('N  (m=N)'); a.set_ylabel(t)
        a.legend(); a.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("M_eigen_value_explosion.png")

lam = explosion(d=2, Ns=(3,4,5,6,7,8,9))
plot_explosion(lam, d=2, Ns=(3,4,5,6,7,8,9), Ls=[1,2,3])