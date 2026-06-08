"""
trace_MB_mpo.py
================

Closed-form computation of  Tr( M  (A ⊗ A) )  for the 4-replica matrix M
of the TT sketch (as in the thesis), where A is given as an MPO.

USAGE
-----
    from trace_MB_mpo import trace_MB_mpo
    val = trace_MB_mpo(m, N, d, A_cores, distribution='gaussian')

A_cores : list of N numpy arrays, A_cores[l].shape == (r_L, d, d, r_R),
          with r_L == 1 at l == 0 and r_R == 1 at l == N - 1.
          A is the d^N x d^N matrix
              A[i_1...i_N, j_1...j_N]
                  = Σ_α A_cores[0][1, i_1, j_1, α_1]
                      · A_cores[1][α_1, i_2, j_2, α_2]
                      · ... ·
                      · A_cores[N-1][α_{N-1}, i_N, j_N, 1]

Returns the exact scalar  Tr( M (A ⊗ A) )  via a length-N chain product
on matrices of size  4 r^2 x 4 r^2,  total cost  O(N · r^6).

DEPENDENCY
----------
This file imports build_W, build_u_init, build_u_end from build_M.py,
which encodes the per-site sketch building blocks W^P, u_init^{(P_1,P_2)},
u_end^{P_N} from formulas (19), (20), (21) of the thesis.  Those depend
only on  m  and the noise distribution.
"""
from __future__ import annotations
import numpy as np

from build_M import (
    build_W, build_u_init, build_u_end,
    closest_dividers, PATTERNS,
)

# ----------------------------------------------------------------------
#  The four 4-replica partitions on {a, b, c, d}
#
#      P     pattern of (a, b, c, d)
#      4K    a = b = c = d
#      A     a = b ≠ c = d
#      U_1   a = c ≠ b = d
#      U_2   a = d ≠ b = c
#
#  Per-site projector for the MPO of A:
#
#      β^{(l)}_P[(α, γ), (α', γ')]
#         = Σ_{(a,b,c,d) ∈ P}
#               A_cores[l][α, a, b, α']
#             · A_cores[l][γ, c, d, γ']
#
#  Stored as an  r_L^2  ×  r_R^2  matrix with row index (α, γ)
#  and column index (α', γ').
# ----------------------------------------------------------------------

def _beta_P(A_core, pattern, d):
    """β^{(l)}_P  as a matrix of shape (r_L^2,  r_R^2)."""
    rL, dphys, _, rR = A_core.shape
    assert dphys == d
    out = np.zeros((rL, rL, rR, rR))

    if pattern == '4K':                                     # a = b = c = d
        for a in range(d):
            out += np.einsum('ax,cy->acxy',
                             A_core[:, a, a, :],
                             A_core[:, a, a, :])

    elif pattern == 'A':                                    # a=b, c=d, a≠c
        diag_sum = sum(A_core[:, a, a, :] for a in range(d))
        out += np.einsum('ax,cy->acxy', diag_sum, diag_sum)
        for a in range(d):                                  # subtract a = c
            out -= np.einsum('ax,cy->acxy',
                             A_core[:, a, a, :],
                             A_core[:, a, a, :])

    elif pattern == 'U1':                                   # a=c, b=d, a≠b
        for a in range(d):
            for b in range(d):
                if a == b:
                    continue
                out += np.einsum('ax,cy->acxy',
                                 A_core[:, a, b, :],
                                 A_core[:, a, b, :])

    elif pattern == 'U2':                                   # a=d, b=c, a≠b
        for a in range(d):
            for b in range(d):
                if a == b:
                    continue
                out += np.einsum('ax,cy->acxy',
                                 A_core[:, a, b, :],
                                 A_core[:, b, a, :])

    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    return out.reshape(rL * rL, rR * rR)


# ----------------------------------------------------------------------
#  Closed-form Tr( M (A ⊗ A) )
# ----------------------------------------------------------------------

def trace_MB_mpo(m, N, d, A_cores, distribution='gaussian'):
    """
    Returns the exact scalar  Tr( M (A ⊗ A) ).

    Parameters
    ----------
    m, N, d       : sketch parameters
    A_cores       : list of length N of arrays of shape (r_L, d, d, r_R)
    distribution  : 'gaussian' or 'rademacher'
    """
    assert N >= 3, "construction requires N ≥ 3"
    assert len(A_cores) == N
    assert A_cores[0].shape[0] == 1 and A_cores[N - 1].shape[3] == 1, \
        "MPO must have rank-1 boundaries"

    # --- sketch-side building blocks (depend only on m, distribution) ---
    m1, m2 = closest_dividers(m)
    sigma2 = 1.0 / m
    sigma4 = sigma2 ** 2 if distribution == 'rademacher' else 3.0 * sigma2 ** 2

    W      = {p: build_W(p, m, m1, m2, sigma2, sigma4)
              for p in PATTERNS}
    u_init = {(p1, p2): build_u_init(p1, p2, m, m1, m2, sigma2, sigma4)
              for p1 in PATTERNS for p2 in PATTERNS}
    u_end  = {p: build_u_end(p, m, m1, m2, sigma2, sigma4)
              for p in PATTERNS}

    # --- A-side building blocks (depend only on A and d) ---
    betas = [{p: _beta_P(A_cores[l], p, d) for p in PATTERNS}
             for l in range(N)]

    # --- aggregated initial vector ---
    # absorbs sites 0 and 1 (i.e. l = 1, 2 in the thesis 1-indexed convention)
    r2_sq = betas[1]['4K'].shape[1]                       # right bond² at site 1
    u_tilde = np.zeros(4 * r2_sq)
    for p1 in PATTERNS:
        for p2 in PATTERNS:
            beta12 = (betas[0][p1] @ betas[1][p2]).flatten()   # shape (r²,)
            for k in range(4):
                u_tilde[k * r2_sq:(k + 1) * r2_sq] += \
                    u_init[(p1, p2)][k] * beta12

    # --- propagate through interior sites l = 2, ..., N - 2 ---
    # at each site:  Wtilde^{(l)} = Σ_P  W^P  ⊗  (β^{(l)}_P)^T
    # the transpose is because β has rows = LEFT bond, cols = RIGHT bond,
    # whereas Kronecker product is naturally indexed (out, in).
    for l in range(2, N - 1):
        rL = betas[l]['4K'].shape[0]
        rR = betas[l]['4K'].shape[1]
        Wt = np.zeros((4 * rR, 4 * rL))
        for p in PATTERNS:
            Wt += np.kron(W[p], betas[l][p].T)
        u_tilde = Wt @ u_tilde

    # --- aggregated end vector at site N - 1 ---
    rN_sq = betas[N - 1]['4K'].shape[0]                   # left bond² at site N-1
    e_tilde = np.zeros(4 * rN_sq)
    for p in PATTERNS:
        beta_end = betas[N - 1][p].flatten()              # shape (r²,)
        for k in range(4):
            e_tilde[k * rN_sq:(k + 1) * rN_sq] += u_end[p][k] * beta_end

    return float(e_tilde @ u_tilde)


def variance_quad_form_mpo(m, N, d, A_cores, distribution='gaussian',
                           trace_A=None):
    """
    Returns Var( <S^T S, A> )  =  Tr( M (A ⊗ A) )  −  tr(A)²
    for A given as MPO.
    """
    if trace_A is None:
        T = np.einsum('aiib->ab', A_cores[0])
        for l in range(1, len(A_cores)):
            T = T @ np.einsum('aiib->ab', A_cores[l])
        trace_A = float(T.squeeze())
    return trace_MB_mpo(m, N, d, A_cores, distribution) - trace_A ** 2


# ----------------------------------------------------------------------
#  Verification (run as a script)
# ----------------------------------------------------------------------

def _random_mpo_cores(N, d, r, seed=0):
    rng = np.random.default_rng(seed)
    cores = []
    for l in range(N):
        rL = 1 if l == 0 else r
        rR = 1 if l == N - 1 else r
        cores.append(rng.standard_normal((rL, d, d, rR)))
    return cores


def _mpo_to_dense(cores, d):
    N = len(cores)
    T = cores[0].reshape(d, d, -1)
    for l in range(1, N):
        nxt = cores[l]
        T = np.einsum('ija,apqb->ipjqb', T, nxt)
        T = T.reshape(T.shape[0] * T.shape[1],
                      T.shape[2] * T.shape[3], T.shape[4])
    return T[:, :, 0]


import matplotlib.pyplot as plt


def _trace_A_mpo(cores):
    """tr(A) for A given as MPO."""
    T = np.einsum('aiib->ab', cores[0])
    for l in range(1, len(cores)):
        T = T @ np.einsum('aiib->ab', cores[l])
    return float(T.squeeze())


def _random_psd_mpo(N, d, r=2, seed=0):
    """A = F F^T  with F a random Gaussian MPO of bond r.
       Result A is PSD with MPO bond r^2."""
    rng = np.random.default_rng(seed)
    cores = []
    for l in range(N):
        sL = 1 if l == 0     else r
        sR = 1 if l == N - 1 else r
        f = rng.standard_normal((sL, d, d, sR))
        # g[(a,a'), i, j, (b,b')] = sum_k f[a,i,k,b] · f[a',j,k,b']
        g = np.einsum('aikb,cjkd->acijbd', f, f)
        cores.append(g.reshape(sL * sL, d, d, sR * sR))
    return cores


def random_data_and_plot_multi_d(d_values=(2, 4),
                                 N_values=range(3, 35), n_trials=20, r=2,
                                 distribution='gaussian', seed=0,
                                 save_path='ratio_vs_N_multi_d.png'):
    """
    Same as random_data_and_plot, but overlays one curve per d in d_values.
    Linear axes, mean ± std error bars, different color per d.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10.colors

    results = {}
    for k, d in enumerate(d_values):
        Ns, means, stds = [], [], []
        for N in N_values:
            ys = []
            for t in range(n_trials):
                cores = _random_psd_mpo(N, d, r=r,
                                        seed=seed + 10000 * N + 100 * d + t)
                num   = trace_MB_mpo(N, N, d, cores, distribution=distribution)
                trA   = _trace_A_mpo(cores)
                ys.append(num / trA ** 2 - 1.0)
            ys = np.asarray(ys)
            Ns.append(N); means.append(ys.mean()); stds.append(ys.std(ddof=1))
            print(f"d={d}  N={N:>2}  mean={means[-1]:.4e}  std={stds[-1]:.4e}")

        Ns    = np.asarray(Ns)
        means = np.asarray(means)
        stds  = np.asarray(stds)
        results[d] = (Ns, means, stds)

        ax.errorbar(Ns, means, yerr=stds, fmt='o-',
                    capsize=4, color=colors[k % len(colors)],
                    label=f'd = {d}')

    ax.set_xlabel('N')
    ax.set_ylabel(r'$\mathrm{Tr}(MB)/\mathrm{Tr}(B)\;-\;1$')
    ax.set_title(f'm = N, {n_trials} random PSD MPOs per (d, N)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    return results

random_data_and_plot_multi_d(d_values=(2, 4))