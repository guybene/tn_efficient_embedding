from __future__ import annotations
import torch
import os
import math
import gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def closest_dividers(m, D=1):
    split_dimension = math.sqrt(m / D)
    closest_b1, closest_b2 = None, None
    min_diff = float('inf')

    for b1 in range(1, int(math.sqrt(m)) + 1):
        if m % b1 == 0:
            b2 = m // b1
            diff = abs(b1 - split_dimension)
            if diff < min_diff:
                closest_b1, closest_b2 = b1, b2
                min_diff = diff
    return closest_b1, closest_b2
# ---------------------------------------------------------------------------
# Building blocks: initial vectors, transition matrices, end vectors
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Building blocks: initial vectors, transition matrices, end vectors
# 4-state basis: [A, U_1, U_2, 4K]
# ---------------------------------------------------------------------------

import math

def build_blocks(m: int, distribution="rademacher", dtype=torch.float64, device="cpu"):
    """
    Hard-coded build_blocks in the 4-state basis [A, U_1, U_2, 4K].

    distribution: "rademacher" or "gaussian"
        - rademacher: sigma_2 = 1/m, sigma_4 = 1/m^2
        - gaussian:   sigma_2 = 1/m, sigma_4 = 3/m^2
    """
    m1, m2 = closest_dividers(m)

    # --- Distribution moments ---------------------------------------------
    s2 = 1.0 / m
    if distribution == "rademacher":
        s4 = s2 ** 2
    elif distribution == "gaussian":
        s4 = 3.0 * s2 ** 2
    else:
        raise ValueError(f"unknown distribution {distribution!r}")

    def t(x):
        return torch.tensor(x, dtype=dtype, device=device)

    # =====================================================================
    # Transition matrix W^{4K}
    # =====================================================================
    # Diagonal blocks of the 2P sub-matrix:
    W4K_diag2P = m**3 * s2**2 * (m * (m - 1) * s2**2 * ((m - 1) * s2**2 + s4))
    # Off-diagonal of the 2P sub-matrix:
    W4K_off2P = m**3 * s2**2 * ((m - 1) * s2**2 * ((m1 + m2 - 2) * s2**2 + s4))
    # Column 4K (rows A, U1, U2):
    W4K_col4K = m**3 * s2**2 * (m - 1) * (
        (m * (m - 1) + 2 * m1 - m2 - 1) * s2**4
        + (m + m2 - 2) * s2**2 * s4
        + s4**2
    )
    # Row 4K (cols A, U1, U2):  same shape with m1<->m2 swap
    W4K_row4K = m**3 * s2**2 * (
        (m * (m - 1) + 2 * m2 - m1 - 1) * s2**4
        + (m + m1 - 2) * s2**2 * s4
        + s4**2
    )
    # Bottom-right corner [4K, 4K]:
    W4K_4K4K = m**3 * (
        3 * (m * (m - 1) - (m1 + m2 - 2)) * s2**6
        + 3 * (m + m1 + m2 - 3) * s2**4 * s4
        + s4**3
    )

    W_4K = t([
        [W4K_diag2P, W4K_off2P,  W4K_off2P,  W4K_col4K],
        [W4K_off2P,  W4K_diag2P, W4K_off2P,  W4K_col4K],
        [W4K_off2P,  W4K_off2P,  W4K_diag2P, W4K_col4K],
        [W4K_row4K,  W4K_row4K,  W4K_row4K,  W4K_4K4K ],
    ])

    # =====================================================================
    # Transition matrix W^{2P}
    # =====================================================================
    W2P_AA   = m**3 * s2**4 * (m**2 * (m - 1) * s2**2)
    W2P_AU   = m**3 * s2**4 * (m2 * (m - 1) * s2**2)
    W2P_A4K  = m**3 * s2**4 * (m2 * (m - 1) * ((m * m1 - 1) * s2**2 + s4))
    W2P_UA   = m**3 * s2**4 * (m1 * (m - 1) * s2**2)
    W2P_UU_d = m**3 * s2**4 * (m * (m - 1) * s2**2)
    W2P_UU_o = m**3 * s2**4 * ((m - 1) * s2**2)
    W2P_U4K  = m**3 * s2**4 * ((m - 1) * ((m + m1 - 2) * s2**2 + s4))
    W2P_4KA  = m**3 * s2**4 * (m1 * ((m * m2 - 1) * s2**2 + s4))
    W2P_4KU  = m**3 * s2**4 * ((m + m2 - 2) * s2**2 + s4)
    # NOTE: this entry has a DIFFERENT common prefactor from the others
    # (m^3 * sigma_2^2 instead of m^3 * sigma_2^4)
    W2P_4K4K = m**3 * s2**2 * (
        (m * (m + 2) - (m1 + m2) - 1) * s2**4
        + (m1 + m2 - 2) * s2**2 * s4
        + s4**2
    )

    W_2P = t([
        [W2P_AA,  W2P_AU,   W2P_AU,   W2P_A4K ],
        [W2P_UA,  W2P_UU_d, W2P_UU_o, W2P_U4K ],
        [W2P_UA,  W2P_UU_o, W2P_UU_d, W2P_U4K ],
        [W2P_4KA, W2P_4KU,  W2P_4KU,  W2P_4K4K],
    ])

    W = {"4K": W_4K, "2P": W_2P}

    # =====================================================================
    # Initial vectors u_init^{(i_1, i_2)}
    # =====================================================================

    # ---- (4K, 4K): all three of [A, U_1, U_2] are equal -----------------
    u_4K4K_2P = m**4 * s2**2 * (m - 1) * (
        (m**3 - 2 * m**2 + 2 * m * (m1 + m2) - 3 * m - 2 * (m1 + m2) + 4) * s2**6
        + (2 * m**2 + 2 * m1 - m2 - 3) * s2**4 * s4
        + (m + m2 - 2) * s2**2 * s4**2
        + s4**3
    )
    u_4K4K_4K = m**4 * (
        3 * (m**3 - 2 * m**2 - m * m1 + 2 * m * m2 + m1 - 2 * m2 + 1) * s2**8
        + 3 * (2 * m**2 + m * m1 - 4 * m - 2 * m1 - m2 + 4) * s2**6 * s4
        + 3 * (2 * m + m1 + m2 - 4) * s2**4 * s4**2
        + s4**4
    )

    # ---- (4K, 2P) -------------------------------------------------------
    u_4K2P_A = m**4 * s2**4 * (m - 1) * (
        (m**3 - m**2 + 2 * m * m2 - 2 * m2) * s2**4
        + (m**2 - m2) * s2**2 * s4
        + m2 * s4**2
    )
    u_4K2P_U = m**4 * s2**4 * (m - 1) * (
        (m**2 + m * m1 - m1 - 1) * s2**4
        + (m + m1 - 2) * s2**2 * s4
        + s4**2
    )
    u_4K2P_4K = m**4 * s2**2 * (
        (m**3 + m**2 - m * m1 + 2 * m * m2 - 6 * m + m1 - 2 * m2 + 4) * s2**6
        + (m**2 + m * m1 + 4 * m - 2 * m1 - m2 - 3) * s2**4 * s4
        + (m1 + m2 - 2) * s2**2 * s4**2
        + s4**3
    )

    # ---- (2P, 4K) -------------------------------------------------------
    u_2P4K_A = m**4 * s2**4 * (m - 1) * (
        (m**3 - m**2 + 2 * m1 - m2 - 1) * s2**4
        + (m**2 + m2 - 2) * s2**2 * s4
        + s4**2
    )
    u_2P4K_U = m**4 * s2**4 * (m - 1) * (
        (m**2 + m * (m1 + m2) - 3 * m + m1 - 2 * m2 + 1) * s2**4
        + (2 * m + m2 - 3) * s2**2 * s4
        + s4**2
    )
    u_2P4K_4K = m**4 * s2**2 * (
        (m**3 + m**2 - m * m1 + 2 * m * m2 - 3 * m - 2 * m1 - 5 * m2 + 7) * s2**6
        + (m**2 + m * m1 + 2 * m1 + 3 * m2 - 7) * s2**4 * s4
        + (m - 1) * s2**2 * s4**2
        + s4**3
    )

    # ---- (2P, 2P) -------------------------------------------------------
    u_2P2P_A = m**4 * s2**6 * (m - 1) * ((m**3 - m2) * s2**2 + m2 * s4)
    u_2P2P_U = m**4 * s2**6 * (m - 1) * ((m * m1 + m - 2) * s2**2 + s4)
    u_2P2P_4K = m**4 * s2**4 * (
        (m**3 - m * m1 + 2 * m - m2 - 1) * s2**4
        + (m * m1 + m2 - 2) * s2**2 * s4
        + s4**2
    )

    u_init = {
        ("4K", "4K"): t([u_4K4K_2P, u_4K4K_2P, u_4K4K_2P, u_4K4K_4K]),
        ("4K", "2P"): t([u_4K2P_A,  u_4K2P_U,  u_4K2P_U,  u_4K2P_4K]),
        ("2P", "4K"): t([u_2P4K_A,  u_2P4K_U,  u_2P4K_U,  u_2P4K_4K]),
        ("2P", "2P"): t([u_2P2P_A,  u_2P2P_U,  u_2P2P_U,  u_2P2P_4K]),
    }

    # =====================================================================
    # End vectors u_end^{i_N}
    # =====================================================================

    u_end_4K_A = m**3 * s2**2 * (
        (m**3 - m**2 - m1 + 2 * m2 - 1) * s2**4
        + (m**2 + m1 - 2) * s2**2 * s4
        + s4**2
    )
    u_end_4K_U = m**3 * s2**2 * (
        (m**2 + m * (m1 + m2) - 3 * m - 2 * m1 + m2 + 1) * s2**4
        + (2 * m + m1 - 3) * s2**2 * s4
        + s4**2
    )
    u_end_4K_4K = m**3 * (
        (m**3 + m**2 + 2 * m * m1 - m * m2 - 3 * m - 5 * m1 - 2 * m2 + 7) * s2**6
        + (m**2 + m * m2 + 3 * m1 + 2 * m2 - 7) * s2**4 * s4
        + (m - 1) * s2**2 * s4**2
        + s4**3
    )

    u_end_2P_A = m**3 * s2**4 * ((m**3 - m1) * s2**2 + m1 * s4)
    u_end_2P_U = m**3 * s2**4 * ((m * m2 + m - 2) * s2**2 + s4)
    u_end_2P_4K = m**3 * s2**2 * (
        (m**3 - m * m2 + 2 * m - m1 - 1) * s2**4
        + (m * m2 + m1 - 2) * s2**2 * s4
        + s4**2
    )

    u_end = {
        "4K": t([u_end_4K_A, u_end_4K_U, u_end_4K_U, u_end_4K_4K]),
        "2P": t([u_end_2P_A, u_end_2P_U, u_end_2P_U, u_end_2P_4K]),
    }

    return u_init, W, u_end


# ---------------------------------------------------------------------------
# Diagonal trace: explicit loop over all d^{2N} diagonal indices
# ---------------------------------------------------------------------------

def diagonal_entry(i_a, i_b, u_init, W, u_end):
    """Evaluate one diagonal entry via the chain."""
    N = len(i_a)

    # Per-site physical type. Since the diagonal is a=b and c=d,
    # the 4 replicas entering are (a, a, c, c).
    # They collapse into a 4K wildcard iff a == c, otherwise they act as 2P.
    types = ["4K" if i_a[l] == i_b[l] else "2P" for l in range(N)]

    # u_init absorbs sites 0,1 (i_1, i_2 in 1-indexed thesis notation).
    v = u_init[(types[0], types[1])]

    # W matrices act for gadgets g^(2),...,g^(N-2), carrying physical
    # leg i_{p+1}. In 0-indexed Python that is types[2], ..., types[N-2].
    # For N=3, this loop safely skips because range(2, 2) is empty.
    for j in range(2, N - 1):
        v = W[types[j]] @ v

    # u_end absorbs site N-1 (i_N in 1-indexed thesis notation).
    return torch.dot(u_end[types[N - 1]], v)


def compute_trace_M(m: int, N: int, d: int,
                    distribution="rademacher",
                    dtype=torch.float64, device="cpu"):
    """
    Returns Tr(M) = sum over all (a, c) in [d]^N x [d]^N.
    Iterates explicitly over the d^{2N} main-diagonal indices.
    """
    assert N >= 3, "This implementation handles N >= 3."

    u_init, W, u_end = build_blocks(m, distribution=distribution,
                                    dtype=dtype, device=device)

    # Trace mathematical formula
    math_formula_u_init = (d**2 * u_init[("4K","4K")] +
                          d**2 * (d-1) * (u_init[("4K","2P")] + u_init[("2P","4K")])
                          + (d * (d-1))**2 * u_init[("2P","2P")])
    math_formula_W = torch.linalg.matrix_power(d * W["4K"] + d*(d-1)* W["2P"], N-3)
    math_formula_u_end = d * u_end["4K"] + d*(d-1)*u_end["2P"]

    math_trace_formula = math_formula_u_end @ math_formula_W @ math_formula_u_init
    return math_trace_formula.cpu().numpy()

@torch.no_grad()
def calculate_variances_batched(m, N, d, M, batch_size=100,
                                device='cuda',distribution="gaussian",):
    """
    Computes linear tree contraction variance and tracks convergence over M samples.
    """
    sigma = 1.0 / math.sqrt(m)
    m1, m2 = closest_dividers(m)

    expected_trace_linear = 0.0

    # 1. Calculate Exact Analytical Truth Once
    model_trace = compute_trace_M(m, N, d,distribution=distribution, device=device)
    analytical_var = model_trace - (d ** (2 * N))

    iterations = []
    errors = []
    current_M = 0

    # 2. Iterative Monte Carlo Execution
    for b_start in range(0, M, batch_size):
        B = min(batch_size, M - b_start)

        if distribution == "rademacher":
        # Generate Shared Arrays (RADEMACHER DISTRIBUTION)
          leaves = (torch.randint(0, 2, (B, N, d, m), dtype=torch.float32, device=device) * 2 - 1) * sigma
          X1 = (torch.randint(0, 2, (B, N - 1, m, m1, m), dtype=torch.float32, device=device) * 2 - 1) * sigma
          X2 = (torch.randint(0, 2, (B, N - 1, m, m2, m), dtype=torch.float32, device=device) * 2 - 1) * sigma
        elif distribution == "gaussian":
          leaves = torch.randn((B, N, d, m), dtype=torch.float32, device=device) * sigma
          X1 = torch.randn((B, N - 1, m, m1, m), dtype=torch.float32, device=device) * sigma
          X2 = torch.randn((B, N - 1, m, m2, m), dtype=torch.float32, device=device) * sigma


        g_raw = torch.einsum('bnkil, bnljz -> bnkijz', X1, X2)
        g_nodes = g_raw.reshape(B, N - 1, m, m, m)

        # --- LINEAR TREE CONTRACTION ---
        current_lin = leaves[:, 0]
        for i in range(1, N):
            L = current_lin
            R = leaves[:, i]
            g = g_nodes[:, i - 1]
            # RESTORED TO YOUR ORIGINAL, CORRECT TOPOLOGY
            current_lin = torch.einsum('bdm, ben, bmnp -> bdep', L, R, g).reshape(B, -1, m)

        trace_lin = torch.sum(current_lin * current_lin, dim=(1, 2))
        expected_trace_linear += torch.sum(trace_lin ** 2).item()

        current_M += B

        # Compute running variance and absolute error
        running_var = (expected_trace_linear / current_M) - (d ** (2 * N))
        diff = abs(running_var - analytical_var) / analytical_var * 100

        iterations.append(current_M)
        errors.append(diff)

        # Memory Cleanup
        del leaves, X1, X2, g_raw, g_nodes, current_lin, L, R, g, trace_lin
        if device == 'cuda':
            torch.cuda.empty_cache()

    return iterations, errors, analytical_var, running_var

def analyze_convergence(m, N, d, M, batch_size=2000, distribution='gaussian'):
    """
    Executes a single configuration and plots the convergence trajectory.
    """
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running Convergence Test on: {device.upper()} ---")
    print(f"Configuration: m={m}, N={N}, d={d}, M={M}")

    iterations, errors, analytical_var, final_mc_var = calculate_variances_batched(
        m=m,
        N=N,
        d=d,
        M=M,
        batch_size=batch_size,
        device=device,
        distribution=distribution
    )

    print(f"\nFinal Analytical Var: {analytical_var:.4e}")
    print(f"Final MC Var:         {final_mc_var:.4e}")
    print(f"Final Error:          {errors[-1]:.4e}")

    # Plotting the Convergence Trajectory
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations[50:], errors[50:], color='crimson', linewidth=1.5, label='| MC Variance - Exact Analytical |')

    ax.set_title(f'Monte Carlo Convergence Trajectory\n(m={m}, N={N}, d={d})', fontsize=14)
    ax.set_xlabel('Monte Carlo Samples (M)', fontsize=12)
    ax.set_ylabel('Ralative Error', fontsize=12)

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=12)

    save_path = os.path.join(save_dir, f"convergence.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    m = 50000
    N = 4
    d = 3
    distribution = 'gaussian'


    # Execute a single test configuration
    # analyze_convergence(
    #     m=m,
    #     N=N,
    #     d=d,
    #     M=int(1e8),
    #     batch_size=50000,
    #     distribution=distribution
    # )
    print(compute_trace_M(m=m,N=N,d=d,distribution=distribution) - d**(2*N))