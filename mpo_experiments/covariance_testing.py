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


@torch.no_grad()
def calculate_variances_batched(m, N, d, M, batch_size=100, device='cuda'):
    """
    Computes linear, shallow binary tree, and simple kronecker variances.
    """
    sigma = 1.0 / math.sqrt(m)
    m1, m2 = closest_dividers(m)

    expected_trace_linear = 0.0
    expected_trace_shallow = 0.0
    expected_trace_kronecker = 0.0

    D_depth = int(math.ceil(math.log2(N)))
    power_of_2 = 2 ** D_depth
    leaves_to_pair = 2 * N - power_of_2

    for b_start in range(0, M, batch_size):
        B = min(batch_size, M - b_start)

        # 1. Generate Shared Arrays
        leaves = torch.randn(B, N, d, m, dtype=torch.float32, device=device) * sigma

        X1 = torch.randn(B, N - 1, m, m1, m, dtype=torch.float32, device=device) * sigma
        X2 = torch.randn(B, N - 1, m, m2, m, dtype=torch.float32, device=device) * sigma

        g_raw = torch.einsum('bnkil, bnljz -> bnkijz', X1, X2)
        g_nodes = g_raw.reshape(B, N - 1, m, m, m)

        # --- 1. LINEAR TREE CONTRACTION ---
        current_lin = leaves[:, 0]
        for i in range(1, N):
            L = current_lin
            R = leaves[:, i]
            g = g_nodes[:, i - 1]
            current_lin = torch.einsum('bdm, ben, bmnp -> bdep', L, R, g).reshape(B, -1, m)

        trace_lin = torch.sum(current_lin * current_lin, dim=(1, 2))
        expected_trace_linear += torch.sum(trace_lin ** 2).item()

        # --- 2. SHALLOW BINARY TREE CONTRACTION ---
        layer = [leaves[:, i] for i in range(N)]
        g_idx = 0

        next_layer = []
        for j in range(0, leaves_to_pair, 2):
            L = layer[j]
            R = layer[j + 1]
            g = g_nodes[:, g_idx]
            g_idx += 1
            next_layer.append(torch.einsum('bdm, ben, bmnp -> bdep', L, R, g).reshape(B, -1, m))

        next_layer.extend(layer[leaves_to_pair:])
        layer = next_layer

        while len(layer) > 1:
            next_layer = []
            for j in range(0, len(layer), 2):
                L = layer[j]
                R = layer[j + 1]
                g = g_nodes[:, g_idx]
                g_idx += 1
                next_layer.append(torch.einsum('bdm, ben, bmnp -> bdep', L, R, g).reshape(B, -1, m))
            layer = next_layer

        S_mat_shal = layer[0]
        trace_shal = torch.sum(S_mat_shal * S_mat_shal, dim=(1, 2))
        expected_trace_shallow += torch.sum(trace_shal ** 2).item()

        # --- 3. SIMPLE KRONECKER SKETCH ---
        # ||A_1 \otimes ... \otimes A_N||_F^2 = \prod ||A_i||_F^2
        # Calculate squared frobenius norm for each leaf: shape (B, N)
        leaves_frob_sq = torch.sum(leaves * leaves, dim=(2, 3))
        # Product across the N dimension: shape (B,)
        trace_kron = torch.prod(leaves_frob_sq, dim=1)
        expected_trace_kronecker += torch.sum(trace_kron ** 2).item()

        # Memory Cleanup for Colab
        del leaves, X1, X2, g_raw, g_nodes, current_lin, layer, next_layer, S_mat_shal, L, R, g
        leaves_frob_sq = None
        trace_kron = None
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Calculate final variances
    linear_var = (expected_trace_linear / M) - (d ** (2 * N))
    shallow_var = (expected_trace_shallow / M) - (d ** (2 * N))
    kron_var = (expected_trace_kronecker / M) - (d ** (2 * N))

    return linear_var, shallow_var, kron_var


def analyze_parameter_effects(m_list, N_list, base_m=2, base_N=3, base_d=3, M=int(1e4)):
    save_dir = "variance_results"
    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running computation on: {device.upper()} ---")

    base_params = {'m': base_m, 'N': base_N, 'd': base_d}
    param_grids = {'m': m_list, 'N': N_list}

    for param_name, param_values in param_grids.items():
        linear_vars, shallow_vars, kron_vars = [], [], []
        print(f"\n--- Running experiments varying '{param_name}' ---")

        for val in param_values:
            curr_params = base_params.copy()
            curr_params[param_name] = val
            print(f"Running with: {curr_params}")

            lin_var, shal_var, kron_var = calculate_variances_batched(
                m=curr_params['m'],
                N=curr_params['N'],
                d=curr_params['d'],
                M=M,
                batch_size=30,
                device=device
            )

            linear_vars.append(lin_var)
            shallow_vars.append(shal_var)
            kron_vars.append(kron_var)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(param_values, linear_vars, marker='o', label='Linear Variance')
        ax.plot(param_values, shallow_vars, marker='s', label='Shallow Variance')
        ax.plot(param_values, kron_vars, marker='^', linestyle='-.', label='Simple Kronecker Variance')

        ax.set_title(f'Effect of Varying {param_name.upper()}')
        ax.set_xlabel(f'{param_name} values')
        ax.set_ylabel('Variance')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        if all(isinstance(x, int) for x in param_values):
            ax.set_xticks(param_values)

        fixed_params = {key: val for key, val in base_params.items() if key != param_name}
        fixed_str = "Fixed Base Params:\n" + "\n".join([f"{key} = {val}" for key, val in fixed_params.items()])
        ax.text(0.05, 0.95, fixed_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        save_path = os.path.join(save_dir, f"{param_name}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    # Force garbage collection before starting just in case Colab is lingering
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    m_test_list = [100, 150, 200, 250]
    N_test_list = [5, 10, 12, 15]

    analyze_parameter_effects(
        m_list=m_test_list,
        N_list=N_test_list,
        base_m=30,
        base_N=5,
        base_d=2,
        M=int(5000)
    )