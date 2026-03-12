import numpy as np
import itertools
from collections import Counter
from numpy.linalg import matrix_power


def regression_formula(N, m, sigma):
    T = np.array([[m ** 3 * (m - 1), 3 * m ** 3 * (m - 1), 3 * m ** 3 * (m - 1)],
                  [m ** 3, 3 * m ** 2 * (m - 1), 3 * m ** 2 * (m - 1)],
                  [0, m ** 2, m ** 2]])
    g_1_vector = np.array([3*m**5*(m-1),
                           3*m**3*(m**2-1),
                           m**3])

    two_pairs = np.array([1/3*(m*m*m + m*(m-1)*m**2) + 2/3*m*m**2,
                          3*m**2*(m-1) + m**3*(m-1),
                          3*m**2*(m-1) + m**3*(m-1)])
    four_of_a_kind = np.array([0,
                               m**2,
                               m**2])

    count = g_1_vector
    for _ in range(1, N-2):
        count = T @ count

    return np.dot(count, two_pairs) + np.dot(count, four_of_a_kind)


def count_valid_sequences(N, m):
    print(f"Start {N}, {m}")
    num_tensors = N - 1
    num_vars_per_copy = 2 * N - 1
    total_vars = 4 * num_vars_per_copy

    sequence_counts = Counter()
    percent = 0
    for i, combo in enumerate(itertools.product(range(m), repeat=total_vars)):
        curr_percent = i/(m**total_vars) * 100
        if int(curr_percent) > percent:
            print(int(curr_percent))
            percent = int(curr_percent)
        tensors = []
        for c in range(4):
            base = c * num_vars_per_copy
            # chain: left_0, right_0, right_1 ..., right_N-2, int_0, int_1 ..., int_N-2
            L_0 = combo[base]
            rights = combo[base + 1: base + N]
            internals = combo[base + N: base + 2 * N - 1]

            lefts = [L_0] + list(rights[:-1])
            chain = [(lefts[k], rights[k], internals[k]) for k in range(num_tensors)]
            tensors.append(chain)

        # The right index of the last tensor g^(N-1) must match for a=b and c=d
        right_a = tensors[0][-1][-1]
        right_b = tensors[1][-1][-1]
        right_c = tensors[2][-1][-1]
        right_d = tensors[3][-1][-1]

        if right_a != right_b or right_c != right_d:
            continue

        config_seq = []
        for k in range(num_tensors):
            tk = [tensors[c][k] for c in range(4)]
            counts = sorted(Counter(tk).values())

            if counts == [4]:
                config_seq.append(4)
            elif counts == [2, 2]:
                config_seq.append(2)
            else:
                break  # Invalid match, prune branch

        if len(config_seq) == num_tensors:
            sequence_counts[tuple(config_seq)] += 1

    return dict(sequence_counts)


from collections import Counter


def count_tree_sequences_optimized(m):
    # Symmetries: '4' (Identity), '2_1' (a=b!=c=d), '2_2' (a=c!=b=d), '2_3' (a=d!=b=c)
    syms = ['4', '2_1', '2_2', '2_3']

    sequence_counts = Counter()

    # 1. Number of ways to choose an index vector with a specific symmetry
    def z_ways(sym):
        if sym == '4':
            return m
        else:
            return m * (m - 1)

    # 2. Number of ways to choose k_i, k_j such that Tensor T has symmetry S_T, given z has symmetry S_z
    def t_ways(S_T, S_z):
        if S_z == '4':
            if S_T == '4':
                return m ** 2
            else:
                return m ** 4 - m ** 2
        else:
            if S_T == S_z:
                return m ** 4
            else:
                return 0

    # 3. Determine the symmetry of Tensor 3 given the symmetries of its 3 inputs
    def get_t3_sym(S_z1, S_z2, S_n):
        inputs = [S_z1, S_z2, S_n]
        if all(x == '4' for x in inputs):
            return '4'

        # If they are a mix of '4' and exactly ONE type of '2_x', it takes that '2_x' symmetry
        target_2 = None
        for x in inputs:
            if x != '4':
                if target_2 is None:
                    target_2 = x
                elif target_2 != x:
                    return None  # Invalid clash (e.g., trying to merge 2_1 and 2_2)
        return target_2

    # --- The Main Loop ---
    # Iterate over valid symmetries for the bonds z1, z2, n
    # Your condition: n^(a)=n^(b) and n^(c)=n^(d). This restricts 'n' strictly to '4' or '2_1'.
    for Z1 in syms:
        for Z2 in syms:
            for N in ['4', '2_1']:

                # Check if Tensor 3 forms a valid symmetry
                S3 = get_t3_sym(Z1, Z2, N)
                if S3 is None: continue

                # Calculate base combinations for the intermediate bonds
                base_ways = z_ways(Z1) * z_ways(Z2) * z_ways(N)

                # Iterate over target symmetries for Tensor 1 and Tensor 2
                for S1 in syms:
                    for S2 in syms:
                        w1 = t_ways(S1, Z1)
                        w2 = t_ways(S2, Z2)

                        if w1 == 0 or w2 == 0: continue

                        # Total combinations is the product of all independent choices
                        total_combos = base_ways * w1 * w2

                        # Convert internal symmetry labels back to your output format (4 or 2)
                        seq = (
                            4 if S1 == '4' else 2,
                            4 if S2 == '4' else 2,
                            4 if S3 == '4' else 2
                        )
                        sequence_counts[seq] += total_combos

    return dict(sequence_counts)


def closest_dividers(m, D=1):
    """
    Finds exact divisors of m closest to sqrt(m/D).
    """
    split_dimension = np.sqrt(m / D)
    closest_b1, closest_b2 = None, None
    min_diff = float('inf')

    for b1 in range(1, int(m ** 0.5) + 1):
        if m % b1 == 0:
            b2 = m // b1
            diff = abs(b1 - split_dimension)
            if diff < min_diff:
                closest_b1, closest_b2 = b1, b2
                min_diff = diff
    return closest_b1, closest_b2


def create_g_node(m, sigma):
    """
    Creates the internal g node of shape (m, m, m) by contracting two
    i.i.d Gaussian 3-tensors. Uses exact divisors to avoid padding.
    """
    m1, m2 = closest_dividers(m, D=1)

    X1 = np.random.normal(0, sigma, size=(m, m1, m)).astype(np.float32)
    X2 = np.random.normal(0, sigma, size=(m, m2, m)).astype(np.float32)

    g_raw = np.einsum('kil, ljz -> kijz', X1, X2)
    g = g_raw.reshape(m, m, m)

    return g


def estimate_linear_tree_variance(m, N, d, M):
    """
    Estimates E[S^T S \kron S^T S] for the Linear Tree sketch using float32.
    """
    sigma = 1.0 / np.sqrt(m)
    # Explicitly set dtype to float32 for the large expected matrix
    expected_kron = np.zeros((d ** (2 * N), d ** (2 * N)), dtype=np.float32)

    for q in range(M):
        if q % 1000 == 0:
            print("Linear tree:   ", q)
        # Cast leaf nodes to float32
        leaves = [np.random.normal(0, sigma, size=(d, m)).astype(np.float32) for _ in range(N)]
        T = leaves[0]

        for i in range(1, N):
            g = create_g_node(m, sigma)
            T_new = np.einsum('al, br, lrp -> abp', T, leaves[i], g)
            T = T_new.reshape(-1, m)

        S = T.T
        STS = S.T @ S
        expected_kron += np.kron(STS, STS)

    return expected_kron / M


def estimate_binary_tree_variance(m, N, d, M):
    """
    Estimates E[S^T S \kron S^T S] for the Shallow Full Binary Tree sketch using float32.
    Builds a perfectly balanced complete binary tree topology.
    """
    sigma = 1.0 / np.sqrt(m)
    expected_kron = np.zeros((d ** (2 * N), d ** (2 * N)), dtype=np.float32)
    D = int(np.ceil(np.log2(N)))
    power_of_2 = 2 ** D
    leaves_to_pair = 2 * N - power_of_2

    for q in range(M):
        if q % 1000 == 0:
            print("Shallow tree:   ", q)
        # Generate N leaf nodes
        layer = [np.random.normal(0, sigma, size=(d, m)).astype(np.float32) for _ in range(N)]

        next_layer = []
        for j in range(0, leaves_to_pair, 2):
            L, R = layer[j], layer[j + 1]
            g = create_g_node(m, sigma)
            parent = np.einsum('al, br, lrp -> abp', L, R, g).reshape(-1, m)
            next_layer.append(parent)

        # Append the rest of the untouched leaves that sit one level higher
        next_layer.extend(layer[leaves_to_pair:])
        layer = next_layer

        # 3. Now len(layer) is guaranteed to be a perfect power of 2.
        # We can do perfect level-by-level pairwise contractions up to the root.
        while len(layer) > 1:
            next_layer = []
            for j in range(0, len(layer), 2):
                L, R = layer[j], layer[j + 1]
                g = create_g_node(m, sigma)
                parent = np.einsum('al, br, lrp -> abp', L, R, g).reshape(-1, m)
                next_layer.append(parent)
            layer = next_layer

        S = layer[0].T
        STS = S.T @ S
        expected_kron += np.kron(STS, STS)

    return expected_kron / M


def create_rademacher_kron(d, N):
    # Calculate the dimension D = d^N
    D = d ** N

    # 1. Create the Rademacher i.i.d matrix A
    # np.random.choice is perfect for uniformly sampling -1 and 1
    A = np.random.choice([-1, 1], size=(D, D))

    # 2. Compute A \otimes A using np.kron
    A_kron_A = np.kron(A, A)

    return A, A_kron_A


import os
import numpy as np
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power


# Assuming your custom functions are defined above this script:
# def estimate_linear_tree_variance(m, N, d, M): ...
# def estimate_binary_tree_variance(m, N, d, M): ...
# def create_rademacher_kron(d, N): ...

def calculate_variances(m, N, d, k, M):
    linear = estimate_linear_tree_variance(m=m, N=N, d=d, M=M)
    shallow = estimate_binary_tree_variance(m=m, N=N, d=d, M=M)
    A, A_kron_A = create_rademacher_kron(d=d, N=N)

    A_trace = np.trace(matrix_power(A, k))

    linear_variance =  np.trace(matrix_power(linear @ A_kron_A, k)) - A_trace ** 2
    shallow_variance = np.trace(matrix_power(shallow @ A_kron_A, k)) - A_trace ** 2

    return np.real(linear_variance), np.real(shallow_variance)


def analyze_parameter_effects(m_list, N_list, k_list,
                              base_m=2, base_N=3, base_d=3, base_k=1, M=int(1e4)):
    save_dir = "variance_results"
    os.makedirs(save_dir, exist_ok=True)

    base_params = {'m': base_m, 'N': base_N, 'd': base_d, 'k': base_k}

    param_grids = {
        'm': m_list,
        'N': N_list,
        'k': k_list
    }

    for param_name, param_values in param_grids.items():
        linear_vars = []
        shallow_vars = []

        print(f"\n--- Running experiments varying '{param_name}' ---")

        for val in param_values:
            curr_params = base_params.copy()
            curr_params[param_name] = val

            print(f"Running with: {curr_params}")

            lin_var, shal_var = calculate_variances(
                m=curr_params['m'],
                N=curr_params['N'],
                d=curr_params['d'],
                k=curr_params['k'],
                M=M
            )

            linear_vars.append(lin_var)
            shallow_vars.append(shal_var)

        # Create figure silently in memory
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(param_values, linear_vars, marker='o', label='Linear Variance')
        ax.plot(param_values, shallow_vars, marker='s', label='Shallow Variance')

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

        # Explicitly close the figure to free up memory instantly
        plt.close(fig)

        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    m_test_list = [2, 3,4,5, 6]
    N_test_list = [ 3, 4, 5,6]
    k_test_list = [1, 2, 3, 5]

    analyze_parameter_effects(
        m_list=m_test_list,
        N_list=N_test_list,
        k_list=k_test_list,
        base_m=2,
        base_N=3,
        base_d=3,
        base_k=1,
        M=int(3e4)
    )