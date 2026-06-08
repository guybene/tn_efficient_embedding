import numpy as np
import itertools
from collections import Counter
from numpy.linalg import matrix_power
from tensornetwork import Node
import tensornetwork as tn

def novel_combinatorics_count(m, N):
    """
    Calculates the pure combinatorial path count for the 4K state
    across a tensor network of length N, stripped of all expectations.
    """
    if N < 3:
        raise ValueError("Network length N must be at least 3.")

    # 1. Initial State Vector (u_start) stripped of mu
    u_start = np.array([
        (m**5) * (m - 1),
        2 * (m**5) * (m - 1),
        (m**3) * (3 * m**2 - 2)
    ], dtype=object) # Using object dtype to prevent integer overflow for large m, N

    # 2. Transition Matrix (W_4K) stripped of mu
    W_4K = np.array([
        [(m**3) * (m - 1), 0, (m**3) * (m - 1)],
        [0, (m**3) * (m - 1), 2 * (m**3) * (m - 1)],
        [m**3, m**3, (m**2) * (3 * m - 2)]
    ], dtype=object)

    # 3. Boundary Vector (u_end) stripped of mu
    u_end = np.array([
        m**4,
        m**3,
        m**4 + 2*(m**3) - 2*(m**2)
    ], dtype=object)

    # 4. Calculate (W_4K)^(N-3)
    # Using np.linalg.matrix_power works, but since we are using 'object' arrays
    # to handle massive integers safely, manual successive multiplication is safer.
    W_power = np.eye(3, dtype=object)
    for _ in range(N - 3):
        W_power = np.dot(W_power, W_4K)

    # 5. Compute the final scalar count: u_end^T * W^(N-3) * u_start
    intermediate_state = np.dot(W_power, u_start)
    total_count = np.dot(u_end, intermediate_state)

    return total_count


def original_regression_formula(m, N):
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


def calculate_binary_vector(N, m):
    """
    Calculates the final state vector for a single path in the tensor network.

    N: Number of leaves (assumed to be a power of 2, N >= 4)
    m: Dimension of the indices
    """
    if N < 4:
        raise ValueError("N must be at least 4. For N=2, the path length N/2 - 2 would be negative.")

    # Define the transition matrix U based on your equations
    U = np.array([
        [m ** 2 * (m - 1) ** 2, 2 * m ** 2 * (m - 1) ** 2, 3 * m ** 2 * (m - 1) ** 2, 3 * m ** 2 * (m - 1) ** 2],
        [m ** 2 * (m - 1), 4 * m ** 2 * (m - 1), 6 * m ** 2 * (m - 1), 6 * m ** 2 * (m - 1)],
        [m ** 2, 0.5 * m ** 2, 0, 0],
        [0, 0.5 * m ** 2, m ** 2, m ** 2]
    ], dtype=float)

    # Define the starting vector u_start based on your equations
    u_start = np.array([
        [m ** 4 * (m - 1) ** 2],
        [2 * m ** 4 * (m - 1)],
        [m ** 3 * (m - 1)],
        [m ** 3]
    ], dtype=float)

    # The number of intermediate steps is
    transitions = int(N  - 1)

    print(f"Calculating for N={N}, m={m}")
    print(f"Raising matrix U to the power of {transitions}...")

    # Calculate U^(N/2 - 2)
    U_pow = np.linalg.matrix_power(U, transitions)

    # Multiply the powered matrix by the starting vector
    final_vector = U_pow @ u_start

    return final_vector

def count_valid_tree_sequences_optimized(N, m):
    num_tensors = N - 1

    if not (N > 0 and (N & (N - 1)) == 0):
        print("Warning: N should ideally be a power of 2 for a perfect binary tree.")

    print(f"Starting optimized brute-force for N={N}, m={m}")

    # INDEPENDENT VARIABLES:
    # 1. We force the boundary condition: U^(a)=U^(b) (1 var) and U^(c)=U^(d) (1 var). -> 2 variables total.
    # 2. Each replica still needs its own L and R indices for every tensor. -> 2 * num_tensors per replica.
    legs_per_replica = 2 * num_tensors
    total_leg_vars = 4 * legs_per_replica

    # Total variables is now 2 + total_leg_vars
    total_vars = 2 + total_leg_vars

    sequence_counts = Counter()
    total_iterations = m ** total_vars
    print(f"Total combinations to check: {total_iterations:,} (Reduced by a factor of m^2)\n")

    percent = 0
    # Generate the 2 root variables first, then all the leg variables
    for i, combo in enumerate(itertools.product(range(m), repeat=total_vars)):
        curr_percent = (i / total_iterations) * 100
        if int(curr_percent) > percent:
            print(f"Progress: {int(curr_percent)}%")
            percent = int(curr_percent)

        # Extract the pre-constrained root variables
        U_ab = combo[0]
        U_cd = combo[1]

        # The rest of the array belongs to the L and R legs
        legs = combo[2:]

        tensors = []
        for c in range(4):
            # Assign the correct U_root based on replica to maintain the constraint
            if c == 0 or c == 1:  # Replica a or b
                U_root = U_ab
            else:  # Replica c or d
                U_root = U_cd

            base = c * legs_per_replica
            L = legs[base: base + num_tensors]
            R = legs[base + num_tensors: base + 2 * num_tensors]

            # Build the tree topology dynamically
            replica_tensors = []
            for k in range(num_tensors):
                if k == 0:
                    U_k = U_root  # Root node
                elif k % 2 == 1:
                    parent_idx = (k - 1) // 2
                    U_k = L[parent_idx]  # Left child
                else:
                    parent_idx = (k - 2) // 2
                    U_k = R[parent_idx]  # Right child

                replica_tensors.append((U_k, L[k], R[k]))

            tensors.append(replica_tensors)

        # No need to check boundary conditions—they are guaranteed!

        config_seq = []
        valid_tree = True

        # Check agreement for every tensor in the tree
        for k in range(num_tensors):
            tk = [tensors[c][k] for c in range(4)]
            counts = sorted(Counter(tk).values())

            if counts == [4]:
                config_seq.append(4)
            elif counts == [2, 2]:
                config_seq.append(2)
            else:
                valid_tree = False
                break  # Prune early

        if valid_tree:
            sequence_counts[tuple(config_seq)] += 1

    return dict(sequence_counts)


import numpy as np


def sanity_check_4K(m, m1, m2, N):
    """
    Computes the 4K configuration assuming all \mu_X and \mu_g = 1.
    m1 * m2 must equal m.
    """
    if m1 * m2 != m:
        raise ValueError("m1 * m2 must equal m")

    # 1. Initial State Vector u_initial^(4K, 4K)
    u_1 = m ** 3 * (m - 1) * (1 + (m2 - 1) + m2 * (m1 - 1) + (m - 1) + (m2 - 1) * (m - 1) + m2 * (m1 - 1) * (m - 1))
    u_2 = 2 * u_1
    u_3 = m ** 3 * (1 + 3 * (m2 - 1) + 3 * (m1 - 1) + 3 * (m - 1) + 3 * (m1 - 1) * (m - 1) +
                    3 * (m1 - 1) * (m2 - 1) + 3 * (m - 1) * (m2 - 1) + 3 * (m - 1) * (m1 - 1) * (m2 - 1))

    u_initial = np.array([[u_1], [u_2], [u_3]], dtype=object)

    # 2. Transition Matrix W^4K
    W_11 = m ** 2 * (m - 1) * (1 + (m - 1))
    W_22 = W_11
    W_31 = m ** 2 * (1 + (m1 - 1) + m1 * (m2 - 1))
    W_32 = W_31
    W_13 = m ** 2 * (m - 1) * (1 + (m2 - 1) + m2 * (m1 - 1))
    W_23 = 2 * W_13
    W_33 = m ** 2 * (1 + 3 * (m1 - 1) + 3 * (m2 - 1) + 3 * (m1 - 1) * (m2 - 1))

    W = np.array([
        [W_11, 0, W_13],
        [0, W_22, W_23],
        [W_31, W_32, W_33]
    ], dtype=object)

    # 3. Boundary Vector u_end^4K (Sum of Row 1 and Row 3 transposed)
    u_end = np.array([
        [W_11 + W_31],
        [W_32],
        [W_13 + W_33]
    ], dtype=object)

    # Calculate final path count
    transitions = max(0, N - 3)
    W_k = np.linalg.matrix_power(W, transitions)
    total_paths = (u_end.T @ W_k @ u_initial)[0, 0]

    return total_paths





print(novel_combinatorics_count(2,4))
print(original_regression_formula(2,4))
print(sanity_check_4K(2, 2, 1, 4))