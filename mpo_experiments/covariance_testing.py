import numpy as np
import itertools
from collections import Counter


def regression_formula(N, m, sigma):
    T = np.array([[m ** 3 * (m - 1), 3 * m ** 3 * (m - 1), 3 * m ** 3 * (m - 1)],
                  [m ** 3, 3 * m ** 2 * (m - 1), 3 * m ** 2 * (m - 1)],
                  [0, m ** 2, m ** 2]])
    g_1_vector = np.array([3*m**5*(m-1), 3*m**3*(m**2-1),m**3])

    two_pairs = np.array([1/3*(m*m*m + m*(m-1)*m**2) + 2/3*m*m**2,
                          3*m**2*(m-1) + m**3*(m-1),
                          3*m**2*(m-1) + m**3*(m-1)])
    four_of_a_kind = np.array([0,
                               m**2,
                               m**2])

    count = g_1_vector
    for _ in range(1, N-3):
        count = T @ count

    return np.dot(count, two_pairs) + np.dot(count, four_of_a_kind)


def count_valid_sequences(N, m):
    num_tensors = N - 1
    num_vars_per_copy = 2 * N - 1
    total_vars = 4 * num_vars_per_copy

    sequence_counts = Counter()

    for i, combo in enumerate(itertools.product(range(m), repeat=total_vars)):
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


counter = {}
for N in [3,4,5]:
    for m in [2,3,4]:
        tensor_counter = count_valid_sequences(N=N, m=m)
        counter[(N,m)] = tensor_counter
with open("tensor_results.txt", "w") as f:
    f.write(str(counter))

N = 4
m = 2
d = 2
sigma = 1

# print(tensor_counter)
# print(sum(tensor_counter.values()))
# regression_ans = regression_formula(N=N,m=m, sigma=sigma)
# print(regression_ans)

