import sympy as sp

# 1. Define symbolic variables
m, m1, m2, s2, s4 = sp.symbols('m m1 m2 s2 s4')
eps = sp.Symbol('eps', positive=True)
PATTERNS = ['A', 'U1', 'U2', '4K']


# 2. Implement Definition 2: Meet of patterns
def meet(K1, K2):
    if K1 == '4K': return K2
    if K2 == '4K': return K1
    if K1 == K2: return K1
    return 'Other'


# 3. Implement Lemma 3: Pattern multiplicity
def N_pat(s, m_val):
    if s == '4K':
        return m_val
    if s in ['A', 'U1', 'U2']:
        return m_val * (m_val - 1)
    return 0


# 4. Implement Lemma 1: Leaf expectation \mu^X
def mu_X(I, B):
    m_val = meet(I, B)
    if m_val == '4K':
        return s4
    if m_val in ['A', 'U1', 'U2']:
        return s2 ** 2
    return 0  # 'Other'


# 5. Admissible inner patterns \mathcal{K}(I) (Definition 2)
def K_admissible(I):
    """
    Returns the set of inner patterns where the expectation does not vanish.
    Mathematically, this is true if and only if I \wedge B is not 'Other'.
    """
    return [B for B in PATTERNS if meet(I, B) != 'Other']

# 6. Implement Lemma 4: Gadget expectation \mu^g
def mu_g(L, R):
    if L == 'Other' or R == 'Other':
        return 0

    # Values from the table in Lemma 4
    val_2P_2P_same = m ** 2 * s2 ** 4
    val_2P_2P_diff = m * s2 ** 4
    val_2P_4K = m * (m - 1) * s2 ** 4 + m * s4 * s2 ** 2
    val_4K_4K = 3 * m * (m - 1) * s2 ** 4 + m * s4 ** 2

    if L == '4K' and R == '4K':
        return val_4K_4K
    elif L == '4K' or R == '4K':
        return val_2P_4K
    elif L == R:
        return val_2P_2P_same
    else:
        return val_2P_2P_diff


# 7. Implement Theorem 1: Transfer matrix formula (Eq. 1)
def compute_W(P_out):
    """
    Computes the 4x4 matrix W^{P_out} in the basis (A, U1, U2, 4K).
    Rows represent Z_out, Columns represent Z_in.
    """
    W = sp.zeros(4, 4)

    for i, Z_out in enumerate(PATTERNS):
        for j, Z_in in enumerate(PATTERNS):

            total_sum = 0

            # \sum_{K \in \mathcal{K}(P_{p+1})}
            for K in K_admissible(P_out):
                mu_X_val = mu_X(P_out, K)

                inner_sum = 0

                # \sum_{K_1, K_2 \in \mathcal{P} | K_1 \wedge K_2 = K}
                for K1 in PATTERNS:
                    for K2 in PATTERNS:
                        if meet(K1, K2) == K:
                            n_k1 = N_pat(K1, m1)
                            n_k2 = N_pat(K2, m2)

                            # \mu^g_{Z_in \wedge K_1, K_2 \wedge Z_out}
                            g_L = meet(Z_in, K1)
                            g_R = meet(K2, Z_out)
                            mu_g_val = mu_g(g_L, g_R)

                            inner_sum += n_k1 * n_k2 * mu_g_val

                total_sum += mu_X_val * inner_sum

            # Multiply by N_{Z_out}(m)
            W[i, j] = sp.expand(N_pat(Z_out, m) * total_sum)

    return W


print("Building matrices dynamically from theorems...")
matrices = {
    "W^{4K}": compute_W('4K'),
    "W^{A}": compute_W('A'),
    "W^{U_1}": compute_W('U1'),
    "W^{U_2}": compute_W('U2')
}

import numpy as np
import matplotlib.pyplot as plt

# 1. Define the specific substitutions for the Gaussian case and m1, m2
gauss_sqrt_subs = {
    s2: 1 / m,
    s4: 3 / m ** 2,
    m1: sp.sqrt(m),
    m2: sp.sqrt(m)
}

# Compile the matrices into fast numpy functions of a single variable (m)
print("\nCompiling SymPy matrices to fast NumPy functions (this takes a moment)...")
numpy_matrix_funcs = {}
for name, W in matrices.items():
    # Substitute all scaling rules so only 'm' remains
    W_scaled = sp.simplify(W.subs(gauss_sqrt_subs))
    # Compile
    numpy_matrix_funcs[name] = sp.lambdify(m, W_scaled, modules="numpy")

# Map your short pattern names to the dictionary keys
pattern_to_key = {
    '4K': "W^{4K}",
    'A': "W^{A}",
    'U1': "W^{U_1}",
    'U2': "W^{U_2}"
}

# 2. Define the m values and calculate the largest eigenvalue for each matrix
m_values = np.linspace(10, 1000, 5)  # You can adjust this range

# Store the maximum eigenvalues for each pattern
# Structure: {pattern: [eig_val_m1, eig_val_m2, ...]}
max_eigenvalues = {pat: [] for pat in pattern_to_key.keys()}

print("Calculating numerical eigenvalues across m...")
for m_val in m_values:
    for pat, key in pattern_to_key.items():
        # Evaluate the matrix for this m
        W_num = numpy_matrix_funcs[key](m_val)

        # Ensure it is treated as a float array
        W_num = np.array(W_num, dtype=float)

        # Calculate eigenvalues
        eigvals = np.linalg.eigvals(W_num)

        # Extract the largest real eigenvalue
        # (Transfer matrices typically have a dominant real eigenvalue)
        max_eig = np.max(np.real(eigvals))
        max_eigenvalues[pat].append(max_eig)

# Convert lists to numpy arrays for easy multiplication
for pat in max_eigenvalues:
    max_eigenvalues[pat] = np.array(max_eigenvalues[pat])

# 3. Receive the list of chains
chains_to_plot = [
    ['4K', '4K', '4K'],
    ['A', 'A', 'A'],
    ['4K', '4K', 'U2'],
    ['A', 'A', 'U1'],
    ['U1', 'U1', 'U1']

]

# 4. Plot the multiplication of the eigenvalues for each chain
plt.figure(figsize=(10, 6))

for chain in chains_to_plot:
    # Start with an array of ones
    chain_product = np.ones(len(m_values))

    # Multiply the eigenvalues element-wise for the given chain
    for pat in chain:
        chain_product *= max_eigenvalues[pat]

    # Plot the result
    label = " $\\rightarrow$ ".join(chain)
    plt.plot(m_values, chain_product, label=label, marker='o', markersize=3)

plt.title("Product of Largest Eigenvalues for Various Chains\n(Gaussian, $s_2=1/m, s_4=3/m^2, m_1=m_2=\sqrt{m}$)")
plt.xlabel("Sketching Parameter ($m$)")
plt.ylabel("Eigenvalue Product")

# Keeping standard linear scales to prevent obscuring the asymptotic behavior
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("chain_eigenvalues_boung.png")
