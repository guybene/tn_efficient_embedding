import numpy as np
import string
import opt_einsum as oe
from tensornetwork import Node
import matplotlib.pyplot as plt
import matplotlib
import tensornetwork as tn

import torch
tn.set_default_backend("pytorch")

matplotlib.use('Qt5Agg')
plt.ion()

def generate_symmetrical_mpos_symmetricly(d, D, N, K, is_mpop=False):
    """
    Generates K symmetrical MPOs with the specified parameters.

    Args:
        d (int): Physical dimension.
        D (int): Bond dimension.
        N (int): Length of the MPO.
        K (int): Number of MPOs to generate.


    Returns:
        list: A list of K items, where each item is a list of [Node, Node, ...].
    """
    rng = np.random.default_rng()

    names = string.ascii_uppercase
    results = []

    for _ in range(K):
        # 1. Create random tensors A
        A = [rng.standard_normal((d, D, D), dtype=np.float64) for _ in range(N)]

        # 2. Convert to symmetric MPO tensors W
        W = []
        for Ai in A:
            # Wi shape: (d, Dl, Dr, d)
            Wi = np.einsum('sab,tab->sabt', Ai, Ai, optimize=True)
            W.append(Wi)

        # 3. Apply slicing to endpoints
        # Fix left leg of first tensor and right leg of last tensor to 0
        W[0] = W[0][:, 0, :, :]
        W[-1] = W[-1][:, 0, :, :]

        # 4. Create Nodes
        nodes = [Node(W[i], name=names[i % len(names)]) for i in range(len(W))]

        # Append just the list of nodes to the results
        results.append(nodes)

    #Make the actual tensor symmetric again
    if not is_mpop:
        symmetrical_results = [[node.copy() for node in mpo] for mpo in results]
        for i in range(1,K):
            symmetrical_results.append([node.copy() for node in results[-i-1]])
        results = symmetrical_results
    else:
        actual_results = []
        for i in range(K):
            actual_results.append([node.copy() for node in results[0]])
        results = actual_results
    return results


def calc_optimal_path_for_tensor_network(nodes):
    edge_map = {}
    current_edge_id = 0
    einsum_args = []
    for node in nodes:
        # Add the tensor
        einsum_args.append(node.tensor)

        # Create/Get integer IDs for edges
        node_indices = []
        for edge in node.edges:
            if edge not in edge_map:
                edge_map[edge] = current_edge_id
                current_edge_id += 1
            node_indices.append(edge_map[edge])

        # Add the indices
        einsum_args.append(node_indices)


    path, path_info = oe.contract_path(*einsum_args, [], optimize='greedy')
    return path, path_info


def calc_trace_of_product_mpo(mpos_input):
    # 1. Recreate Nodes to ensure a clean graph
    #    (We copy the data into new Nodes so we don't mess up original references)
    mpo_list = []
    for mpo in mpos_input:
        mpo_list.append([Node(n.tensor) for n in mpo])

    K = len(mpo_list)  # Number of MPOs
    N = len(mpo_list[0])  # Length of each MPO

    for k in range(K):
        for j in range(N - 1):
            # Connect Right leg (-2) of site j to Left leg (1) of site j+1
            tn.connect(mpo_list[k][j][-2], mpo_list[k][j + 1][1])

    # 3. Connect Vertical Edges (Physical Dimensions) & Trace
    #    We process this column by column

    for j in range(N):
        col = [mpo_list[k][j] for k in range(K)]

        # A. Connect Product Chain: MPO_k (In) -> MPO_k+1 (Out)
        #    Top Node's In leg (-1) connects to Bottom Node's Out leg (0)
        for k in range(K - 1):
            tn.connect(col[k][-1], col[k + 1][0])

        # B. Connect Trace: MPO_0 (Out) -> MPO_K-1 (In)
        #    Connect Top Node's Out leg (0) to Bottom Node's In leg (-1)
        tn.connect(col[0][0], col[-1][-1])

    all_nodes = []
    for mpo in mpo_list:
        for node in mpo:
            all_nodes.append(node)
    return tn.contractors.auto(all_nodes).tensor

def closest_dividers(m, d, D):
    """
    In order to divide the dimension of m as close to the factor as we can, we find the closest divider possible.
    :return: closest_b1, closest_b2. Such that their product is m and closest_b1 is closest to a
    """
    split_dimension = d / D
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

def create_embedding_tree(m, d, D, name, v1_dim, v2_dim):

    v_1 = np.random.randn(m, v1_dim, m) / np.sqrt(m) #Assume (up,mid,down)
    v_2 = np.random.randn(m, v2_dim, m) / np.sqrt(m)


    v_1 = Node(v_1, name=name + "_A")
    v_2 = Node(v_2, name=name + "_B")
    v_1[-1] ^ v_2[0]
    return v_1, v_2



def create_sketching_network_tn(N, d, m, D):
    """
    Creates a sketching tensor network for an MPO using the tensornetwork library.

    Structure:
    - Leaves: N matrices (d, m) that sketch the physical indices.
    - Spine: A chain of N-1 core tensors (m, m, m) that fuse the sketches iteratively.

    Args:
        N (int): Number of MPO nodes.
        d (int): Physical dimension size (input dimension).
        m (int): Sketch dimension size (bond dimension).

    Returns:
        nodes (list): All tn.Node objects created (leaves and cores).
        physical_edges (list): The N dangling edges (dimension d) waiting for MPO connection.
        final_sketch_edge (tn.Edge): The single output edge (dimension m) of the entire sketch.
    """
    v1_dim, v2_dim = closest_dividers(m, d, D)
    # Set backend to numpy for Gaussian generation
    tn.set_default_backend("numpy")

    all_nodes = []
    leaves = []
    physical_edges = []

    # --- 1. Create the Leaf Nodes (Kronecker Embedding) ---
    # These project the physical dimension 'd' down to 'm'.
    for i in range(N):
        # Normalized Gaussian initialization
        matrix_data = np.random.randn(d, m) / np.sqrt(m)
        matrix_data = matrix_data.astype(np.float64)
        leaf = tn.Node(matrix_data, name=f"Leaf_{i}")

        leaves.append(leaf)
        all_nodes.append(leaf)

        # Leaf edges:
        # index 0 is 'd' (Physical - connects to MPO)
        # index 1 is 'm' (Sketch - connects to Spine)
        physical_edges.append(leaf[0])

    # --- 2. Create the Spine and Connect (Iterative Fusion) ---
    # We fuse the leaves one by one into a central spine.

    # Initial Step: Fuse Leaf 0 and Leaf 1
    # Core tensor shape: (m, m, m) -> (Input_Left, Input_Right, Output)
    v_1, v_2 = create_embedding_tree(m, d,D, "Core_0", v1_dim, v2_dim)
    all_nodes.append(v_1)
    all_nodes.append(v_2)

    # Actual Connections using tn.connect
    v1_edge, v_2_edge = tn.split_edge(leaves[1][1], (v1_dim, v2_dim))

    tn.connect(leaves[0][1], v_1[0])
    tn.connect(v1_edge, v_1[1])
    tn.connect(v_2_edge, v_2[1])

    # The output of this fusion
    current_bond_edge = v_2[2]

    # Iterative Steps: Fuse the rest of the leaves (Leaf 2 to Leaf N-1)
    for i in range(2, N):
        # Create next core tensor
        v_1, v_2 = create_embedding_tree(m, d, D, f"Core_{i - 1}", v1_dim, v2_dim)
        all_nodes.append(v_1)
        all_nodes.append(v_2)

        # Connect the accumulated spine (current_bond_edge) to the new core
        tn.connect(current_bond_edge, v_1[0])

        # Connect the next leaf (Leaf i) to the new core
        v1_edge, v_2_edge = tn.split_edge(leaves[i][1], (v1_dim, v2_dim))
        tn.connect(v1_edge, v_1[1])
        tn.connect(v_2_edge, v_2[1])

        # Update the bond edge to be the output of this new core
        current_bond_edge = v_2[2]

    return all_nodes, physical_edges, current_bond_edge


def deep_copy_sketching_network(original_nodes, original_physical_edges, original_final_edge):
    """
    Creates a deep copy of the sketching network and returns the corresponding
    connectors for the new network.

    Args:
        original_nodes (list): List of tn.Node objects from the original network.
        original_physical_edges (list): List of tn.Edge objects (dangling) used to connect to MPO.
        original_final_edge (tn.Edge): The final output edge of the spine.

    Returns:
        new_nodes (list): The new independent nodes.
        new_physical_edges (list): The new dangling edges to connect to the MPO.
        new_final_edge (tn.Edge): The new final output edge.
    """
    node_map = {}
    new_nodes_list = []

    # 1. Duplicate Nodes (Deep Copy Data)
    for node in original_nodes:
        # Copy the numpy tensor data to ensure independence
        new_data = node.tensor.copy()
        new_node = tn.Node(new_data, name=node.name)

        node_map[node] = new_node
        new_nodes_list.append(new_node)

    # 2. Reconstruct Internal Connections
    # We track processed edges to avoid double-connecting
    processed_original_edges = set()

    for original_node in original_nodes:
        new_node = node_map[original_node]

        for edge in original_node.edges:
            if edge in processed_original_edges:
                continue

            # If the edge is internal (connects two nodes), recreate the connection
            if not edge.is_dangling():
                node1, node2 = edge.node1, edge.node2
                axis1, axis2 = edge.axis1, edge.axis2

                # Verify both nodes are in our map (part of this network)
                if node1 in node_map and node2 in node_map:
                    new_node1 = node_map[node1]
                    new_node2 = node_map[node2]
                    tn.connect(new_node1[axis1], new_node2[axis2])

            processed_original_edges.add(edge)

    # 3. Retrieve the specific edges for the NEW network
    # We use the node_map and axis indices to find the matching new edges

    new_physical_edges = []
    for edge in original_physical_edges:
        # Find the new node corresponding to the old node
        old_node = edge.node1
        new_node = node_map[old_node]

        # Get the edge at the exact same axis index
        new_edge = new_node[edge.axis1]
        new_physical_edges.append(new_edge)

    # Retrieve the final sketch edge
    old_final_node = original_final_edge.node1
    new_final_node = node_map[old_final_node]
    new_final_edge = new_final_node[original_final_edge.axis1]

    return new_nodes_list, new_physical_edges, new_final_edge




def contract_block_zipper(N, mpo_nodes, top_sketch_nodes, bot_sketch_nodes):
    """
    Manually contracts the block S_top^T * A * S_bot using an optimized zipper path.

    Order:
    1. Contract all Leaves (d->m projection) into the MPO nodes.
    2. Contract the Spines (Gadgets) iteratively from left to right.
    """
    # --- Phase 1: Contract Leaves (Projection) ---
    # Top/Bot sketch nodes structure: [Leaf_0, ..., Leaf_N-1, Core_0, ..., Core_N-2]

    top_leaves = top_sketch_nodes[:N]
    top_cores = top_sketch_nodes[N:]

    bot_leaves = bot_sketch_nodes[:N]
    bot_cores = bot_sketch_nodes[N:]

    projected_sites = []

    for i in range(N):
        # Current MPO node
        node = mpo_nodes[i]

        # 1. Contract Top Leaf into MPO (Axis 0)
        # Note: We must locate the specific edge connecting them.
        # Since we know the structure, we can find the shared edge.
        node = tn.contract_between(node, top_leaves[i])
        node = tn.contract_between(node, bot_leaves[i])

        projected_sites.append(node)
    projected_sites.reverse()
    for i in range(N-1):
        u = projected_sites.pop()
        v = projected_sites.pop()
        u_prime = tn.contract_between(u, top_cores[i])
        v_prime = tn.contract_between(v, bot_cores[i])
        contracted_nodes = tn.contract_between(u_prime, v_prime)
        projected_sites.append(contracted_nodes)
    sketched_matrix = projected_sites[0].tensor
    return sketched_matrix

def sketch_k_mpos_optimal(N, d, m, K, D, mpos):
    """
    Creates K MPOs and connects them with sketching networks in the pattern:
    S1 -> A1 -> S2 -> S2^T -> A2 ... -> Ak -> S1^T

    Args:
        N, d, m: Network dimensions.
        K: Number of MPOs to generate.
        D_mpo: Bond dimension for the random MPOs.
    """
    tn.set_default_backend("numpy")
    all_nodes = []

    # 1. Create K Random MPOs
    # We assume 'create_random_mpo' is available from previous steps
    for mpo in mpos:
        all_nodes.extend(mpo)

    # 2. Create and Connect the Outer Cap S1 (Top of A1)
    s1_nodes, s1_phys, s1_final = create_sketching_network_tn(N, d, m, D)
    all_nodes.extend(s1_nodes)

    # Connect S1 to A1 Top (Axis 2)
    for i in range(N):
        tn.connect(s1_phys[i], mpos[0][i][0])

    # 3. Create Intermediate Zippers (S_i, S_i^T)
    # This connects Bottom of A_{k} to Top of A_{k+1}
    for k in range(K - 1):
        # Create S_{k+2} (e.g., S2, S3...)
        s_mid_nodes, s_mid_phys, s_mid_final = create_sketching_network_tn(N, d, m, D)

        # Create S_{k+2}^T (Deep Copy)
        s_mid_T_nodes, s_mid_T_phys, s_mid_T_final = deep_copy_sketching_network(
            s_mid_nodes, s_mid_phys, s_mid_final
        )

        all_nodes.extend(s_mid_nodes)
        all_nodes.extend(s_mid_T_nodes)

        # Connect S_{k+2} to Bottom of A_{k} (Axis 1)
        # Connect S_{k+2}^T to Top of A_{k+1} (Axis 2)
        for i in range(N):
            tn.connect(s_mid_phys[i], mpos[k][i][-1])
            tn.connect(s_mid_T_phys[i], mpos[k + 1][i][0])

        # Connect the spines of S and S^T (Approximating identity)
        tn.connect(s_mid_final, s_mid_T_final)

    # 4. Create and Connect the Bottom Cap S1^T (Bottom of Ak)
    # Deep copy S1 to close the loop
    s1_T_nodes, s1_T_phys, s1_T_final = deep_copy_sketching_network(
        s1_nodes, s1_phys, s1_final
    )
    all_nodes.extend(s1_T_nodes)

    # Connect S1^T to Bottom of Ak (Axis 1)
    for i in range(N):
        tn.connect(s1_T_phys[i], mpos[K - 1][i][-1])

    # 5. Connect the Outer Spines (S1 to S1^T)
    tn.connect(s1_final, s1_T_final)

    return tn.contractors.auto(all_nodes).tensor

def sketch_k_mpos_manual_contraction(N, d, m, K, D, mpos):
    """
    (*) This method contains a more manual contraction, the other one builds the entire sketch
    and run tn.contract.optimal
    Optimized sketching of K MPOs with Manual Contraction.
    Iteratively computes M_i = S_i^T * A_i * S_{i+1} and accumulates.
    We assume that each tensors 0-dimension is down and (-1)-dimension is up
    """
    tn.set_default_backend("numpy")

    # 1. Generate First Sketch (S1) & Save Copy for Loop Closure
    s_curr_nodes, s_curr_phys, s_curr_final = create_sketching_network_tn(N, d, m, D)

    s_first_nodes_copy, s_first_phys_copy, s_first_final_copy = deep_copy_sketching_network(
        s_curr_nodes, s_curr_phys, s_curr_final
    )

    accumulated_matrix = np.eye(m, dtype=np.float64)
    current_bottom_edge = None  # Tracks the open edge at bottom of accumulator

    for k in range(K):
        # 2. Prepare S_next
        if k < K - 1:
            s_next_nodes, s_next_phys, s_next_final = create_sketching_network_tn(N, d, m, D)
            # Deep copy for future use as Top Sketch in next iteration
            s_next_future_nodes, s_next_future_phys, s_next_future_final = deep_copy_sketching_network(
                s_next_nodes, s_next_phys, s_next_final
            )
        else:
            # Wrap around
            s_next_nodes = s_first_nodes_copy
            s_next_phys = s_first_phys_copy
            s_next_final = s_first_final_copy

        # 3. Connect Graph: S_curr -> A_k -> S_next
        current_mpo = mpos[k]
        for i in range(N):
            tn.connect(s_curr_phys[i], current_mpo[i][0])
            tn.connect(s_next_phys[i], current_mpo[i][-1])

        # 4. Manual Contraction (Leaves then Gadgets)
        # We pass the full lists of nodes; the helper extracts leaves/cores
        # sketched_block_matrix = contract_block_zipper(N, current_mpo, s_curr_nodes, s_next_nodes)

        sketched_block_matrix = tn.contractors.auto(current_mpo + s_curr_nodes + s_next_nodes,
                                                    output_edge_order=[s_curr_final, s_next_final] ).tensor

        # 5. Accumulate (Matrix Multiplication)
        accumulated_matrix @= sketched_block_matrix

        # 6. Update pointers
        if k < K - 1:
            s_curr_nodes = s_next_future_nodes
            s_curr_phys = s_next_future_phys
            s_curr_final = s_next_future_final


    return np.trace(accumulated_matrix)

def create_mpo_from_nodes(mpos_data, K, N):
    """
    Copies the mpo data for reuse in tests
    """
    curr_mpo = [[node.copy() for node in mpo] for mpo in mpos_data]
    for k in range(K):
        for j in range(N - 1):
            # Connect Right leg (-2) of site j to Left leg (1) of site j+1
            tn.connect(curr_mpo[k][j][-2], curr_mpo[k][j + 1][1])
    return curr_mpo

if __name__ == "__main__":
    d = 2
    D = 2  # will have bond D^2
    m = 2
    N = 3
    is_mpop = False
    K = 3 # If not MPOP will generate 2K-1 tensors for symmetry
    M = 50000


    mpos_data = generate_symmetrical_mpos_symmetricly(d, D, N, K, is_mpop)
    if not is_mpop: # Update that we actually have 2K - 1 nodes
        K = K * 2 - 1
    all_nodes = []
    for mpo in mpos_data:
        all_nodes += mpo


    from time import time
    start = time()
    actual_trace = calc_trace_of_product_mpo(mpos_data)
    naive_time = time() - start

    expectation = 0
    errors = []

    algo_time = 0


    for i in range(1,M+1):
        if i % 100==0:
            print(i)

        curr_mpos = create_mpo_from_nodes(mpos_data, K, N)

        start = time()
        expectation += sketch_k_mpos_optimal(N, d, m, K, D, curr_mpos)
        algo_time += time() - start

        errors.append(np.abs(actual_trace - expectation/i) / actual_trace)



    plt.plot(np.arange(101,M+1, 10), errors[100::10])
    print(f"Algo time = {algo_time} Naive time = {naive_time}")
    plt.title(f"Trace: {actual_trace}, Final estimator: {expectation/M}, Final Error: {errors[-1]}\n"
              f"d:{d},D:{D},N:{N},m:{m},K:{K},M:{M}")
    plt.show(block=True)

