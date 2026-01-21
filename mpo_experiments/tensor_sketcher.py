import numpy as np
from tensornetwork import Node
import tensornetwork as tn

from mpo_experiments.ISketch import Sketcher


class TensorSketcher(Sketcher):
    """
    The main sketch method, adding Gaussian tree like gadgets to embedd a tensor
    """

    @staticmethod
    def closest_dividers(m, D):
        """
        In order to divide the dimension of m as close to the factor as we can, we find the closest divider possible.
        :return: closest_b1, closest_b2. Such that their product is m and closest_b1 is closest to a
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

    def create_embedding_tree(self, name, v1_dim, v2_dim):
        m = self.m
        v_1 = np.random.randn(m, v1_dim, m).astype(np.float32) / np.sqrt(m) #Assume (up,mid,down)
        v_2 = np.random.randn(m, v2_dim, m).astype(np.float32) / np.sqrt(m)

        v_1 = Node(v_1, name=name + "_A")
        v_2 = Node(v_2, name=name + "_B")
        v_1[-1] ^ v_2[0]
        return v_1, v_2

    def create_sketching_network_tn(self):
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
        v1_dim, v2_dim = self.closest_dividers(m=self.m, D=self.D)
        # Set backend to numpy for Gaussian generation
        tn.set_default_backend("numpy")

        all_nodes = []
        leaves = []
        physical_edges = []

        # --- 1. Create the Leaf Nodes (Kronecker Embedding) ---
        # These project the physical dimension 'd' down to 'm'.
        for i in range(self.N):
            # Normalized Gaussian initialization
            matrix_data = np.random.randn(self.d, self.m).astype(np.float32) / np.sqrt(self.m)
            matrix_data = matrix_data.astype(np.float32)
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
        v_1, v_2 = self.create_embedding_tree(name="Core_0",
                                              v1_dim=v1_dim,
                                              v2_dim=v2_dim)
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
        for i in range(2, self.N):
            # Create next core tensor
            v_1, v_2 = self.create_embedding_tree( name=f"Core_{i - 1}",
                                                   v1_dim=v1_dim,
                                                   v2_dim=v2_dim)
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


    def deep_copy_sketching_network(self, original_nodes, original_physical_edges, original_final_edge):
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

    def contract_single_S_contraction(self,u,v,top_core_1,top_core_2,
                                      bot_core_1,bot_core_2):

        contract_1 = tn.contract_between(top_core_1, top_core_2)
        contract_2 = tn.contract_between(contract_1, u)
        contract_3 = tn.contract_between(contract_2, v)
        contract_4 = tn.contract_between(contract_3, bot_core_1)
        contract_5 = tn.contract_between(contract_4, bot_core_2)
        return contract_5

    def sketch_single_mpo(self, mpo_nodes, top_sketch_nodes, bot_sketch_nodes):
        """
        Sketches a single MPO
        """
        # --- Phase 1: Contract Leaves (Projection) ---
        # Top/Bot sketch nodes structure: [Leaf_0, ..., Leaf_N-1, Core_0, ..., Core_N-2]
        N = self.N
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
        top_cores.reverse()
        bot_cores.reverse()

        for i in range(N-1):
            u = projected_sites.pop()
            v = projected_sites.pop()
            top_core_1 = top_cores.pop()
            top_core_2 = top_cores.pop()
            bot_core_1 = bot_cores.pop()
            bot_core_2 = bot_cores.pop()

            sketched_node = self.contract_single_S_contraction(u=u,
                                                               v=v,
                                                               top_core_1=top_core_1,
                                                               top_core_2=top_core_2,
                                                               bot_core_1=bot_core_1,
                                                               bot_core_2=bot_core_2)

            projected_sites.append(sketched_node)

        sketched_matrix = projected_sites[0].tensor
        return sketched_matrix

    def sketch(self, mpos):
        """
        (*) This method contains a more manual contraction, the other one builds the entire sketch
        and run tn.contract.auto
        Iteratively computes M_i = S_i^T * A_i * S_{i+1} and accumulates.
        We assume that each tensors 0-dimension is down and (-1)-dimension is up
        """
        tn.set_default_backend("numpy")

        # 1. Generate First Sketch (S1) & Save Copy for Loop Closure
        s_curr_nodes, s_curr_phys, s_curr_final = self.create_sketching_network_tn()

        s_first_nodes_copy, s_first_phys_copy, s_first_final_copy = self.deep_copy_sketching_network(
            s_curr_nodes, s_curr_phys, s_curr_final
        )

        accumulated_matrix = np.eye(self.m, dtype=np.float32)
        current_bottom_edge = None  # Tracks the open edge at bottom of accumulator

        for k in range(self.K):
            # 2. Prepare S_next
            if k < self.K - 1:
                s_next_nodes, s_next_phys, s_next_final = self.create_sketching_network_tn()
                # Deep copy for future use as Top Sketch in next iteration
                s_next_future_nodes, s_next_future_phys, s_next_future_final = self.deep_copy_sketching_network(
                    s_next_nodes, s_next_phys, s_next_final
                )
            else:
                # Wrap around
                s_next_nodes = s_first_nodes_copy
                s_next_phys = s_first_phys_copy
                s_next_final = s_first_final_copy

            # 3. Connect Graph: S_curr -> A_k -> S_next
            current_mpo = mpos[k]
            for i in range(self.N):
                tn.connect(s_curr_phys[i], current_mpo[i][0])
                tn.connect(s_next_phys[i], current_mpo[i][-1])

            # 4. Manual Contraction (Leaves then Gadgets)
            sketched_block_matrix = self.sketch_single_mpo(current_mpo, s_curr_nodes, s_next_nodes)

            # 5. Accumulate (Matrix Multiplication)
            accumulated_matrix @= sketched_block_matrix

            # 6. Update pointers
            if k < self.K - 1:
                s_curr_nodes = s_next_future_nodes
                s_curr_phys = s_next_future_phys
                s_curr_final = s_next_future_final
        return np.trace(accumulated_matrix)

    def __str__(self):
        return "Tensor Sketcher"
