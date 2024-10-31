from typing import List
import numpy as np
from tensornetwork import Node, Edge
import tensornetwork as tn


class TensorNetwork:

    def __init__(self, v: List[Node], edge_list: List):
        """
        A tensor graph object
        :param v: The nodes of the graph
        :param edge_list:  Each element represent an edge in a hyper-graph of a tensor network.
                   Each element is of the form <[i, dim_to_contract], [j, dim_to_contract]>
        """
        self._v = v
        self._save_original_tensor()
        self._edge_list = edge_list
        self._e = self._parse_edges()
        self._edges_to_sketch = self._get_edges_to_sketch()

        self.contractions_cost = 0

    def _save_original_tensor(self):
        """
        Saves the original nodes for later comparison. Assumes the nodes a before the edge connecting
        """
        nodes_copies = []
        for node in self._v:
            nodes_copies.append(Node(node.tensor.copy(), name=node.name))
        self._original_tensor = nodes_copies

    def sketch(self, edge: Edge, m: int) -> None:
        """
        Sketches the node which dangles edge
        :param m: The dimension to sketch v[i]'s dangling dimensions
        :param edge: The edge of v[i] to sketch
        """
        assert edge.node2 is None, f"Tried to sketch {edge}, but it is connected to {edge.node2}"
        node = edge.node1
        sketched_node = node
        dim = edge.axis1

        sketching_node = Node(np.random.randn(m, sketched_node.shape[dim]) / np.sqrt(m))
        sketch_edge = sketched_node[dim] ^ sketching_node[1]
        self._count_contraction_cost(sketching_node, node)
        sketched_node = tn.contract(sketch_edge)

        sketched_node.set_name(f"s_{node.name}")

        for i in range(len(self._v)):
            if self._v[i] == node:
                self._v[i] = sketched_node

    def get_original_tensor(self) -> np.ndarray:
        """
        Gets the actual original tensor before embedding
        :return: The original tensor
        """
        nodes_amount = len(self._original_tensor)
        for i in range(nodes_amount):
            u = self._original_tensor[i]
            for j in range(i + 1, nodes_amount):
                v = self._original_tensor[j]
                if set(v.edges).intersection(set(u.edges)):
                    uv = u @ v
                    uv.set_name(u.name + "_" + v.name)
                    self._original_tensor[i] = uv
                    self._original_tensor[j] = uv
        orig = self._original_tensor[-1].tensor
        del self._original_tensor  # Trying to save memory
        return orig

    def contract(self, i: int, j: int) -> None:
        """
        Contracts the node in index i with index j
        :param i: The index of the node to contract
        :param j: The index of the node to contract
        """
        u = self._v[i]
        v = self._v[j]
        self._count_contraction_cost(u, v)
        uv = u @ v
        uv.set_name(u.name + v.name)
        for i in range(len(self._v)):
            if self._v[i] in [u, v]:
                self._v[i] = uv

    def _parse_edges(self) -> List:
        """
        Takes the edges received in the constructor and parses them for each node. i.e. builds the tensors network
        """
        edges = []
        for u_tuple, v_tuple in self._edge_list:
            u_index, u_dim = u_tuple
            v_index, v_dim = v_tuple

            u = self._v[u_index]
            v = self._v[v_index]

            u_copy = self._original_tensor[u_index]
            v_copy = self._original_tensor[v_index]

            try:
                edges.append(u[u_dim] ^ v[v_dim])
                u_copy[u_dim] ^ v_copy[v_dim]  # Saves the original tensor for later compare
            except Exception as e:
                print(f"Problem with {u.name} and {v.name} with {u_dim} -> {v_dim}")
                raise e
        return edges

    def _get_edges_to_sketch(self) -> List[Node]:
        """
        Returns the nodes with uncontracted dimensions in the graph.
        The node appears the amount of dimension to sketch he has
        In the paper returns nodes with edges in E_OVERLINE
        :return: A list of edges to be sketched
        """
        edges_to_sketch = []
        for node in self._v:
            edges_to_sketch += node.get_all_dangling()
        return edges_to_sketch

    def get_edges_to_sketch(self) -> List[Node]:
        """
        Returns the nodes of the graph with uncontracted dimensions.
        These edges should be sketched, i.e. edges in E_1
        :return: The edges that are uncontracted
        """
        return self._edges_to_sketch

    def __getitem__(self, key) -> Node:
        """
        Returns the node at index key
        :param key: the index of the node to get
        :return: A node at the index
        """
        return self._v[key]

    def get_node_index(self, v: Node):
        """
        Returns which index in the network is the node
        :param v: The node to check for
        :return: The index of the node
        """
        return self._v.index(v)

    def _count_contraction_cost(self, U: Node, V: Node) -> None:
        """
        Counts the cost of contractions between the two given nodes
        :param U: The first node the contract
        :param V: The second node to contract
        """
        U_size = U.tensor.size
        V_size = V.tensor.size
        shared_dims = [u_edge.dimension for u_edge in U.edges if u_edge in V.edges]
        contraction_cost = U_size * V_size / np.prod(shared_dims)
        self.contractions_cost += contraction_cost

    def tree_sketch_and_contract(self, i: int, j: int, m: int) -> None:
        """
        In case we want to tree embed and not use the actuall paper algorithm.
        :param i: The index for the first node
        :param j: The index for the second node
        :param m: The dimension size to embed to
        """
        u_orig = self[i]
        v_orig = self[j]

        cut_u_v_edges = [e for e in u_orig.edges if e in v_orig.edges]
        assert len(cut_u_v_edges), "Experiments require only cases where there is a single edge between the two"

        v_dangling = [e for e in v_orig.get_all_dangling() if e.dimension == m][0]
        u_dangling = [e for e in u_orig.get_all_dangling() if e.dimension == m][0]

        new_node = Node(np.random.randn(m, m, m) / np.sqrt(m))
        new_node[0] ^ u_orig[u_dangling.axis1]
        new_node[1] ^ v_orig[v_dangling.axis1]
        self.contract(i, j)

        u = self._v[i]

        self._count_contraction_cost(new_node, u)
        new_node = u @ new_node
        new_node.set_name(f"tree_{u.name}")

        for i in range(len(self._v)):
            if self._v[i] == u:
                self._v[i] = new_node

    @staticmethod
    def _closest_dividers(m: int, split_dimension: float):
        """
        In order to divide the dimension of m as close to the factor as we can, we find the closest divider possible.
        :param m: The number to get its dividers
        :param split_dimension: The split we want to have
        :return: closest_b1, closest_b2. Such that their product is m and closest_b1 is closest to a
        """
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

    def tn_sketch_and_contract_s(self, i: int, j: int, m: int) -> None:
        """
        Takes two indices that describe nodes that should be contracted under the S partition and does the necessary
        embedding and contracting.
        Follows Appendix C.1 of the paper
        :param i: The index of the first node
        :param j: The index of the second node
        :param m: The sketching size
        """
        u_orig = self[i]
        v_orig = self[j]

        cut_u_v_edges = [e for e in u_orig.edges if e in v_orig.edges]
        assert len(cut_u_v_edges), "Experiments require only cases where there is a single edge between the two"
        cut_u_v = np.prod([e.dimension for e in cut_u_v_edges])

        a_i = u_orig.tensor.size // cut_u_v
        c_i = v_orig.tensor.size // cut_u_v
        b_i = cut_u_v

        # Resolves the symmetry of both cases
        if a_i <= c_i:
            u = u_orig
            v = v_orig
            factor = c_i / b_i
        else:
            u = v_orig
            v = u_orig
            factor = a_i / b_i

        v_dangling = [e for e in v.get_all_dangling() if e.dimension == m][0]
        u_dangling = [e for e in u.get_all_dangling() if e.dimension == m][0]

        split_dim_1, split_dim_2 = self._closest_dividers(m, np.sqrt(m * factor))

        # Reshape v accordingly
        e_v1, e_v2 = tn.split_edge(v_dangling, [split_dim_1, split_dim_2])

        # Create z_i tree tensor
        v_1 = Node(np.random.randn(m, split_dim_1, m) / np.sqrt(m))
        v_2 = Node(np.random.randn(m, split_dim_2, m) / np.sqrt(m))

        # Connect the relevant edges
        v_1[1] ^ v[e_v1.axis1]
        v_2[1] ^ v[e_v2.axis1]

        v_1[0] ^ u[u_dangling.axis1]
        v_1[2] ^ v_2[0]

        # contract nodes
        self._count_contraction_cost(v_1, u)
        new_node = v_1 @ u
        self._count_contraction_cost(new_node, v)
        new_node = new_node @ v
        self._count_contraction_cost(new_node, v_2)
        new_node = new_node @ v_2

        new_node.set_name(f"z_i_{u_orig.name}_{v_orig.name}")
        for i in range(len(self._v)):
            if self._v[i] in [u_orig, v_orig, v]:
                self._v[i] = new_node