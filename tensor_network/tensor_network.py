from typing import List, Optional, Tuple, Dict
import numpy as np
from tensornetwork import Node, Edge
import tensornetwork as tn


from numpy.linalg import norm

class TensorNetwork:

    def __init__(self, v: List[Node], edge_list: List, is_tree_embedding: bool = False):
        """
        A tensor graph object
        :param v: The nodes of the graph
        :param edge_list:  Each element represent an edge in a hyper-graph of a tensor network.
                   Each element is of the form <[i, dim_to_contract], [j, dim_to_contract]>
        :param is_tree_embedding: if true, the contractions in S are treated different
        """
        self._v = v
        self._save_original_tensor()
        self._edge_list = edge_list
        self._e = self._parse_edges()
        self.is_tree_embedding = is_tree_embedding
        self._edges_to_sketch = self._get_edges_to_sketch()

        self.contractions_cost = 0

    def _save_original_tensor(self):
        """
        Saves the original nodes for later comparison. Assumes the nodes a before the edge connecting
        """
        nodes_copies = []
        for node in self._v:
            nodes_copies.append(Node(node.tensor.copy(),name=node.name))
        self._original_tensor = nodes_copies

    def sketch(self, edge: Edge, m: int) -> None:
        """
        Sketches the node which dangles edge
        :param m: The dimension to sketch v[i]'s dangling dimensions
        :param edge: The edge of v[i] to sketch
        """
        print(f"Sketching {edge.node1}")
        assert edge.node2 is None, f"Tried to sketch {edge}, but it is connected to {edge.node2}"
        node = edge.node1
        print("Size:", norm(node.tensor), "Shape:", node.shape)
        sketched_node = node
        dim = edge.axis1

        sketching_node = Node(np.random.randn(m, sketched_node.shape[dim]) / np.sqrt(m))
        sketch_edge = sketched_node[dim] ^ sketching_node[1]
        self._count_contraction_cost(sketching_node, node)
        sketched_node = tn.contract(sketch_edge)
        print("Size:", norm(sketched_node.tensor), "Shape:", sketched_node.shape)

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
            for j in range(i+1, nodes_amount):
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
        print(f"Contracting: {i} - {j}")
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

    def _tree_embed(self, i: int, j: int, m: int) -> None:
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

    def sketch_and_contract_s(self, i: int, j: int, m: int) -> None:
        """
        Takes two indices that describe nodes that should be contracted under the S partition and does the necessary
        embedding and contracting.
        Follows Appendix C.1 of the paper
        :param i: The index of the first node
        :param j: The index of the second node
        :param m: The sketching size
        """
        if self.is_tree_embedding:
            self._tree_embed(i, j, m)
            return

        u_orig = self[i]
        v_orig = self[j]

        cut_u_v_edges = [e for e in u_orig.edges if e in v_orig.edges]
        assert len(cut_u_v_edges), "Experiments require only cases where there is a single edge between the two"
        cut_u_v = np.prod([e.dimension for e in cut_u_v_edges])

        a_i = u_orig.tensor.size // cut_u_v
        c_i = v_orig.tensor.size // cut_u_v
        b_i = cut_u_v

        if a_i <= c_i:
            u = u_orig
            v = v_orig
            factor = int(b_i // c_i + 1)
        else:
            u = v_orig
            v = u_orig
            factor = int(b_i // a_i + 1)

        v_dangling = [e for e in v.get_all_dangling() if e.dimension == m][0]
        u_dangling = [e for e in u.get_all_dangling() if e.dimension == m][0]

        m_sketching = int(np.sqrt(m * factor))
        b_i_sketching = int(np.sqrt(m / factor))

        self.sketch(v_dangling, m_sketching)
        v = self[j]

        e = [e for e in u.edges if e in v.edges][0]
        e.disconnect()
        dim_to_sketch = e.axis1 if e.node1 == v else e.axis2
        self.sketch(v[dim_to_sketch], b_i_sketching)
        v = self[j]

        # connect tree nodes
        tree_node_1 = Node(np.random.randn(m, b_i_sketching, m) / np.sqrt(m))
        tree_node_2 = Node(np.random.randn(m, m_sketching, m) / np.sqrt(m))

        # Names like appendix C.1
        tree_node_1[0] ^ u[u_dangling.axis1]
        tree_node_1[1] ^ v[-1]

        tree_node_1[2] ^ tree_node_2[2]
        tree_node_2[1] ^ v[-2]

        # contract nodes
        self._count_contraction_cost(tree_node_1, u)
        new_node = tree_node_1 @ u
        self._count_contraction_cost(new_node, v)
        new_node = new_node @ v
        self._count_contraction_cost(new_node, tree_node_2)
        new_node = new_node @ tree_node_2

        new_node.set_name(f"z_i_{u_orig.name}_{v_orig.name}")
        for i in range(len(self._v)):
            if self._v[i] in [u_orig, v_orig, v]:
                self._v[i] = new_node