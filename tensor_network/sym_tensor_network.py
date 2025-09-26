from typing import List, Optional, Tuple
import numpy as np
from tensornetwork import Node, Edge
import tensornetwork as tn


class SymmetricalTensorNetwork:

    def __init__(self, v: List[Node], edge_list: List[Tuple], symmetry: List[Tuple]):
        """
        A tensor graph object
        :param v: The nodes of the graph
        :param edge_list:  Each element represent an edge in a hyper-graph of a tensor network.
                   Each element is of the form <[i, dim_to_contract], [j, dim_to_contract]>
        :param symmetry: A list of the symmetry the tensor has. Form similar to edge list
        """
        self._v = v
        self._save_original_tensor()
        self._edge_list = edge_list

        self._parse_edges()
        self._edges_to_sketch = self._get_edges_to_sketch()

        self._symmetry = None
        self._symmetry_map = None
        self.cached_kronecker_edges = {}
        self._create_symmetry(symmetry)

        self.contractions_cost = 0



    def _save_original_tensor(self):
        """
        Saves the original nodes for later comparison. Assumes the nodes a before the edge connecting
        """
        nodes_copies = []
        for node in self._v:
            nodes_copies.append(Node(node.tensor.copy(), name=node.name))
        self._original_tensor = nodes_copies

    def _create_symmetry(self, symmetry) -> None:
        """
        Registers the relevant edges needed for out symmetry
        :param symmetry: A list of the symmetry the tensor has. Form similar to edge list
        """
        sym = []
        sym_map = {}
        assert len(symmetry) * 2 == len(self._edges_to_sketch), "Not all sketch dimension got their symmatry" # Only sanity check
        for sym_up, sym_down in symmetry:
            node_up, dim_up = sym_up
            node_down, dim_down = sym_down

            edge_up = self._v[node_up][dim_up]
            edge_down = self._v[node_down][dim_down]

            edge_up_orig = self._original_tensor[node_up][dim_up]
            edge_down_orig = self._original_tensor[node_down][dim_down]

            edge_up.up = True
            edge_up.down = False
            edge_up_orig.up = True
            edge_up_orig.down = False


            edge_down.up = False
            edge_down.down = True
            edge_down_orig.up = False
            edge_down_orig.down = True

            edge_up.symmetry = edge_down
            edge_down.symmetry = edge_up

            # assert edge_up.dimension == edge_down.dimension, "Got symmetry that have different dimensions"
            sym_map[edge_up] = edge_down
            sym_map[edge_down] = edge_up
            sym.append([edge_up, edge_down])

        self._symmetry = sym
        self._symmetry_map = sym_map



    def kronecker_sketch(self, edge: Edge, m: int) -> None:
        """
        Sketches the node which dangles edge
        :param m: The dimension to sketch v[i]'s dangling dimensions
        :param edge: The edge of v[i] to sketch
        """
        assert edge.node2 is None, f"Tried to sketch {edge}, but it is connected to {edge.node2}"
         # If I already sketched the edges symmetrical partner

        node = edge.node1
        sketched_node = node
        dim = edge.axis1

        if edge in self.cached_kronecker_edges:
            sketch_matrix = self.cached_kronecker_edges[edge]
        else:
            sketch_matrix = np.random.randn(m, sketched_node.shape[dim]) / np.sqrt(m)
            self.cached_kronecker_edges[edge] = sketch_matrix
            self.cached_kronecker_edges[self._symmetry_map[edge]] = sketch_matrix

        sketching_node = Node(sketch_matrix)
        new_dangling_edge = sketching_node[0]

        new_dangling_edge.up = edge.up
        new_dangling_edge.down = edge.down


        sketch_edge = sketched_node[dim] ^ sketching_node[1]
        self._count_contraction_cost(sketching_node, node)
        sketched_node = tn.contract(sketch_edge)

        sketched_node.set_name(f"s_{node.name}")
        for i in range(len(self._symmetry)):
            upper, lower = self._symmetry[i]
            if edge == upper:
                self._symmetry[i][0] = new_dangling_edge
                self._symmetry_map.pop(edge)
                self._symmetry_map[new_dangling_edge] = lower
                self._symmetry_map[lower] = new_dangling_edge
                break
            elif edge == lower:
                self._symmetry[i][1] = new_dangling_edge
                self._symmetry_map.pop(edge)
                self._symmetry_map[new_dangling_edge] = upper
                self._symmetry_map[upper] = new_dangling_edge
                break
        for i in range(len(self._v)):
            if self._v[i] == node:
                self._v[i] = sketched_node

    def get_original_symmetrical_matrix(self) -> np.ndarray:
        #This method sort of breaks desing, we go straight to the symetry
        nodes_amount = len(self._original_tensor)
        for i in range(nodes_amount):
            u = self._original_tensor[i]
            for j in range(i + 1, nodes_amount):
                v = self._original_tensor[j]
                if set(v.edges).intersection(set(u.edges)):
                    e_uv = tn.get_shared_edges(u, v)
                    keep = [e for e in list(u.get_all_edges()) + list(v.get_all_edges()) if e not in e_uv]
                    uv = tn.contract_between(u, v, output_edge_order=keep)
                    uv.set_name(u.name + "_" + v.name)
                    self._original_tensor[i] = uv
                    self._original_tensor[j] = uv

        contracted_tensor = self._original_tensor[-1]
        es = contracted_tensor.get_all_dangling()
        ups_dims = [i for i in range(len(es)) if es[i].up]
        down_dims = [i for i in range(len(es)) if es[i].down]

        up_dim_size = np.prod(np.array(contracted_tensor.shape)[ups_dims])
        down_dim_size = np.prod(np.array(contracted_tensor.shape)[down_dims])

        contracted_tensor = contracted_tensor.reorder_axes(ups_dims + down_dims)
        sym_mat = contracted_tensor.tensor.reshape(up_dim_size, down_dim_size)
        del self._original_tensor
        return sym_mat

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

    def _parse_edges(self) -> None:
        """
        Takes the edges received in the constructor and parses them for each node. i.e. builds the tensors network
        """
        for u_tuple, v_tuple in self._edge_list:
            u_index, u_dim = u_tuple
            v_index, v_dim = v_tuple

            u = self._v[u_index]
            v = self._v[v_index]

            u_copy = self._original_tensor[u_index]
            v_copy = self._original_tensor[v_index]

            try:
                e = u[u_dim] ^ v[v_dim]
                e.up = False
                e.down = False
                e.symmetry = None

                # Saves the original tensor for later compare
                e_copy = u_copy[u_dim] ^ v_copy[v_dim]
                e_copy.up = False
                e_copy.down = False
                e_copy.symmetry = None
            except Exception as e:
                print(f"Problem with {u.name} and {v.name} with {u_dim} -> {v_dim}")
                raise e

    def _get_edges_to_sketch(self) -> List[Node]:
        """
        If self._eges_to_sketch is not None returns all the dangling edges
        In the paper returns nodes with edges in E_OVERLINE
        :return: A list of edges to be sketched
        """
        edges_to_sketch = []
        for node in self._v:
            edges_to_sketch += node.get_all_dangling()
        return edges_to_sketch

    def get_symmetry(self) -> List[Tuple[Edge, Edge]]:
        """
        Returns the symmetries of the tensor
        :return:
        """
        return self._symmetry

    def get_upper_sketch_dimensions(self):
        return [sym[0] for sym in self._symmetry]

    def get_lower_sketch_dimensions(self):
        return [sym[1] for sym in self._symmetry]

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


    def sketch_and_contract_S_up_and_down(self, i: int, j: int, m: int) -> None:
        #TODO: Assumes for start that the up and down edges are symmetrical
        """
        Takes two indices that describe nodes that should be contracted and have a single upper dangling dimension
        and a single lower one.
        We will use the method from the paper but with reflecting it
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

        v_dangling_up = [e for e in v.get_all_dangling() if e.dimension == m and e.up][0]
        u_dangling_up = [e for e in u.get_all_dangling() if e.dimension == m and e.up][0]

        v_dangling_down = [e for e in v.get_all_dangling() if e.dimension == m and e.down][0]
        u_dangling_down = [e for e in u.get_all_dangling() if e.dimension == m and e.down][0]

        split_dim_1, split_dim_2 = self._closest_dividers(m, np.sqrt(m * factor))

        # Reshape v accordingly
        e_v1_up, e_v2_up = tn.split_edge(v_dangling_up, [split_dim_1, split_dim_2])
        e_v1_down, e_v2_down = tn.split_edge(v_dangling_down, [split_dim_1, split_dim_2])


        # Create z_i tree tensor
        v_1_mat = np.random.randn(m, split_dim_1, m) / np.sqrt(m)
        v_2_mat = np.random.randn(m, split_dim_2, m) / np.sqrt(m)

        v_1_up = Node(v_1_mat)
        v_2_up = Node(v_2_mat)

        v_1_down = Node(v_1_mat)
        v_2_down = Node(v_2_mat)

        # Connect the relevant edges
        v_1_up[1] ^ v[e_v1_up.axis1]
        v_2_up[1] ^ v[e_v2_up.axis1]

        v_1_up[0] ^ u[u_dangling_up.axis1]
        v_1_up[2] ^ v_2_up[0]

        v_1_down[1] ^ v[e_v1_down.axis1]
        v_2_down[1] ^ v[e_v2_down.axis1]

        v_1_down[0] ^ u[u_dangling_down.axis1]
        v_1_down[2] ^ v_2_down[0]

        final_e_up = v_2_up.get_all_dangling()[0]
        final_e_down = v_2_down.get_all_dangling()[0]

        final_e_up.up = True
        final_e_up.down = False

        final_e_down.up = False
        final_e_down.down = True

        # contract nodes
        self._count_contraction_cost(v_1_up, u)
        new_node = v_1_up @ u

        self._count_contraction_cost(v_1_down, u)
        new_node = v_1_down @ new_node


        self._count_contraction_cost(new_node, v)
        new_node = new_node @ v

        self._count_contraction_cost(new_node, v_2_up)
        new_node = new_node @ v_2_up

        self._count_contraction_cost(new_node, v_2_down)
        new_node = new_node @ v_2_down

        new_node.set_name(f"z_{u_orig.name}_{v_orig.name}")
        for i in range(len(self._v)):
            if self._v[i] in [u_orig, v_orig, v]:
                self._v[i] = new_node