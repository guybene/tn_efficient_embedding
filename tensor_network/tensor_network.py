from typing import List, Optional, Tuple, Dict
import numpy as np
from tensornetwork import Node
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
        self._edge_list = edge_list
        self._e = self._parse_edges()
        self._nodes_to_sketch = self._get_nodes_with_dimensions_to_be_sketched()

    def sketch(self, i: int, m: int) -> None:
        """
        Sketches node i
        :param i: The node to sketch
        :param m: The dimension to sketch v[i]'s dangling dimensions
        """
        node = self[i]
        sketched_node = node

        dims_to_sketch = [edge.axis1 for edge in node.get_all_dangling()]

        for dim in dims_to_sketch:
            sketching_node = Node(np.random.randn(m, node.shape[dim]) / np.sqrt(m))
            #TODO: Add flop counter here for sketching
            sketched_node = tn.contract(sketched_node[dim] ^ sketching_node[1], name=f"sketched_{node.name}")

        for i in range(len(self._v)):
            if self._v[i] == node:
                self._v[i] = sketched_node



    def contract(self, i: int, j: int) -> None:
        """
        Contracts the node in index i with index j
        :param i: The index of the node to contract
        :param j: The index of the node to contract
        """
        u = self._v[i]
        v = self._v[j]
        uv = u @ v
        # TODO: Add flop counter here for contracting
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

            edges.append(u[u_dim] ^ v[v_dim])
        return edges

    def _get_nodes_with_dimensions_to_be_sketched(self) -> List[Node]:
        """
        Returns the nodes with uncontracted dimensions in the graph.
        The node appears the amount of dimension to sketch he has
        In the paper returns nodes with edges in E_OVERLINE
        :return: A list of nodes with dimensions to be sketched
        """
        nodes_with_external_dimensions = []
        for i, node in enumerate(self._v):
            nodes_with_external_dimensions += [node] * len(node.get_all_dangling())
        return nodes_with_external_dimensions

    def get_nodes_to_sketch(self) -> List[Node]:
        """
        Returns the nodes of the graph with uncontracted dimensions.
        These nodes should be sketched
        :return: The nodes with dimensions to be sketched
        """
        return self._nodes_to_sketch

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
