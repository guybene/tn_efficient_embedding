from typing import List, Tuple, Dict
import numpy as np
from itertools import permutations

from tensornetwork import Node, Edge

from tensor_network.sym_tensor_network import SymmetricalTensorNetwork


class SymmetricMpoEmbedding:
    #TODO: The code will only assume for now an MPO format tensor.
    #TODO: The code assumes all contraction are in S_1 and S_2


    def __init__(self, eps: float, delta: float, m_scalar: float, is_tn_embedding=True):
        """
        Creates the Efficient Gaussian Embedding object.
        The embedding is eps-delta accurate
        :param eps: The change in norm allowed for the sketching in a normalized vector
        :param delta: The probability for a higher change than epsilon
        :param m_scalar: The scalar to multiply m by in order to actually get good results.
        m=theta(N_E*log(1/delta)/eps^2) so we have m up to this scalar
        :param is_tn_embedding: Should we use TN or tree embedding
        """
        self.eps = eps
        self.delta = delta
        self.m_scalar = m_scalar
        self.is_tn_embedding = is_tn_embedding

    def calc_m(self, x: SymmetricalTensorNetwork):
        """
        We calculate the dimension size to sketch with
        :return: The dimensions size to sketch to
        """
        m_theta = len(x.get_edges_to_sketch()) * np.log(1 / self.delta) / self.eps ** 2
        m = int(m_theta * self.m_scalar)
        print(f"m calcaulted is: {m}")
        return max(m, 1)

    def embed_mpo(self, x: SymmetricalTensorNetwork, contraction_path: List[Tuple[int, int]],
                  m_override: int =None) -> SymmetricalTensorNetwork:
        """
        Embeds the tensor network X, assumes that it is in MPO form, or PEPS. Shouldnt have any contractions not in S
        :param x: The MPO/PEPS tensor network to embed
        :param contraction_path: The contraction path of x, T_0
        :
        :return: An embedded tensor network
        """
        m = self.calc_m(x) if not m_override else m_override

        D, S_up, S_down, I_S = self._partition_contractions(x, contraction_path)
        self._contract_and_sketch_kronecker_product(x, D, m)
        self._contract_and_sketch_tree_embedding(S_up, S_down, I_S, x, m_override)
        return x

    def _partition_contractions(self, x: SymmetricalTensorNetwork, contraction_path: List[Tuple[int, int]]):
        """
        Partitions the contractions for the sketching algorithm
        :param x: The tensor network to sketch
        :param contraction_path: The contraction path for the tensor network
        :return:  D - Dict of the format <i : List[Contractions]>. Where each tuple represents D(e_i)
                  S_up - A list of contractions with both nodes having upwards dimensions to be sketched
                  S_down -  A list of contractions with both nodes having downward dimensions to be sketched
                  I_S - A list of contractions with no nodes to be sketched union with S
        """
        dimensions_to_sketch = x.get_edges_to_sketch()
        D = {e: [] for e in dimensions_to_sketch}

        up_edges = set(x.get_upper_sketch_dimensions())
        down_edges = set(x.get_lower_sketch_dimensions())

        S_up = []
        S_down = []
        I_S = []

        for contraction in contraction_path:
            u_i = x[contraction[0]]
            v_i = x[contraction[1]]

            u_i_dangling_up = [e for e in u_i.get_all_dangling() if e in up_edges]
            v_i_dangling_up = [e for e in v_i.get_all_dangling() if e in up_edges]

            u_i_dims_to_sketch_up = len(u_i_dangling_up)
            v_i_dims_to_sketch_up = len(v_i_dangling_up)

            u_i_dangling_down = [e for e in u_i.get_all_dangling() if e in down_edges]
            v_i_dangling_down = [e for e in v_i.get_all_dangling() if e in down_edges]

            u_i_dims_to_sketch_down = len(u_i_dangling_down)
            v_i_dims_to_sketch_down = len(v_i_dangling_down)


            if u_i_dims_to_sketch_up == v_i_dims_to_sketch_up == 1:
                S_up.append(contraction)

            if u_i_dims_to_sketch_down == v_i_dims_to_sketch_down == 1:
                S_down.append(contraction)
            I_S.append(contraction)
        return  D, S_up, S_down, I_S

    def _contract_and_sketch_kronecker_product(self, x: SymmetricalTensorNetwork, D: Dict[Edge, List[Tuple[int, int]]],
                                               m: int) -> None:
        """
        Sketches the kroncker product part of the algorithm and contracts when necessary
        :param x: The tensor network to contract
        :param D: A dict where each tuple represents D(e_i)
        :param m: sketch dimension size
        """
        for e, D_e_i in D.items():
            x.kronecker_sketch(edge=e, m=m)
        x.delete_symmetry_relevant_caches()



    def _calculate_contraction_path_shapes(self, x: SymmetricalTensorNetwork, D_e_i: List[Tuple[int, int]], i: int):
        """
        Calculates for a list of contraction of the form D_e_i when to add the sketching so we will have a minimal
        sketching cost. i.e. how to contract the sub tensor network such that node_e_i has minimal size to sketch
        :param x: The tensor network
        :param D_e_i: A list of contraction where each tuple if of the form {i, j} where i is the node to minimize
        :param i: The node index to minimize the computations on
        :return: A reordering of the contractions list with an added (i, None) for the sketching adding
        """
        u = x[i]

        j_nodes = [u if v == i else v for u, v in D_e_i]

        contractions_permutations = list(permutations(j_nodes))
        u_possible_sizes = []
        u_possible_sketching_index = []
        for permutation in contractions_permutations:
            contraction_nodes = [x[i] for i in permutation]
            min_u_size, min_u_size_index = self._contraction_path_minimize_U(u, contraction_nodes)
            u_possible_sizes.append(min_u_size)
            u_possible_sketching_index.append(min_u_size_index)

        chosen_path_index = np.argmin(u_possible_sizes)
        sketch_index = u_possible_sketching_index[chosen_path_index]
        chosen_path = list(contractions_permutations[chosen_path_index])
        chosen_path.insert(sketch_index + 1, None)
        return list(zip([i] * len(chosen_path), chosen_path))

    @staticmethod
    def _contraction_path_minimize_U(U: Node, contraction_order: List[Node]):
        """
        Calculates with a given contraction path, what is the minimal size U gets here and where
        :param U: The node to minimize
        :param contraction_order: List of nodes to contract with U in the specified order
        :return: Minimum size of U reached during contractions and the index where this occurs
        """
        min_U_size = U.tensor.size
        min_index = -1
        current_shape = U.tensor.shape

        U_edges = {edge: current_shape[i] for i, edge in enumerate(U.edges)}

        for i, V in enumerate(contraction_order):
            contracted_indices = []
            for edge in U.edges:
                if edge in V.edges:
                    contracted_indices.append(U_edges[edge])

            new_shape = [dim for dim in current_shape if dim not in contracted_indices]
            new_shape += [dim for dim in V.tensor.shape if dim not in contracted_indices]


            new_U_size = np.prod(new_shape)
            current_shape = new_shape
            if new_U_size < min_U_size:
                min_U_size = new_U_size
                min_index = i

        return min_U_size, min_index


    def _contract_and_sketch_tree_embedding(self, S_up, S_down,  I_S, x: SymmetricalTensorNetwork, m) -> None:
        """
        Assumes for now that all contraction are in S and all of them are both in S_up and S_down, like MPO or PEPS
        """
        for i, j in S_up:
            x.sketch_and_contract_S_up_and_down(i, j, m)



if __name__ == "__main__":


    A = Node(np.ones((2,2,5)), name="A")
    B = Node(np.ones((5,4,4,2)), name="B")
    C = Node(np.ones((2,3,3,4)), name="C")
    D = Node(np.ones((4,3,3)), name="D")



    e1 = ((0,2), (1,0))
    e2 = ((1,3), (2,0))
    e3 = ((2,3),(3,0))

    symmetry =   [((0,0), (0,1)), ((1,1), (1,2)), ((2,1),(2,2)), ((3,1),(3,2))]

    contraction_path =  [(0,1), (1,2),(2,3)]

    network = SymmetricalTensorNetwork([A,B,C, D], [e1, e2, e3], symmetry)

    embedding = SymmetricMpoEmbedding(eps=0.1, delta=0.1, m_scalar=0.07)
    mat = network.get_original_symmetrical_matrix()

    result = embedding.embed_mpo(network,contraction_path)

    print(np.trace(mat))
    print(np.trace(result[0].tensor))