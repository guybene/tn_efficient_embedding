from typing import List

import numpy as np
import pandas as pd

from tensornetwork import Node

from tensor_network.tensor_network import TensorNetwork
from embeddings.efficient_gaussian_embedding import EfficientGaussianEmb


class compare_tt_data:

    def __init__(self, rank_power=6, tt_order=6, dim_size=500, embed_eps=0.2, embed_delta=0.8):
        """
        The parameters of the experiments
        :param rank_power: The powers of 2 to get the ranks from (The rank are 4, 8,16...256)
        :param tt_order: The order to the tensor represented by the TT
        :param dim_size: The dimension size for all dimensions
        :param embed_eps: The eps for the embedding
        :param embed_delta: The delta for the embedding
        """
        self.rank_power = rank_power
        self.tt_order = tt_order
        self.dim_size = dim_size
        self.embed_eps = embed_eps
        self.embed_delta = embed_delta

    def create_data(self, rank) -> List[np.ndarray]:
        """
        Creates the tensors that represent the TT
        :param rank: The TT-rank for the TT
        :return: A list of tensors that represent the TT blocks
        """
        tt_data = [np.random.uniform(size=(500, rank)).astype(np.float32)]
        for _ in range(4):
            tt_data.append(np.random.uniform(size=(rank, 500, rank)).astype(np.float32))
        tt_data.append(np.random.uniform(size=(rank, 500)).astype(np.float32))
        return tt_data

    def create_tt_network(self, Nodes: List[np.ndarray], is_tree_embedding) -> TensorNetwork:
        """
        Creates the actual network
        :param Nodes: A list of nodes to connect
        :param is_tree_embedding: If true, will use the Tree embedding and not the TN embedding
        :return: The network of tensors that represent a TT
        """
        nodes = [Node(d) for d in Nodes]
        edges = [[(0, 1), [1, 0]],
                 [(1, 2), [2, 0]],
                 [(2, 2), [3, 0]],
                 [(3, 2), [4, 0]],
                 [(4, 2), [5, 0]]]

        return TensorNetwork(v=nodes, edge_list=edges, is_tree_embedding=is_tree_embedding)

    def calc_tt_norm(self, tt_tensors: List[np.ndarray]):
        """
        Calculates the original tensor norm
        :param tt_tensors: A list of the TT blocks
        :return: The norm of th tensor represented by the TT decomposition
        """
        tensor = tt_tensors[0]
        first_sum = np.sum([np.kron(tensor[i], tensor[i]) for i in range(tensor.shape[0])], axis=0)

        between_values = []
        for tensor in tt_tensors[1:-1]:
            curr_krons = np.zeros(np.kron(tensor[:, 0], tensor[:, 0]).shape)
            for i in range(tensor.shape[1]):
                print(i)
                curr_krons += np.kron(tensor[:, i], tensor[:, i])
            between_values.append(curr_krons)
        mid_sum = between_values[0]
        #TODO: Continue here
        tensor = tt_tensors[-1]
        last_sum = np.sum([np.kron(tensor[:, i], tensor[:, i]) for i in range(tensor.shape[0])], axis=0)

        return 0

    def run(self):
        contraction_path = [[[0, 1], [1, 0]],
                            [[1, 2], [2, 0]],
                            [[2, 2], [3, 0]],
                            [[3, 2], [4, 0]],
                            [[4, 2], [5, 0]]]
        for power_of_2 in range(2, self.rank_power + 1):
            print("Working on Rank:", 2 ** power_of_2)
            rank = 2 ** power_of_2
            algo = EfficientGaussianEmb(eps=0.2, delta=0.8, m_scalar=1)
            for is_TN in [True]:
                rank = 2 ** 6
                tt_data = self.create_data(rank)
                network = self.create_tt_network(tt_data, not is_TN)
                orig_norm = self.calc_tt_norm(tt_data)
                embed_tensor = algo.embed(network, contraction_path)
                print(2)


if __name__ == "__main__":
    compare_tt_data().run()
