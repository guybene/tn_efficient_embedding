from typing import List, Tuple

import numpy as np
import pandas as pd

from tensornetwork import Node

from tensor_network.tensor_network import TensorNetwork
from embeddings.efficient_gaussian_embedding import EfficientGaussianEmb


class TTDataTester:

    TT_RANK_DATA_PATH = "results/tt_rank_test/rank_{rank}_results.csv"

    def __init__(self, rank, m_scalar_options, tt_order=6, dim_size=500, embed_eps=0.2, embed_delta=0.9):
        """
        The parameters of the experiments
        :param rank: The rank of the TT blocks
        :param m_scalar_options: The scalar to enter the formula to calculate the sketch size
        :param tt_order: The order to the tensor represented by the TT
        :param dim_size: The dimension size for all dimensions
        :param embed_eps: The eps for the embedding
        :param embed_delta: The delta for the embedding
        """
        self.rank = rank
        self.m_scalar_options = m_scalar_options
        self.tt_order = tt_order
        self.dim_size = dim_size
        self.embed_eps = embed_eps
        self.embed_delta = embed_delta


    def create_rank_test_data(self, rank) -> List[np.ndarray]:
        """
        Creates the tensors that represent the TT
        :param rank: The TT-rank for the TT
        :return: A list of tensors that represent the TT blocks
        """
        tt_data = [np.random.uniform(size=(self.dim_size, rank)).astype(np.float32)]
        for _ in range(self.tt_order - 2):
            tt_data.append(np.random.uniform(size=(rank, self.dim_size, rank)).astype(np.float32))
        tt_data.append(np.random.uniform(size=(rank, self.dim_size)).astype(np.float32))
        return tt_data

    def create_tt_network(self, Nodes: List[np.ndarray]) -> Tuple[TensorNetwork, List]:
        """
        Creates the actual network
        :param Nodes: A list of nodes to connect
        :return: The network of tensors that represent a TT and the TT contraction path
        """
        contraction_path = [(i, i+1) for i in range(self.tt_order - 1)]

        nodes = [Node(d, name=str(i)) for i, d in enumerate(Nodes)]

        edges = [[(0, 1), [1, 0]]]
        for i in range(1, self.tt_order - 1):
            edges.append([(i, 2), (i + 1, 0)])

        return TensorNetwork(v=nodes, edge_list=edges), contraction_path

    def calc_tt_norm(self, tt_tensors: List[np.ndarray]):
        """
        Calculates the original tensor norm
        :param tt_tensors: A list of the TT blocks
        :return: The norm of th tensor represented by the TT decomposition
        """
        D = np.tensordot(tt_tensors[0], tt_tensors[0], axes=([0], [0]))
        tt_tensors[-1] = tt_tensors[-1].reshape(list(tt_tensors[-1].shape) + [1])
        for i in range(1, len(tt_tensors)):
            C = np.tensordot(tt_tensors[i], tt_tensors[i], axes=([1], [1]))
            D = np.tensordot(D, C, axes=[[0,1], [0, 2]])
        tt_tensors[-1] = tt_tensors[-1].reshape(tt_tensors[-1].shape[:-1])
        return np.sqrt(D.flat[0])

    def _embed_and_eval(self, tt_data, algo, network, contraction_path):

        orig_norm = self.calc_tt_norm(tt_data)
        embded_network = algo.embed(network, contraction_path)
        embeded_tensor = embded_network._v[0].tensor
        embed_norm = np.linalg.norm(embeded_tensor)
        return embed_norm/orig_norm

    def run_single_configuration(self, i, res, rank):
        """
        Runs an eperiment
        :param i: experiment index
        :param res: The results to update
        :param rank: The tt rank of the tensors
        :param contraction_path: The contraction path
        :return:
        """
        tt_data = self.create_rank_test_data(rank)
        for m_scalar in self.m_scalar_options:
            for is_TN in [True, False]:
                res["rank"].append(rank)
                res["batch_num"].append(i)
                res["m_factor"].append(m_scalar)
                algo = EfficientGaussianEmb(eps=self.embed_delta, delta=self.embed_eps,
                                            m_scalar=m_scalar, is_tn_embedding=is_TN)

                # First sketching try
                network, contraction_path = self.create_tt_network(tt_data)
                chosen_m = algo.calc_m(network)
                res["actual_m"].append(chosen_m)
                tn = "TN" if is_TN else "Tree"
                res["algo"].append(tn)
                sketch_score = self._embed_and_eval(tt_data, algo, network, contraction_path)
                res["cost"].append(network.contractions_cost)
                res["sketch_score_1"].append(sketch_score)

                # Second Sketching Try
                network, contraction_path = self.create_tt_network(tt_data)
                sketch_score = self._embed_and_eval(tt_data, algo, network, contraction_path)
                res["sketch_score_2"].append(sketch_score)

    def run_rank_test_and_save(self) -> None:
        """
        Runs the rank test experiments and saves the results in a csv.
        Compares the execution for the TN and Tree embedding.
        Runs 25 experiments for each configuration
        """
        res = {"rank": [], "batch_num": [],
              "m_factor":[], "actual_m": [], "sketch_score_1": [],
               "sketch_score_2": [], "cost": [], "algo": []}
        for rank in self.rank:
            print("TT data experiment. Working on Rank:", rank)
            for i in range(25):
                print(f"TT Tensors generated: {i}/25")
                self.run_single_configuration(i, res, rank)

            path_to_save = self.TT_RANK_DATA_PATH.format(rank=rank)
            print(f"Saved TT results in {path_to_save}")
            pd.DataFrame(res).to_csv(path_to_save)  # Note we save a checkpoint for every rank


if __name__ == "__main__":
    ranks = [2 ** i for i in range(2, 7)]
    m_scalar_options = [3, 4, 5, 6, 7, 8]
    tt_data_tester = TTDataTester(rank=ranks, m_scalar_options=m_scalar_options)
    tt_data_tester.run_rank_test_and_save()