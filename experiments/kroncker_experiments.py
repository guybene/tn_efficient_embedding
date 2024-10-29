from typing import List, Tuple

import numpy as np
import pandas as pd

from tensornetwork import Node

from tensor_network.tensor_network import TensorNetwork
from embeddings.efficient_gaussian_embedding import EfficientGaussianEmb


class KronckerDataTester:

    KRONCKER_DATA_PATH = "results/kroncker_order_test/order_{order}_results.csv"

    def __init__(self, order: List[int], m_scalar_options: List[int], katri_rao_sketch_sizes: List[int],
                 dim_size=1000, embed_eps=0.2, embed_delta=0.95):
        """
        Builds the parameters of the experiments
        :param order: The order to tensors to check for
        :param m_scalar_options: The scalar to enter the formula to calculate the sketch size
        :param dim_size: The dimension size for all dimensions
        :param embed_eps: The eps for the embedding
        :param embed_delta: The delta for the embedding
        :param katri_rao_sketch_sizes: The sketch sizes to check for the katri rao embeddings
        """
        self.order = order
        self.m_scalar_options = m_scalar_options
        self.dim_size = dim_size
        self.embed_eps = embed_eps
        self.embed_delta = embed_delta
        self.katri_rao_sketch_sizes = katri_rao_sketch_sizes


    def create_order_test_data(self, order) -> List[np.ndarray]:
        """
        Creates the tensors that represent the TT
        :param order: The order of the data
        :return: A list of tensors that represent the kroncker data
        """
        kroncker_data = [np.random.uniform(size=[self.dim_size] + [1] * (order-1)).astype(np.float32)]
        for i in range(1, order):
            kroncker_data.append(np.random.uniform(size=[1, self.dim_size]).astype(np.float32))
        return kroncker_data

    def create_kroncker_network(self, Nodes: List[np.ndarray], order) -> Tuple[TensorNetwork, List]:
        """
        Creates the actual network
        :param order: The order of the data to create
        :param Nodes: A list of nodes to connect
        :return: The network of tensors that represent a TT and the TT contraction path
        """
        contraction_path = [(i, i+1) for i in range(order -1)]

        nodes = [Node(d, name=str(i)) for i, d in enumerate(Nodes)]

        edges = []
        for i in range(1, order):
            edges.append([(0, i), (i, 0)])

        return TensorNetwork(v=nodes, edge_list=edges), contraction_path

    def calc_kroncker_data_norm(self, krnocker_data: List[np.ndarray]):
        """
        Calculates the original tensor norm
        :param krnocker_data: A list of the vectors tha tare part of a kroncker tensor network
        :return: The norm of the full kroncker product
        """
        norm = np.prod([np.linalg.norm(vec) for vec in krnocker_data])
        return norm

    def _embed_and_eval_katri_rao(self, kroncker_data: List[np.ndarray], sketch_size):
        embeded_tensor = np.random.randn(sketch_size, self.dim_size) @ kroncker_data[0].flat / np.sqrt(sketch_size)
        for data in kroncker_data[1:]:
            s_data = np.random.randn(sketch_size, self.dim_size) @ data.flat / np.sqrt(sketch_size)
            embeded_tensor *= s_data
        orig_norm = self.calc_kroncker_data_norm(kroncker_data)
        embed_norm = np.linalg.norm(embeded_tensor)
        sketch_score = embed_norm / orig_norm

        sketch_cost = len(kroncker_data) * sketch_size * self.dim_size  # The sketching cost
        sketch_cost += self.dim_size * len(kroncker_data)  # Hadmard Product cost
        return sketch_score, sketch_cost


    def _embed_and_eval_TN(self, kroncker_data, algo, network, contraction_path):
        orig_norm = self.calc_kroncker_data_norm(kroncker_data)
        embded_network = algo.embed(network, contraction_path)
        embeded_tensor = embded_network._v[0].tensor
        embed_norm = np.linalg.norm(embeded_tensor)
        return embed_norm/orig_norm

    def run_tree_and_TN_on_kroncker(self, i, res, order, kroncker_data):
        """
        Runs an eperiment
        :param i: experiment index
        :param res: The results to update
        :param order: The order of the tensors
        :param contraction_path: The contraction path
        :param kroncker_data: The data to run on
        """
        for m_scalar in self.m_scalar_options:
            for is_TN in [True, False]:
                res["order"].append(order)
                res["batch_num"].append(i)
                res["m_factor"].append(m_scalar)
                algo = EfficientGaussianEmb(eps=self.embed_delta, delta=self.embed_eps,
                                            m_scalar=m_scalar, is_tn_embedding=is_TN)

                # First sketching try
                network, contraction_path = self.create_kroncker_network(kroncker_data, order)
                chosen_m = algo.calc_m(network)
                res["actual_m"].append(chosen_m)
                tn = "TN" if is_TN else "Tree"
                res["algo"].append(tn)
                sketch_score = self._embed_and_eval_TN(kroncker_data, algo, network, contraction_path)
                res["cost"].append(network.contractions_cost)
                res["sketch_score_1"].append(sketch_score)

                # Second Sketching Try
                network, contraction_path = self.create_kroncker_network(kroncker_data, order)
                sketch_score = self._embed_and_eval_TN(kroncker_data, algo, network, contraction_path)
                res["sketch_score_2"].append(sketch_score)

    def run_katri_rao_on_kroncker(self, i, res, kroncker_data):
        order = kroncker_data[0].shape[0]
        sketches_to_check = [m for m in self.katri_rao_sketch_sizes if m < kroncker_data[0].shape[0]]
        for sketch_size in sketches_to_check:
            res["order"].append(order)
            res["batch_num"].append(i)
            res["m_factor"].append(1)

            res["actual_m"].append(sketch_size)

            res["algo"].append("katri_rao")

            # First sketching try
            sketch_score, contractions_cost = self._embed_and_eval_katri_rao(kroncker_data, sketch_size)

            res["cost"].append(contractions_cost)
            res["sketch_score_1"].append(sketch_score)

            # Second Sketching Try
            sketch_score, contractions_cost = self._embed_and_eval_katri_rao(kroncker_data, sketch_size)

            res["sketch_score_2"].append(sketch_score)


    def run_kroncker_order_test_and_save(self) -> None:
        """
        Runs the order test experiments and saves the results in a csv.
        Compares the execution for the TN and Tree embedding & katri-rao embeddings
        Runs 25 experiments for each configuration
        """
        res = {"order": [], "batch_num": [],
              "m_factor":[], "actual_m": [], "sketch_score_1": [],
               "sketch_score_2": [], "cost": [], "algo": []}
        for order in self.order:
            print("Working on order:", order)
            for i in range(25):
                print(f"{i}/25")
                kroncker_data = self.create_order_test_data(order)
                print("Working on katri-rao")
                self.run_katri_rao_on_kroncker(i, res, kroncker_data)
                print("Working on TN")
                self.run_tree_and_TN_on_kroncker(i, res, order, kroncker_data)

            pd.DataFrame(res).to_csv(self.KRONCKER_DATA_PATH.format(order=order))  # Note we save a checkpoint for every order


if __name__ == "__main__":
    order = [4, 7, 10, 15, 18]
    m_scalar_options = [4, 5, 6, 7]
    katri_rao_sketch_sizes = [50, 100, 500, 10 ** 3, 2000, 10 ** 4]
    tt_data_tester = KronckerDataTester(order=order, m_scalar_options=m_scalar_options,
                                        katri_rao_sketch_sizes=katri_rao_sketch_sizes)
    tt_data_tester.run_kroncker_order_test_and_save()