from math import ceil
from typing import List, Tuple

import numpy as np
import pandas as pd

from tensornetwork import Node

from tensor_network.tensor_network import TensorNetwork
from embeddings.efficient_gaussian_embedding import EfficientGaussianEmb


class KroneckerDataTester:

    KRONECKER_DATA_PATH = "results/kronecker_order_test/order_{order}_results.csv"

    def __init__(self, order: List[int], m_scalar_options: List[int], kronecker_sketch_sizes: List[int],
                 kronecker_factor: List[int], dim_size=1000, embed_eps=0.2, embed_delta=0.95):
        """
        Builds the parameters of the experiments
        :param order: The order to tensors to check for
        :param m_scalar_options: The scalar to enter the formula to calculate the sketch size
        :param dim_size: The dimension size for all dimensions
        :param embed_eps: The eps for the embedding
        :param embed_delta: The delta for the embedding
        :param kronecker_sketch_sizes: The sketch sizes to check for the katri rao embeddings
        :param kronecker_factor: We sub sketch with the kronecker structure with s_dim = m ** 1/n * kronecker_factors
        """
        self.order = order
        self.m_scalar_options = m_scalar_options
        self.dim_size = dim_size
        self.embed_eps = embed_eps
        self.embed_delta = embed_delta
        self.kronecker_sketch_sizes = kronecker_sketch_sizes
        self.kronecker_factor = kronecker_factor


    def create_order_test_data(self, order) -> List[np.ndarray]:
        """
        Creates the tensors that represent the kronecker structured data
        :param order: The order of the data
        :return: A list of tensors that represent the kronecker data
        """
        kronecker_data = [np.random.uniform(size=[self.dim_size] + [1] * (order-1)).astype(np.float32)]
        for i in range(1, order):
            kronecker_data.append(np.random.uniform(size=[1, self.dim_size]).astype(np.float32))
        return kronecker_data

    def create_kronecker_network(self, Nodes: List[np.ndarray], order) -> Tuple[TensorNetwork, List]:
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

    def calc_kronecker_data_norm(self, krnocker_data: List[np.ndarray]):
        """
        Calculates the original tensor norm
        :param krnocker_data: A list of the vectors tha tare part of a kronecker tensor network
        :return: The norm of the full kronecker product
        """
        norm = np.prod([np.linalg.norm(vec) for vec in krnocker_data])
        return norm

    def _embed_and_eval_kronecker(self, kronecker_data: List[np.ndarray], m, s_factor):
        """
        Embeds with the kronecker structered embedding and evals the sketch error
        :param kronecker_data: A list of tensors that represent a kronecker structured data
        :param m: The sketch size to calcualte with the actuall sketching amtrix dims
        :param s_factor: The factor to multiply the sketch size
        :return: Sketch_score, sketch_cose and actual sketch_size
        """
        sketch_size = ceil(pow(m, 1 / len(kronecker_data)) * s_factor)
        embeded_tensor = np.random.randn(sketch_size, self.dim_size) @ kronecker_data[0].flat / np.sqrt(sketch_size)
        for data in kronecker_data[1:]:
            s_data = np.random.randn(sketch_size, self.dim_size) @ data.flat / np.sqrt(sketch_size)
            embeded_tensor = np.kron(embeded_tensor.astype(np.float32), s_data.astype(np.float32))
        embed_norm = np.linalg.norm(embeded_tensor)
        del embeded_tensor
        orig_norm = self.calc_kronecker_data_norm(kronecker_data)
        sketch_score = embed_norm / orig_norm

        sketch_cost = len(kronecker_data) * sketch_size * self.dim_size  # The sketching cost
        sketch_cost += sketch_size ** len(kronecker_data)  # kronecker produdct
        return sketch_score, sketch_cost, sketch_size ** len(kronecker_data)


    def _embed_and_eval_TN(self, kronecker_data: List[np.ndarray], algo: EfficientGaussianEmb,
                           network: TensorNetwork, contraction_path: List):
        """
        Embeds and evals the TN and Tree
        :param kronecker_data: The tensors representing a kronecker data
        :param algo: The algo to embed with
        :param network: THe network to embed
        :param contraction_path: THe contraction path of the tensor network
        :return: The sketch error
        """
        orig_norm = self.calc_kronecker_data_norm(kronecker_data)
        embded_network = algo.embed(network, contraction_path)
        embeded_tensor = embded_network._v[0].tensor
        embed_norm = np.linalg.norm(embeded_tensor)
        return embed_norm/orig_norm

    def run_tree_and_TN_on_kronecker(self, i, res, order, kronecker_data):
        """
        Runs an eperiment
        :param i: experiment index
        :param res: The results to update
        :param order: The order of the tensors
        :param contraction_path: The contraction path
        :param kronecker_data: The data to run on
        """
        for m_scalar in self.m_scalar_options:
            for is_TN in [True, False]:
                res["order"].append(order)
                res["batch_num"].append(i)
                res["m_factor"].append(m_scalar)
                algo = EfficientGaussianEmb(eps=self.embed_delta, delta=self.embed_eps,
                                            m_scalar=m_scalar, is_tn_embedding=is_TN)

                # First sketching try
                network, contraction_path = self.create_kronecker_network(kronecker_data, order)
                chosen_m = algo.calc_m(network)
                res["actual_m"].append(chosen_m)
                tn = "TN" if is_TN else "Tree"
                res["algo"].append(tn)
                sketch_score = self._embed_and_eval_TN(kronecker_data, algo, network, contraction_path)
                res["cost"].append(network.contractions_cost)
                res["sketch_score_1"].append(sketch_score)

                # Second Sketching Try
                network, contraction_path = self.create_kronecker_network(kronecker_data, order)
                sketch_score = self._embed_and_eval_TN(kronecker_data, algo, network, contraction_path)
                res["sketch_score_2"].append(sketch_score)

    def run_katri_rao_on_kronecker(self, i, res, order, kronecker_data):
        """
        Runs the actual metrics on a single network with a specific order config
        :param i: The index of run insied the order batch
        :param res: The results to update
        :param order: THe order of the tensors
        :param kronecker_data: The data representing kronecker data
        """
        for m in self.kronecker_sketch_sizes:
            for s_factor in self.kronecker_factor:
                res["order"].append(order)
                res["batch_num"].append(i)
                res["m_factor"].append(1)
                res["algo"].append("katri_rao")

                # First sketching try
                sketch_score, contractions_cost, sketch_size = self._embed_and_eval_kronecker(kronecker_data,
                                                                                             m,
                                                                                             s_factor)

                res["actual_m"].append(sketch_size)
                res["cost"].append(contractions_cost)
                res["sketch_score_1"].append(sketch_score)

                # Second Sketching Try
                sketch_score, contractions_cost, sketch_size = self._embed_and_eval_kronecker(kronecker_data,
                                                                                             m,
                                                                                             s_factor)

                res["sketch_score_2"].append(sketch_score)


    def run_kronecker_order_test_and_save(self) -> None:
        """
        Runs the order test experiments and saves the results in a csv.
        Compares the execution for the TN and Tree embedding & katri-rao embeddings
        Runs 25 experiments for each configuration
        """
        res = {"order": [], "batch_num": [],
              "m_factor":[], "actual_m": [], "sketch_score_1": [],
               "sketch_score_2": [], "cost": [], "algo": []}
        for order in self.order:
            print("Kronecker Data Experiment. Working on order:", order)
            for i in range(25):
                print(f"Kroncker Tensor generated: {i}/25")
                kronecker_data = self.create_order_test_data(order)
                self.run_katri_rao_on_kronecker(i, res, order, kronecker_data)
                self.run_tree_and_TN_on_kronecker(i, res, order, kronecker_data)

            path_to_save = self.KRONECKER_DATA_PATH.format(order=order)
            print(f"Saved Kronecker results in {path_to_save}")
            pd.DataFrame(res).to_csv(path_to_save)  # Note we save a checkpoint for every order


if __name__ == "__main__":
    order = [4, 6, 8, 10]
    m_scalar_options = [4, 5, 6, 7]
    katri_rao_sketch_sizes = [1000, 3000, 5000, 10 ** 4, 15000, 20000]
    kronecker_data_tester = KroneckerDataTester(order=order, m_scalar_options=m_scalar_options, kronecker_factor=[1, 2],
                                                kronecker_sketch_sizes=katri_rao_sketch_sizes)
    kronecker_data_tester.run_kronecker_order_test_and_save()