import numpy as np
from tensornetwork import Node
import tensornetwork as tn

from mpo_experiments.ISketch import Sketcher


class MpoHutch(Sketcher):

    def sketch(self, mpos):
        d = self.d
        N = self.N
        K = self.K

        bounding_sketch_data = [np.random.randn(self.d, self.m) / np.sqrt(self.m) for _ in range(N)]
        lower_sketch_data = [data.copy() for data in bounding_sketch_data]

        curr_mat = np.eye(self.m ** self.N)
        for i, mpo in enumerate(mpos):
            if i != K - 1 and K != 1:
                upper_sketch_data = [np.random.randn(self.d, self.m) / np.sqrt(self.m) for _ in range(N)]
            else:
                upper_sketch_data = bounding_sketch_data
            sketch_results = []
            upper_edges = []
            lower_edges = []
            for node_i, node in enumerate(mpo):
                upper_sketching_node = Node(upper_sketch_data[node_i])
                lower_sketching_node = Node(lower_sketch_data[node_i])

                upper_edges.append(upper_sketching_node[1])
                lower_edges.append(lower_sketching_node[1])

                node[0] ^ upper_sketching_node[0]
                node[-1] ^ lower_sketching_node[0]
                sketched_node = tn.contractors.auto(nodes=[upper_sketching_node,
                                                           lower_sketching_node,
                                                           node],
                                                    ignore_edge_order=True)
                sketch_results.append(sketched_node)

            mpo_sketched_node = sketch_results[0]
            for i in range(1,len(sketch_results)):
                mpo_sketched_node = tn.contract_between(mpo_sketched_node,
                                                        sketch_results[i])
            mpo_sketched_mat = mpo_sketched_node.tensor.reshape(self.m ** self.N, self.m ** self.N)
            curr_mat @= mpo_sketched_mat
            lower_sketch_data = upper_sketch_data

        return np.trace(curr_mat)

    def __str__(self):
        return "MPO Hutch"