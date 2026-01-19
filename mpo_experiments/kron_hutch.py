import numpy as np
import tensornetwork as tn
from tensornetwork import Node

from mpo_experiments.ISketch import Sketcher

class KronHutch(Sketcher):

    def sketch(self, mpos):
        sketch_matrices = [np.random.randn(self.d, self.m)/np.sqrt(self.m) for _ in range(self.N)]

        up_nodes = [Node(data) for data in sketch_matrices]
        down_nodes = [Node(data) for data in sketch_matrices]

        up_edges = [node[-1] for node in up_nodes]
        down_edges = [node[-1] for node in down_nodes]




        for k in range(len(mpos) - 1):
            for j in range(len(mpos[k])):
                tn.connect(mpos[k][j][-1], mpos[k + 1][j][0])

        lower_layer = mpos[0]
        upper_layer = mpos[-1]

        for i in range(self.N):
            lower_layer[i][0] ^ down_nodes[i][0]
            upper_layer[i][-1] ^ up_nodes[i][0]

            lower_layer[i] = tn.contract_between(lower_layer[i], down_nodes[i])
            upper_layer[i] = tn.contract_between(upper_layer[i], up_nodes[i])

        contract_mpop = mpos[0]
        for k in range(1, len(mpos)):
            for j in range(len(mpos[k])):
                contract_mpop[j] = tn.contract_between(contract_mpop[j], mpos[k][j])

        contracted = contract_mpop[0]
        for i in range(1, len(contract_mpop)):
            contracted = tn.contract_between(contracted,
                                             contract_mpop[i])

        tn.flatten_edges(up_edges)
        tn.flatten_edges(down_edges)

        total_matrix = contracted.tensor
        return np.trace(total_matrix)


    def __str__(self):
        return "Kron Hutch"