import numpy as np
import tensornetwork as tn

from mpo_experiments.ISketch import Sketcher

class PlainHutch(Sketcher):

    def sketch(self, mpos):
        """
        Contracts each MPO to a dense matrix and applies a basic Hutchinson estimator.

        1. Generates a Rademacher vector v (+1 or -1).
        2. Contracts every MPO to a (d^N x d^N) matrix A_i.
           (Rows = Down indices, Columns = Up indices).
        3. Computes the product M = A_{K-1} ... A_1 A_0.
        4. Returns v.T @ M @ v.
        """
        dim = self.d ** self.N
        v = np.random.randint(0, 2, size=dim) * 2 - 1

        # 2. Accumulate Results
        if not hasattr(self, "total_matrix"):
            up_edges = [node[0] for node in mpos[0]]
            down_edges = [node[-1] for node in mpos[-1]]

            # 1. Connect all MPOs vertically (Down of k -> Up of k+1)
            for k in range(len(mpos) - 1):
                for j in range(len(mpos[k])):
                    tn.connect(mpos[k][j][-1], mpos[k + 1][j][0])

            contract_mpop = mpos[0]
            for k in range(1, len(mpos)):
                for j in range(len(mpos[k])):
                    contract_mpop[j] = tn.contract_between(contract_mpop[j], mpos[k][j])

            # 3. Contract all nodes together
            contracted = contract_mpop[0]
            for i in range(1, len(contract_mpop)):
                contracted = tn.contract_between(contracted,
                                                 contract_mpop[i])

            tn.flatten_edges(up_edges)
            tn.flatten_edges(down_edges)

            total_matrix = contracted.tensor
            self.total_matrix = total_matrix
        else:
            total_matrix = self.total_matrix
        return v.T @ total_matrix @ v

    def __str__(self):
        return "Plain Hutch"