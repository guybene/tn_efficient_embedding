from typing import List, Dict, Type
import numpy as np
import string
import opt_einsum as oe
from tensornetwork import Node
import matplotlib.pyplot as plt
import matplotlib
import tensornetwork as tn
from scipy.sparse.linalg import eigsh
from time import time
from datetime import datetime

# Adjust these imports to match your actual project structure
from mpo_experiments.ISketch import Sketcher
from mpo_experiments.kron_hutch import KronHutch
from mpo_experiments.mpo_hutch import MpoHutch
from mpo_experiments.plain_hutch import PlainHutch
from mpo_experiments.tensor_sketcher import TensorSketcher

matplotlib.use('Qt5Agg')
plt.ion()


class ExperimentManager:

    def __init__(self, algos: List[Type[Sketcher]], d: int, D: int,
                 m_list: List[int], N_list: List[int], K_list: List[int],
                 is_mpop: bool, M: int):
        """
        :param algos: List of algorithm classes
        :param d: Physical dimension
        :param D: Bond dimension
        :param m_list: List of sketching dimensions to test
        :param N_list: List of MPO lengths to test
        :param K_list: List of K values (MPOP power) to test
        :param is_mpop: Boolean flag for MPOP vs Symmetric trace
        :param M: Max iterations
        """
        self.d = d
        self.D = D
        self.is_mpop = is_mpop
        self.M = M
        self.algo_classes = algos

        # Base values (defaults when holding a parameter fixed)
        # We assume the first element of each list is the "default"
        base_m = m_list[0]
        base_N = N_list[0]
        base_K = K_list[0]

        self.definitions = [
            {'vary': 'm', 'values': m_list, 'fixed': {'N': base_N, 'K': base_K}},
            {'vary': 'N', 'values': N_list, 'fixed': {'m': base_m, 'K': base_K}},
            {'vary': 'K', 'values': K_list, 'fixed': {'m': base_m, 'N': base_N}}
        ]

        self.results = []

    def actual_trace(self, data, N, K):
        """
        Returns the exact trace of the data.
        :return:
        """

        mpo_list = []
        for mpo in data:
            mpo_list.append([Node(n.tensor) for n in mpo])
        for k in range(K):
            for j in range(N - 1):
                # Connect Right leg (-2) of site j to Left leg (1) of site j+1
                tn.connect(mpo_list[k][j][-2], mpo_list[k][j + 1][1])

        # 3. Connect Vertical Edges (Physical Dimensions) & Trace
        for j in range(N):
            col = [mpo_list[k][j] for k in range(K)]

            # A. Connect Product Chain: MPO_k (In) -> MPO_k+1 (Out)
            for k in range(K - 1):
                tn.connect(col[k][-1], col[k + 1][0])

            # B. Connect Trace: MPO_0 (Out) -> MPO_K-1 (In)
            tn.connect(col[0][0], col[-1][-1])
        all_nodes = []
        for mpo in mpo_list:
            for node in mpo:
                all_nodes.append(node)
        trace = tn.contractors.auto(all_nodes).tensor
        return trace
    

    @staticmethod
    def get_eigenvalues_for_mpo_data(mpo_tensors):
        """
        Assumes the tensors are shaped (up,bond,down), (up, bond, bond, down) and are not yet connected
        :param mpo_tensors: A list of tensors that represnt an mpo
        :return: The largest eigenvalue of the matricized mpo
        """
        d = mpo_tensors[0].shape[0]
        up_dims = [tensor[0] for tensor in mpo_tensors]
        down_dims = [tensor[-1] for tensor in mpo_tensors]
        for i in range(len(mpo_tensors) - 1):
            mpo_tensors[i][-2] ^ mpo_tensors[i + 1][1]
        contracted_tensor = tn.contractors.auto(mpo_tensors, output_edge_order=up_dims + down_dims)
        tensor_mat = contracted_tensor.tensor.reshape(d ** len(mpo_tensors), d ** len(mpo_tensors))
        eigen_vals = eigsh(tensor_mat, which='LM', return_eigenvectors=False)
        return eigen_vals

    def generate_data(self, N, K) -> List[List[Node]]:
        """
        Generates K symmetrical MPOs with the specified parameters.

        Args:
            N (int): Length of the MPO.
            K (int): Number of MPOs to generate.

        Returns:
            list: A list of K items, where each item is a list of [Node, Node, ...].
        """
        np.random.seed(10)
        names = string.ascii_uppercase
        results = []
        size = (self.d, self.D, self.D)
        largest_eigen_val = None
        for _ in range(K):
            # 1. Create random tensors A
            A = [np.random.randn(*size) for _ in range(N)]

            # 2. Convert to symmetric MPO tensors W
            W = []
            for Ai in A:
                # Wi shape: (d, Dl, Dr, d)
                Wi = np.einsum('sab,tab->sabt', Ai, Ai, optimize=True)
                W.append(Wi)

            # 3. Apply slicing to endpoints
            W[0] = W[0][:, 0, :, :]
            W[-1] = W[-1][:, 0, :, :]

            nodes = [Node(W[i], name=names[i % len(names)]) for i in range(len(W))]

            # 5. Normalize
            if largest_eigen_val is None or not self.is_mpop:
                largest_eigen_val = ExperimentManager.get_eigenvalues_for_mpo_data(nodes)[-1]
            normalize_factor = largest_eigen_val ** (1 / N)
            normalized_nodes = [node / normalize_factor for node in nodes]

            results.append(normalized_nodes)

        # Make the actual tensor symmetric again by taking A B C -> A B C B A
        if not self.is_mpop:
            symmetrical_results = [[node.copy() for node in mpo] for mpo in results]
            for i in range(1, K):
                symmetrical_results.append([node.copy() for node in results[-i - 1]])
            results = symmetrical_results
        else:
            actual_results = []
            for i in range(K):
                actual_results.append([node.copy() for node in results[0]])
            results = actual_results
        return results

    @staticmethod
    def calc_good_path_for_tensor_network(nodes, optimize_type="greedy"):
        """
        Calculates which path the contractor is taking
        :param nodes: A list of connected nodes
        :return: The path and its info
        """
        edge_map = {}
        current_edge_id = 0
        einsum_args = []
        for node in nodes:
            # Add the tensor
            einsum_args.append(node.tensor)

            # Create/Get integer IDs for edges
            node_indices = []
            for edge in node.edges:
                if edge not in edge_map:
                    edge_map[edge] = current_edge_id
                    current_edge_id += 1
                node_indices.append(edge_map[edge])

            # Add the indices
            einsum_args.append(node_indices)

        path, path_info = oe.contract_path(*einsum_args, [], optimize=optimize_type)
        return path, path_info

    def connect_mpo_from_data(self, data, N, K):
        """
        Connects MPO nodes for sketching.
        Must operate on copies to preserve original data for next iteration.
        """
        copied_data = [[Node(n.tensor) for n in mpo] for mpo in data]

        effective_K = len(copied_data)

        for k in range(effective_K):
            for j in range(N - 1):
                tn.connect(copied_data[k][j][-2], copied_data[k][j + 1][1])
        return copied_data

    def draw_graphs(self):
        nrows = len(self.definitions)
        # Determine ncols by finding the maximum number of values in any variation list
        ncols = max(len(d['values']) for d in self.definitions)

        # Create subplots: One row per definition, One column per value tested
        # Width adjusts dynamically based on how many columns we have
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        fig_iter, axes_iter = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        title = f"MPO Sketching Results ({timestamp}),\n d:{self.d}, D:{self.D}, M:{self.M}, MPOP:{self.is_mpop}"
        fig.suptitle(title, fontsize=14)
        fig_iter.suptitle(title, fontsize=14)

        for row_idx, definition in enumerate(self.definitions):
            vary_name = definition['vary']
            fixed_params = definition['fixed']
            values = definition['values']
            row_data = self.results[row_idx]

            # Build label for the row (fixed parameters)
            fixed_str = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])
            row_label_text = f"Varying {vary_name}\n[{fixed_str}]"

            for col_idx in range(ncols):
                ax = axes[row_idx][col_idx]
                ax_iter = axes_iter[row_idx][col_idx]

                # Check if this row has a value for this column (handles lists of different lengths)
                if col_idx < len(values):
                    val = values[col_idx]

                    # Retrieve data for this specific value (e.g., m=5)
                    val_data = row_data[val]

                    # Plot each algorithm for this specific parameter setting
                    for algo_name, res in val_data.items():
                        start = 100
                        step = 1
                        if len(res['error']) < start: start = 0  # Safety fallback

                        errors = res['error'][start::step]
                        time = res["time"][start::step]

                        iterations = np.arange(len(errors)) * step + start

                        ax.plot(time, errors, label=algo_name)
                        ax_iter.plot(iterations, errors, label=algo_name)


                    # Subplot Formatting
                    ax.set_title(f"{vary_name} = {val}")
                    ax.grid(True, which="both", ls="-", alpha=0.5)
                    ax.legend(fontsize='x-small')
                    ax_iter.set_title(f"{vary_name} = {val}")
                    ax_iter.grid(True, which="both", ls="-", alpha=0.5)
                    ax_iter.legend(fontsize='x-small')

                    # Labels: Only add Y-label to the first column, X-label to the last row
                    if col_idx == 0:
                        ax.set_ylabel(f"{row_label_text}\nError")
                        ax_iter.set_ylabel(f"{row_label_text}\nError")
                    if row_idx == nrows - 1:
                        ax.set_xlabel("Time")
                        ax_iter.set_xlabel("Iterations")


                else:
                    # If the list is shorter than ncols, hide the empty axis
                    ax.axis('off')
                    ax_iter.axis('off')


        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_iter.tight_layout(rect=[0, 0.03, 1, 0.95])

        file_name_time = f"experiment_scenarios_{timestamp}_time.png"
        file_name_iterations = f"experiment_scenarios_{timestamp}_iterations.png"

        save_path_time = f"../mpo_experiments/experiment_results/{file_name_time}"
        save_path_iterations = f"../mpo_experiments/experiment_results/{file_name_iterations}"

        fig.savefig(save_path_time)
        fig_iter.savefig(save_path_iterations)

    def run(self, timer_cap=None) -> None:
        """
        Runs the experiment scenarios defined in __init__.
        Skips execution if the exact (m, N, K) configuration has already been run.
        """
        # Dictionary to cache results: Key=(m, N, K), Value=val_results
        experiment_cache = {}

        for definition in self.definitions:
            vary_name = definition['vary']
            values = definition['values']
            fixed = definition['fixed']

            print(f"Scenario: Change {vary_name}, Fix {fixed}")

            # Storage for this row: { value: { algo_name: result_dict } }
            row_results = {}

            for val in values:
                # 1. Set current parameters
                current_params = fixed.copy()
                current_params[vary_name] = val

                m = current_params['m']
                N = current_params['N']
                K = current_params['K']

                # Create a unique key for this configuration
                config_key = (m, N, K)

                # 2. Check if we already ran this configuration
                if config_key in experiment_cache:
                    print(f"  Skipping Run (Cached): m={m}, N={N}, K={K}")
                    # Retrieve cached results
                    val_results = experiment_cache[config_key]

                else:
                    # If not in cache, run the experiment
                    print(f"  Run: m={m}, N={N}, K={K}")

                    # Generate Data
                    data = self.generate_data(N=N, K=K)
                    effective_K_trace = len(data)

                    # Calculate True Trace
                    trace = self.actual_trace(data, N, effective_K_trace)

                    # Initialize Algorithms
                    current_algos = []
                    for algo_cls in self.algo_classes:
                        algo_k_param = effective_K_trace
                        current_algos.append(algo_cls(m, self.d, self.D, N, algo_k_param))

                    # Run Execution Loop
                    val_results = {}

                    if timer_cap is None:
                        stop_criteria = lambda it, t: it <= self.M
                    else:
                        stop_criteria = lambda it, t: t <= timer_cap

                    for algo in current_algos:
                        algo_name = str(algo)
                        val_results[algo_name] = {'error': [], 'time': []}

                        algo_accumulate = 0
                        algo_time_counter = 0
                        i = 1

                        print(f"    Running {algo_name}")

                        while stop_criteria(i, algo_time_counter):
                            if i % 1000 == 0:
                                print(i)
                            # Re-connect fresh copies for sketching
                            curr_mpos = self.connect_mpo_from_data(data, N, effective_K_trace)

                            start = time()
                            algo_accumulate += algo.sketch(curr_mpos)
                            algo_time_counter += time() - start

                            # Relative Error
                            error = (trace - algo_accumulate / i) / trace

                            val_results[algo_name]['error'].append(error)
                            val_results[algo_name]['time'].append(algo_time_counter)
                            i += 1

                    # 3. Store new results in cache
                    experiment_cache[config_key] = val_results

                # Assign results (either cached or new) to the current row
                row_results[val] = val_results

            # Save all results for this row definition
            self.results.append(row_results)

        # Finally, draw
        self.draw_graphs()


def main():
    d = 5
    D = 4
    is_mpop = True
    M = 5000

    # Define lists for parameters
    # Row 1 will use defaults for N, K (N=5, K=1) and vary m
    m_list = [2]
    N_list = [3,4]
    K_list = [2,3]

    algos = [KronHutch]
    timer_cap = None

    experiment = ExperimentManager(algos=algos, d=d, D=D,
                                   m_list=m_list, N_list=N_list, K_list=K_list,
                                   is_mpop=is_mpop, M=M)
    experiment.run(timer_cap=timer_cap)


if __name__ == "__main__":
    main()