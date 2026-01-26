from typing import List, Dict, Type
import numpy as np
import string
import os
import opt_einsum as oe
from tensornetwork import Node
import matplotlib.pyplot as plt
import matplotlib
import tensornetwork as tn
from time import time
from datetime import datetime
import pickle as pkl

from scipy.sparse.linalg import LinearOperator, eigsh

# Adjust these imports to match your actual project structure
from mpo_experiments.ISketch import Sketcher
from mpo_experiments.kron_hutch import KronHutch
from mpo_experiments.mpo_hutch import MpoHutch
from mpo_experiments.plain_hutch import PlainHutch
from mpo_experiments.tensor_sketcher import TensorSketcher

matplotlib.use('Agg')
plt.ion()


class ExperimentManager:

    def __init__(self, algos: List[Type[Sketcher]], d: int, D: int,
                 m_list: List[int], N_list: List[int], K_list: List[int],
                 is_mpop: bool, M: int, min_iteration: int):
        """
        :param algos: List of algorithm classes
        :param d: Physical dimension
        :param D: Bond dimension
        :param m_list: List of sketching dimensions to test
        :param N_list: List of MPO lengths to test
        :param K_list: List of K values (MPOP power) to test
        :param is_mpop: Boolean flag for MPOP vs Symmetric trace
        :param M: Max iterations
        :param min_iteration: The iteration to start plotting with, to skip beginning noise
        """
        self.d = d
        self.D = D
        self.is_mpop = is_mpop
        self.M = M
        self.algo_classes = algos
        self.min_iteration = min_iteration

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
    def get_eigenvalues_for_mpo_data_chat_version(mpo_nodes_data):
        """
        Calculates the largest eigenvalue efficiently using a LinearOperator.
        """
        N = len(mpo_nodes_data)
        d = mpo_nodes_data[0].shape[-1]
        dim_total = d ** N

        def matvec(v):
            v_tensor = v.reshape([d] * N)
            v_node = tn.Node(v_tensor)
            output_edges = [mpo_nodes_data[i][-1] for i in range(N)]
            output_edges2 = [mpo_nodes_data[i][0] for i in range(N)]

            for i in range(N - 1):
                tn.connect(mpo_nodes_data[i][1], mpo_nodes_data[i + 1][-2])

            for i in range(N):
                tn.connect(mpo_nodes_data[i][0], v_node[i])

            result_node = v_node
            for i in range(N):
                result_node = tn.contract_between(mpo_nodes_data[i], result_node)

            # 6. Reorder edges to ensure (Site 0, Site 1, ..., Site N) standard order
            result_node.reorder_edges(output_edges)

            # 7. Flatten back to 1D vector for Scipy
            return result_node.tensor.flatten()

        # Define the Linear Operator
        A = LinearOperator((dim_total, dim_total), matvec=matvec)

        # Run Lanczos
        eigenvalue = eigsh(A, k=1, return_eigenvectors=False)
        return eigenvalue

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
        eigen_vals = eigsh(tensor_mat, k=1, which='LM', return_eigenvectors=False)
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
            A = [np.random.randn(*size).astype(np.float32) for _ in range(N)]

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
                largest_eigen_val = ExperimentManager.get_eigenvalues_for_mpo_data(nodes)[0]
            factor = largest_eigen_val ** (1 / N)
            normalized_nodes = [Node(node.tensor / factor, name=node.name) for node in nodes]
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

    def connect_mpo_from_data(self, data, N):
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

    def draw_graphs(self, save_folder, plot_errors=True, plot_variance=True):
        info_title = f"d:{self.d}, D:{self.D}, M:{self.M}, MPOP:{self.is_mpop}"

        # 2. Iterate over each definition (Vary m, Vary N, Vary K)
        for idx, definition in enumerate(self.definitions):
            vary_name = definition['vary']
            fixed_params = definition['fixed']
            values = sorted(definition['values'])  # Ensure sorted order for plots
            row_data = self.results[idx]

            # Formatted string for fixed parameters
            fixed_str = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])

            # --- PREPARE DATA AGGREGATION ---
            # We need this for both error loops (to plot lines) and variance (to calculate var)
            algo_variance_map = {}  # {algo_name: [var_val1, var_val2...]}

            # Initialize figures only if needed
            fig_time, ax_time = None, None
            fig_iter, ax_iter = None, None
            fig_var, ax_var = None, None

            if plot_errors:
                ncols = len(values)
                # Dynamic width: 5 inches per subplot
                fig_time, ax_time = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False, sharey=True)
                fig_iter, ax_iter = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False, sharey=True)
                fig_time.suptitle(f"Time vs Error: Varying {vary_name} ({fixed_str})\n{info_title}", fontsize=12)
                fig_iter.suptitle(f"Iter vs Error: Varying {vary_name} ({fixed_str})\n{info_title}", fontsize=12)

            if plot_variance:
                fig_var, ax_var = plt.subplots(1, 1, figsize=(8, 6))
                ax_var.set_title(f"Variance Analysis: Varying {vary_name} ({fixed_str})\n{info_title}", fontsize=12)

            # --- MAIN LOOP OVER VALUES (e.g., m=1, m=2...) ---
            for col_idx, val in enumerate(values):
                if val not in row_data:
                    continue

                val_data = row_data[val]

                # Get specific axes for this value if plotting errors
                curr_ax_time = ax_time[0][col_idx] if plot_errors else None
                curr_ax_iter = ax_iter[0][col_idx] if plot_errors else None

                # Iterate Algos
                for algo_name, res in val_data.items():
                    # --- A. ERROR PLOTS ---
                    if plot_errors:
                        start = self.min_iteration
                        step = 1

                        errors = res['error'][start::step]
                        times = res["time"][start::step]
                        iterations = np.arange(len(errors)) * step + start

                        curr_ax_time.plot(times, errors, label=algo_name)
                        curr_ax_iter.plot(iterations, errors, label=algo_name)

                    # --- B. VARIANCE CALCULATION (Always calculate if plotting variance) ---
                    if plot_variance:
                        if algo_name not in algo_variance_map:
                            algo_variance_map[algo_name] = []

                        raw_predictions = np.array(res['prediction'])
                        variance = np.var(raw_predictions)
                        algo_variance_map[algo_name].append(variance)

                # Subplot Formatting (Error Plots)
                if plot_errors:
                    curr_ax_time.set_title(f"{vary_name} = {val}")
                    curr_ax_time.set_xlabel("Time (s)")
                    curr_ax_time.grid(True, alpha=0.5)

                    curr_ax_iter.set_title(f"{vary_name} = {val}")
                    curr_ax_iter.set_xlabel("Iterations")
                    curr_ax_iter.grid(True, alpha=0.5)

                    if col_idx == 0:
                        curr_ax_time.set_ylabel("Error")
                        curr_ax_iter.set_ylabel("Error")
                        curr_ax_time.legend(fontsize='small')
                        curr_ax_iter.legend(fontsize='small')

            # --- SAVE ERROR PLOTS ---
            if plot_errors:
                fig_time.tight_layout(rect=[0, 0.03, 1, 0.90])
                fig_iter.tight_layout(rect=[0, 0.03, 1, 0.90])

                path_time = os.path.join(save_folder, f"{vary_name}_time_error.png")
                path_iter = os.path.join(save_folder, f"{vary_name}_iter_error.png")

                fig_time.savefig(path_time)
                fig_iter.savefig(path_iter)
                plt.close(fig_time)
                plt.close(fig_iter)

            # --- PLOT & SAVE VARIANCE ---
            if plot_variance:
                for algo_name, var_values in algo_variance_map.items():
                    # Slice values just in case data is missing for some points
                    ax_var.plot(values[:len(var_values)], var_values, marker='o', label=algo_name)

                ax_var.set_xlabel(f"{vary_name}")
                ax_var.set_ylabel("Variance")
                ax_var.grid(True, alpha=0.5)
                ax_var.legend()

                # ax_var.set_yscale('log') # Uncomment if you want log scale

                fig_var.tight_layout(rect=[0, 0.03, 1, 0.90])
                path_var = os.path.join(save_folder, f"{vary_name}_variance.png")
                fig_var.savefig(path_var)
                plt.close(fig_var)

    def run(self, backup_file=False, timer_cap=None, plot_errors=True, plot_variance=True) -> None:
        """
        Runs the experiment scenarios defined in __init__.
        Skips execution if the exact (m, N, K) configuration has already been run.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = "../mpo_experiments/experiment_results"
        save_folder = os.path.join(base_dir, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        print(f"Saving results to: {save_folder}")

        pickle_file = os.path.join(save_folder, "experiment_backup")

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

                    trace = self.actual_trace(data, N, K)

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
                        val_results[algo_name] = {'error': [], 'prediction': [], 'time': []}

                        algo_accumulate = 0
                        algo_time_counter = 0
                        i = 1

                        print(f"    Running {algo_name}")

                        while stop_criteria(i, algo_time_counter):
                            if i % 1000 == 0:
                                print(i)
                            # Re-connect fresh copies for sketching
                            curr_mpos = self.connect_mpo_from_data(data, N)

                            start = time()
                            algo_prediction = algo.sketch(curr_mpos)
                            algo_accumulate += algo_prediction
                            algo_time_counter += time() - start

                            # Relative Error
                            error = (trace - algo_accumulate / i) / abs(trace)

                            val_results[algo_name]['prediction'].append(algo_prediction)
                            val_results[algo_name]['error'].append(error)
                            val_results[algo_name]['time'].append(algo_time_counter)

                            i += 1

                    # 3. Store new results in cache
                    experiment_cache[config_key] = val_results

                # Assign results (either cached or new) to the current row
                row_results[val] = val_results
                if backup_file:
                    with open(pickle_file + f"_{vary_name}.pkl", "wb") as f:
                        pkl.dump(row_results, f)

            # Save all results for this row definition
            self.results.append(row_results)

        self.draw_graphs(save_folder,plot_errors, plot_variance)


def main():
    d = 2
    D = 1
    is_mpop = True
    M = 350000

    # Define lists for parameters
    m_list = [1]
    N_list = list(range(3,4))
    K_list = list(range(4,5))

    algos = [TensorSketcher]
    timer_cap = None

    experiment = ExperimentManager(algos=algos, d=d, D=D,
                                   m_list=m_list, N_list=N_list, K_list=K_list,
                                   is_mpop=is_mpop, M=M, min_iteration=1000)
    experiment.run(backup_file=True, timer_cap=timer_cap, plot_errors=True, plot_variance=True)


if __name__ == "__main__":
    main()
