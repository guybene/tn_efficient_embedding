import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

from experiments.tt_experiments import TTDataTester
from experiments.kroncker_experiments import KroneckerDataTester


class AnalyzeResults:

    @staticmethod
    def analyze_and_plot_tt_data(rank):
        data_path = TTDataTester.TT_RANK_DATA_PATH.format(rank=rank)
        res = pd.read_csv(data_path, index_col=0)
        passed_configs = res[(np.abs(res["sketch_score_1"] - 1) <= 0.2) &
                             (np.abs(res["sketch_score_2"] - 1) <= 0.2)]

        minimal_viable_m_per_config = passed_configs.groupby(["rank",
                                                              "batch_num", "algo"])[
            ["actual_m", "cost"]].min().reset_index()
        m_graph = sns.relplot(data=minimal_viable_m_per_config, x="rank", y="actual_m", hue="algo",
                              err_style="bars", kind="line", ci=25)
        m_graph.set_axis_labels("Rank", "Sketch Size")
        m_graph.savefig(f"./results/graphs/tt_m_graph.jpeg")

        cost_graph = sns.relplot(data=minimal_viable_m_per_config, x="rank", y="cost", hue="algo",
                                 err_style="bars", kind="line", ci=25)
        cost_graph.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

        cost_graph.set_axis_labels("Rank", "Flops")
        cost_graph.savefig(f"./results/graphs/tt_cost_graph.jpeg")

    @staticmethod
    def analyze_and_plot_kroncker_data(order):
        data_path = KroneckerDataTester.KRONCKER_DATA_PATH.format(order=order)
        res = pd.read_csv(data_path, index_col=0)
        passed_configs = res[(np.abs(res["sketch_score_1"] - 1) <= 0.1) &
                             (np.abs(res["sketch_score_2"] - 1) <= 0.1)]

        minimal_viable_m_per_config = passed_configs.groupby(["order",
                                                              "batch_num", "algo"])[
            ["actual_m", "cost"]].min().reset_index()
        m_graph = sns.relplot(data=minimal_viable_m_per_config, x="order", y="actual_m", hue="algo",
                              err_style="bars", kind="line", ci=25)
        m_graph.set_axis_labels("Order", "Sketch Size")
        m_graph.savefig(f"./results/graphs/kroncker_m_graph.jpeg")

        cost_graph = sns.relplot(data=minimal_viable_m_per_config, x="order", y="cost", hue="algo",
                                 err_style="bars", kind="line", ci=25)
        cost_graph.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

        cost_graph.set_axis_labels("Order", "Flops")
        cost_graph.savefig(f"./results/graphs/kroncker_cost_graph.jpeg")

        passed_configs_tn_and_tree = passed_configs[passed_configs["algo"].isin(["TN", "Tree"])]
        minimal_viable_m_per_config_tn_tree = passed_configs_tn_and_tree.groupby(["order",
                                                                                  "batch_num", "algo"])[
            ["actual_m", "cost"]].min().reset_index()
        m_tn_tree_graph = sns.relplot(data=minimal_viable_m_per_config_tn_tree, x="order", y="actual_m", hue="algo",
                                      err_style="bars", kind="line", ci=25)
        m_tn_tree_graph.set_axis_labels("Order", "Sketch Size")
        m_tn_tree_graph.savefig(f"./results/graphs/kroncker_m_tree_tn_graph.jpeg")
