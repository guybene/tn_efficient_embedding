import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker


from experiments.tt_experiments import TTDataTester
from experiments.kroncker_experiments import KronckerDataTester


class AnalyzeResults:

    @staticmethod
    def analyze_and_plot(data_path, param, name, sketch_err_threshold):
        res = pd.read_csv(data_path, index_col=0)
        passed_configs = res[(np.abs(res["sketch_score_1"] - 1) <= sketch_err_threshold) &
                             (np.abs(res["sketch_score_2"] - 1) <= sketch_err_threshold)]

        minimal_viable_m_per_config = passed_configs.groupby([param,
                                                              "batch_num", "algo"])[["actual_m", "cost"]].min().reset_index()
        m_graph = sns.relplot(data=minimal_viable_m_per_config, x=param, y="actual_m", hue="algo",
                    err_style="bars", kind="line", ci=25)
        m_graph.set_axis_labels(param[0].upper() + param[1:], "Sketch Size")
        m_graph.savefig(f"./results/graphs/{name}_m_graph.jpeg")

        cost_graph = sns.relplot(data=minimal_viable_m_per_config, x=param, y="cost", hue="algo",
                    err_style="bars", kind="line", ci=25)
        cost_graph.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

        cost_graph.set_axis_labels(param[0].upper() + param[1:], "Flops")
        cost_graph.savefig(f"./results/graphs/{name}_cost_graph.jpeg")





if __name__ == "__main__":
    AnalyzeResults.analyze_and_plot(data_path=TTDataTester.TT_RANK_DATA_PATH.format(rank=8),
                                    param="rank",
                                    name="tt",
                                    sketch_err_threshold=0.2)
    AnalyzeResults.analyze_and_plot(data_path=KronckerDataTester.KRONCKER_DATA_PATH.format(order=10),
                                    param="order",
                                    name="kroncker",
                                    sketch_err_threshold=0.1)

