from experiments.kroncker_experiments import kroneckerDataTester
from experiments.tt_experiments import TTDataTester

from experiments.analyze_results import AnalyzeResults

if __name__ == "__main__":

    # Run Kronecker Experiments
    order = [4, 5, 6, 7, 8]
    m_scalar_options = [4, 5, 6, 7]
    katri_rao_sketch_sizes = [1000, 3000, 5000, 10 ** 4, 15000, 20000]
    kroncker_data_tester = kroneckerDataTester(order=order, m_scalar_options=m_scalar_options, kronecker_factor=[1,2],
                                        kronecker_sketch_sizes=katri_rao_sketch_sizes)
    kroncker_data_tester.run_kronecker_order_test_and_save()

    # Run TT Experiments
    ranks = [2 ** i for i in range(2, 6)] + [40, 50, 64, 70]
    m_scalar_options = [3, 4, 5, 6, 7, 8]
    tt_data_tester = TTDataTester(rank=ranks, m_scalar_options=m_scalar_options)
    tt_data_tester.run_rank_test_and_save()

    AnalyzeResults.analyze_and_plot_tt_data(rank=70)
    AnalyzeResults.analyze_and_plot_kroncker_data(order=10)
    print("Saved graphs at:", "./results/graphs")