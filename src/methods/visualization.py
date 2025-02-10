import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


class ResultsPlotter:
    def __init__(
        self,
        estimated_effect,
        true_effect,
        network_type,
        assignment_type,
        neighbour_influence,
        treatment_mean,
        additional_feature_names
    ):
        self.estimated_effect = estimated_effect
        self.true_effect = true_effect
        self.network_type = network_type
        self.assignment_type = assignment_type
        self.neighbour_influence = neighbour_influence
        self.treatment_mean = treatment_mean
        self.additional_feature_names = additional_feature_names

    def _plot_histogram(self, ax, data, color, title, line_color='black'):
        ax.hist(data, bins=30, alpha=0.7, color=color)
        ax.axvline(np.mean(data), color=line_color, linestyle='--', linewidth=2)
        ax.set_title(title)

    def _calculate_percentage_difference(self, mean_true, mean_estimated):
        mean_true = np.mean(self.true_effect)
        mean_estimated = np.mean(self.estimated_effect)
        return np.abs((mean_true - mean_estimated) / ((mean_true + mean_estimated) / 2)) * 100

    def _save_plot(self, fig):
        graph_name = f"{self.network_type}_{self.assignment_type}_{self.neighbour_influence}_{self.treatment_mean}.png"
        plot_filename = f"visualization_results/{graph_name}"
        fig.savefig(plot_filename)
        plt.close(fig)
        logging.info(f"Plot saved to {plot_filename}")

    def calculate_results(self):
        self.mean_estimated = np.mean(self.estimated_effect)
        self.mean_true = np.mean(self.true_effect)
        self.perc_diff = self._calculate_percentage_difference(self.mean_true, self.mean_estimated)

    def create_results(self):
        results_df = pd.DataFrame([{
            "Network": self.network_type,
            "Assignment": self.assignment_type,
            "Influence": self.neighbour_influence,
            "Treatment": self.treatment_mean,
            "Mean Estimated": round(self.mean_estimated, 3),
            "Mean True": round(self.mean_true, 3),
            "Perc Diff (%)": round(self.perc_diff, 3),
            "Feature Names": self.additional_feature_names,
        }])
        table_name = f"markdown_results/new/{self.network_type}_{self.assignment_type}_{self.neighbour_influence}_{self.treatment_mean}.pkl"
        results_df.to_pickle(table_name)

        return results_df

    def create_plots(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self._plot_histogram(ax1, self.estimated_effect, 'blue', f'Estimated effects: {np.round(self.mean_estimated, 3)}')
        self._plot_histogram(ax2, self.true_effect, 'green', f'True effects: {np.round(self.mean_true, 3)}')
        logging.info(f"Percentage difference between true and estimated effect: {np.round(self.perc_diff, 3)}")
        graph_name = f"Network: {self.network_type}, Assignment: {self.assignment_type}, Influence: {self.neighbour_influence}, Treatment: {self.treatment_mean}, Perc Diff: {self.perc_diff}"
        fig.suptitle(graph_name, fontsize=8, fontweight='bold')
        self._save_plot(fig)
