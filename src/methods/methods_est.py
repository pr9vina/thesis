import statsmodels.api as sm
from tqdm import tqdm
import pandas as pd
from methods.visualization import ResultsPlotter


class LinearRegressionEstimator:
    def __init__(
        self,
        individ_data: pd.DataFrame,
        n_features: int,
        n_sim: int,
        network_type: str,
        assignment_type: str,
        neighbour_influence: float,
        treatment_mean: float
    ):
        self.individ_data = individ_data
        self.n_features = n_features
        self.n_sim = n_sim
        self.network_type = network_type
        self.assignment_type = assignment_type
        self.neighbour_influence = neighbour_influence
        self.treatment_mean = treatment_mean

    def _get_feature_names(self, additional_feature_names: list[str] = None):
        """Generate a list of feature names based on the number of features and additional features."""
        feature_names = [f"covariate_{i}" for i in range(self.n_features)] + ["group"]
        if additional_feature_names:
            feature_names += additional_feature_names
        return feature_names

    def linear_regression(self, df: pd.DataFrame, additional_feature_names: list[str] = None):
        """Perform linear regression and return the estimated treatment effect coefficient and p-value."""
        feature_names = self._get_feature_names(additional_feature_names)
        X = sm.add_constant(df[feature_names])
        model = sm.OLS(df["outcome"], X)
        results = model.fit()
        return results.params["group"], results.pvalues["group"]

    def process_additional_features(self, i_sim, network_features_df, node_community_df):
        """Merge additional network and community features into the dataset for a given simulation"""
        network_features_df_sim = network_features_df[network_features_df['n_sim'] == i_sim].drop(columns=['n_sim'])
        individ_data_sim = self.individ_data[self.individ_data['n_sim'] == i_sim].drop(columns=['n_sim'])
        node_community_df_sim = node_community_df[node_community_df['n_sim'] == i_sim].drop(columns=['n_sim'])

        df = (
            individ_data_sim
            .merge(network_features_df_sim.reset_index(drop=True).reset_index(), on='index', how='left')
            .rename(columns={'index': 'node'})
            .merge(node_community_df_sim, on='node', how='left')
            .drop(columns=['node'])
        )
        df["community"] = df["community"].astype('category')
        return df

    def calculate_results(self, additional_feature_names: list[str] = None, network_features_df: pd.DataFrame = None, node_community_df: pd.DataFrame = None):
        """Compute linear regression results across simulations and plot estimated vs. true effects."""
        coef_list, pvalue_list = [], []

        for i_sim in tqdm(range(self.n_sim)):
            if additional_feature_names:
                df_sim = self.process_additional_features(i_sim, network_features_df, node_community_df)
            else:
                df_sim = self.individ_data[self.individ_data.n_sim == i_sim]

            coef, pvalue = self.linear_regression(df_sim, additional_feature_names)
            coef_list.append(coef)
            pvalue_list.append(pvalue)

        plotter = ResultsPlotter(
            estimated_effect=coef_list,
            true_effect=self.individ_data["treatment_effect"],
            network_type=self.network_type,
            assignment_type=self.assignment_type,
            neighbour_influence=self.neighbour_influence,
            treatment_mean=self.treatment_mean
        )
        plotter.calculate_results()
        plotter.create_plots()
        plotter.create_results()



# # TODO: method class
# def linear_reg_estimation(
#     individ_data: pd.DataFrame,
#     n_features: int,
#     additional_feature_names: list[str] = None
# ):
#     feature_names = [f"сovariate_{i}" for i in range(n_features)] + ["group"]
#     if additional_feature_names:
#         feature_names += additional_feature_names
#     X_df = individ_data[feature_names]
#     X = sm.add_constant(X_df)
#     model = sm.OLS(individ_data["outcome"], X)
#     results = model.fit()
#     coef = results.params["group"]
#     pval = results.pvalues["group"]

#     return coef, pval


# def calculate_lr_results(
#     individ_data: pd.DataFrame,
#     n_sim: int,
#     n_features: int,
#     additional_feature_names: list[str] = None,
#     network_features_df: pd.DataFrame = None,
#     node_community_df: pd.DataFrame = None,
# ):
#     coef_list = []
#     pvalue_list = []
#     for i_sim in tqdm(range(n_sim)):
#         if additional_feature_names:
#             df_sim = process_additional_features(i_sim, network_features_df, individ_data, node_community_df)
#         else:
#             df_sim = individ_data[individ_data.n_sim == i_sim]
#         coef, pvalue = linear_reg_estimation(
#             individ_data=df_sim,
#             n_features=n_features,
#             additional_feature_names=additional_feature_names
#         )
#         coef_list.append(coef)
#         pvalue_list.append(pvalue)
#     get_effect_plots(estimated_effect=coef_list, true_effect=individ_data["treatment_effect"])


# def process_additional_features(i_sim, network_features_df, individ_data, node_community_df):
#     network_features_df_sim = network_features_df[network_features_df['n_sim'] == i_sim].drop(columns=['n_sim'])
#     individ_data_sim = individ_data[individ_data['n_sim'] == i_sim].drop(columns=['n_sim'])
#     node_community_df_sim = node_community_df[node_community_df['n_sim'] == i_sim].drop(columns=['n_sim'])
#     df_1 = individ_data_sim.merge(network_features_df_sim.reset_index(drop=True).reset_index(), on='index', how='left')
#     df_1.rename(columns={'index': 'node'}, inplace=True)
#     df = df_1.merge(node_community_df_sim, on='node', how='left').drop(columns=['node'])
#     df["community"] = df["community"].astype('category')

#     return df
