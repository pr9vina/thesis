import statsmodels.api as sm
from tqdm import tqdm
import pandas as pd
from methods.visualization import get_effect_plots


# TODO: method class
def linear_reg_estimation(
    individ_data: pd.DataFrame,
    n_features: int,
    additional_feature_names: list[str] = None
):
    feature_names = [f"сovariate_{i}" for i in range(n_features)] + ["group"]
    if additional_feature_names:
        feature_names += additional_feature_names
    X_df = individ_data[feature_names]
    X = sm.add_constant(X_df)
    model = sm.OLS(individ_data["outcome"], X)
    results = model.fit()
    coef = results.params["group"]
    pval = results.pvalues["group"]

    return coef, pval


def calculate_lr_results(
    individ_data: pd.DataFrame,
    n_sim: int,
    n_features: int,
    additional_feature_names: list[str] = None,
    network_features_df: pd.DataFrame = None,
    node_community_df: pd.DataFrame = None,
):
    coef_list = []
    pvalue_list = []
    for i_sim in tqdm(range(n_sim)):
        if additional_feature_names:
            df_sim = process_additional_features(i_sim, network_features_df, individ_data, node_community_df)
        else:
            df_sim = individ_data[individ_data.n_sim == i_sim]
        coef, pvalue = linear_reg_estimation(
            individ_data=df_sim,
            n_features=n_features,
            additional_feature_names=additional_feature_names
        )
        coef_list.append(coef)
        pvalue_list.append(pvalue)
    get_effect_plots(estimated_effect=coef_list, true_effect=individ_data["treatment_effect"])


def process_additional_features(i_sim, network_features_df, individ_data, node_community_df):
    network_features_df_sim = network_features_df[network_features_df['n_sim'] == i_sim].drop(columns=['n_sim'])
    individ_data_sim = individ_data[individ_data['n_sim'] == i_sim].drop(columns=['n_sim'])
    node_community_df_sim = node_community_df[node_community_df['n_sim'] == i_sim].drop(columns=['n_sim'])
    df_1 = individ_data_sim.merge(network_features_df_sim.reset_index(drop=True).reset_index(), on='index', how='left')
    df_1.rename(columns={'index': 'node'}, inplace=True)
    df = df_1.merge(node_community_df_sim, on='node', how='left').drop(columns=['node'])
    df["community"] = df["community"].astype('category')

    return df
