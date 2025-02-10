import numpy as np
from pydantic_settings import BaseSettings
from typing import Dict, Any, List


class NetworkSettings(BaseSettings):
    # basic settings
    n_features: int = 2
    n_nodes: int = 100
    n_edges: int = 3
    networks_types: list = [
        ("barabasi_albert_graph", None),
        ("watts_strogatz_graph", 0.7),
        ("erdos_renyi_graph", 0.2)
    ]

    # node2vec settings
    vec_walk_length: int = 30
    vec_num_walks: int = 200
    p_random_walk: float = 1.0
    q_random_walk: float = 1.0
    vec_dimensions: int = 64
    vec_window: int = 10
    min_count: int = 1
    batch_words: int = 4

    # feature sets
    feature_adj_sets: Dict[str, Any] = {
        "naiv": None,
        "network_features_v1": ["degree_centrality", "betweenness_centrality", "community"],
        "network_features_v2": ["degree_centrality", "betweenness_centrality"],
        "network_features_v3": ["community"],
        "embedding_features": [f"emb_{i}" for i in range(vec_dimensions)],
    }

    covariates_features: List = [f"covariate_{i}" for i in range(n_features)]
    feature_adj_sets_ps: Dict[str, Any] = {
        "naiv": covariates_features,
        "treated_covs_v2": covariates_features + ["treated_neighbors_count"],
        "network_features_v1": covariates_features + ["degree_centrality", "betweenness_centrality", "community"],
        "network_features_v2": covariates_features + ["degree_centrality", "betweenness_centrality"],
        "network_features_v3": covariates_features + ["community"],
        "embedding_features": covariates_features + [f"emb_{i}" for i in range(vec_dimensions)],
    }

    # Stochastic Block Model
    n_blocks: int = 3
    inter_block_probs: np.array = np.array([[0.8, 0.1, 0.05],
                                            [0.1, 0.7, 0.1],
                                            [0.05, 0.1, 0.6]])


class DataGenerationSettings(BaseSettings):
    n_sim: int = 100
    n_features: int = 2
    share_treatment: int = 0.5
    beta_mean: int = 3
    beta_std: float = 0
    error_mean: float = 0.3
    error_std: float = 0
    cov_mean_range: tuple = (3, 10)
    cov_std_range: tuple = (0.4, 0.8)
    share_treatment: float = 0.5
    n_influence_list: list = list(np.arange(0.3, 1, 0.3))
    treatment_effects: list = [(0.3, 0), (0.6, 0), (0.9, 0)]
    treatment_effect_mean: float = 0.3
    treatment_effect_std: float = 0
    p_edges: float = 0.3
    assignment_types: list = ["random", "individual_covariates", "individual_and_neighbors"]
