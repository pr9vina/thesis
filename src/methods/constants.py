from pydantic_settings import BaseSettings
import numpy as np


class NetworkSettings(BaseSettings):
    n_nodes: int = 100
    n_edges: int = 3
    networks_types: list = [
    ("barabasi_albert_graph", None),
    ("watts_strogatz_graph", 0.7),
    ("erdos_renyi_graph", 0.2)
]

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
