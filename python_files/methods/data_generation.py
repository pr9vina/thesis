import numpy as np
import networkx as nx
import scipy
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


class DataGeneration:
    def __init__(
        self,
        n_nodes,
        n_edges,
        n_features,
        beta_mean,
        beta_std,
        error_mean,
        error_std,
        cov_mean_range,
        cov_std_range,
        share_treatment,
        treatment_effect_mean,
        treatment_effect_std,
        n_influence_list,
        n_sim,
    ):
        # TODO some function to change
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.n_features = n_features
        self.beta_mean = beta_mean
        self.beta_std = beta_std
        self.error_mean = error_mean
        self.error_std = error_std
        self.cov_mean_range = cov_mean_range
        self.cov_std_range = cov_std_range
        self.share_treatment = share_treatment
        self.treatment_effect_mean = treatment_effect_mean
        self.treatment_effect_std = treatment_effect_std
        self.n_influence_list = n_influence_list
        self.n_sim = n_sim
        self.p_edges = None
        self.network_type = None

    def set_network_type(self, network_type: str):
        self.network_type = network_type

    def set_p_edges(self, p_edges: float):
        self.p_edges = p_edges

    def generate_covariates_mean_std(self):
        means = np.random.uniform(
            self.cov_std_range[0], self.cov_mean_range[1], self.n_features
        )
        stds = np.random.uniform(
            self.cov_std_range[0], self.cov_mean_range[1], self.n_features
        )
        return means, stds

    def generate_random_error(self) -> np.ndarray:
        error = np.random.normal(self.error_mean, self.error_std, self.n_nodes)
        return error

    def generate_random_beta_matrix(self) -> np.ndarray:
        beta = np.random.normal(self.beta_mean, self.beta_std, self.n_features)
        return beta

    def generate_random_covariates(self) -> np.ndarray:
        means, stds = self.generate_covariates_mean_std()
        covariates = np.zeros((self.n_nodes, self.n_features))
        for i in range(self.n_features):
            covariates[:, i] = np.random.normal(means[i], stds[i], self.n_nodes)
        return covariates

    def generate_group_assignment(self) -> np.ndarray:
        n_treatment = int(self.n_nodes * self.share_treatment)
        n_control = self.n_nodes - n_treatment
        group_assignment = np.array([1] * n_treatment + [0] * n_control)
        return group_assignment

    def generate_network(self, network_type):
        if network_type == "barabasi_albert_graph":
            sim_network = nx.barabasi_albert_graph(n=self.n_nodes, m=self.n_edges, seed=1234)
        if network_type == "erdos_renyi_graph":
            sim_network = nx.erdos_renyi_graph(n=self.n_nodes, p=self.p_edges)
        if network_type == "watts_strogatz_graph":
            sim_network = nx.watts_strogatz_graph(n=self.n_nodes, k=self.n_edges, p=self.p_edges)

        adj_matrix = nx.to_numpy_array(sim_network)
        return adj_matrix

    def generate_SAR_outcome(
        self,
        adj_matrix,
        neighbour_influence: float,
        group_assignment: np.ndarray,
        beta: np.ndarray,
        covariates: np.ndarray,
        error: np.ndarray,
        treatment_effect: float
    ) -> np.ndarray:
        adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
        I_matrix = np.eye(self.n_nodes)
        weight = scipy.linalg.inv(I_matrix - neighbour_influence*adj_matrix)

        if group_assignment is not None:
            outcome = weight@covariates@beta + weight@error + weight@group_assignment*treatment_effect
        else:
            outcome = weight@covariates@beta + weight@error

        return outcome

    def generate_treatment_effect(self) -> np.ndarray:
        return np.random.normal(self.treatment_effect_mean, self.treatment_effect_std)

    def generate_simulations(self, network_type: str, neighbour_influence: float, n_jobs: int = 1) -> dict:
        def simulate_single_data(i_sim, neighbour_influence):
            adj_matrix = self.generate_network(network_type)
            error = self.generate_random_error()
            beta = self.generate_random_beta_matrix()
            covariates = self.generate_random_covariates()
            group_assignment = self.generate_group_assignment()

            if self.treatment_effect_mean:
                treatment_effect = self.generate_treatment_effect()

            outcome = self.generate_SAR_outcome(
                adj_matrix=adj_matrix,
                neighbour_influence=neighbour_influence,
                group_assignment=group_assignment,
                beta=beta,
                covariates=covariates,
                error=error,
                treatment_effect=treatment_effect
            )

            # individual data
            outcome_df = pd.DataFrame(outcome)
            outcome_df.columns = ["outcome"]
            covariates_df = pd.DataFrame(covariates)
            covariates_df.columns = [f"сovariate_{i}" for i in range(self.n_features)]
            group_assignment_df = pd.DataFrame(group_assignment)
            group_assignment_df.columns = ["group"]
            individ_data = pd.concat([outcome_df, covariates_df, group_assignment_df], axis=1)
            if self.treatment_effect_mean:
                individ_data["treatment_effect"] = treatment_effect

            return individ_data, beta, adj_matrix

        # parallel computations (saved ~50%)
        parallel_pool = Parallel(n_jobs=n_jobs)
        data_simulated = parallel_pool(delayed(simulate_single_data)(i_sim, neighbour_influence) for i_sim in tqdm(range(self.n_sim)))

        return data_simulated

    def compute_neighbours_data_simulations(self):
        """Generate parallel simulations across multiple neigbour influence values"""
        for neighbour_influence in self.n_influence_list:
            print(f"Simulating {self.n_sim} simulations for {round(neighbour_influence, 2)} neighbour influence")
            data_simulated = self.generate_simulations(network_type=self.network_type, neighbour_influence=neighbour_influence)
            dict_adj_matrix = {}
            dict_beta = {}
            list_df = []
            for i, (individ_data, beta, adj_matrix) in enumerate(data_simulated):
                dict_adj_matrix[i] = adj_matrix
                dict_beta[i] = beta
                individ_data["n_sim"] = i
                list_df.append(individ_data)

            individ_data_all = pd.concat(list_df)
            individ_data_all.to_parquet(f"data/individ_data/individ_data_{self.network_type}_{self.n_sim}_{np.round(neighbour_influence, 2)}.parquet")
            print("Successfully loaded simulation data")

            with open(f"data/beta_matrix/dict_beta_{self.n_sim}_{self.network_type}_{np.round(neighbour_influence, 2)}.pkl", "wb") as file:
                pickle.dump(dict_beta, file)
            print("Beta saved")

            with open(f"data/adj_matrix/dict_adj_matrix_{self.n_sim}_{self.network_type}_{np.round(neighbour_influence, 2)}.pkl", "wb") as file:
                pickle.dump(dict_adj_matrix, file)
