import numpy as np
import networkx as nx
import scipy
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import warnings
import logging
from node2vec import Node2Vec
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
        n_influence_list,
        n_sim,
    ):
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
        self.n_influence_list = n_influence_list
        self.n_sim = n_sim
        self.p_edges = None
        self.network_type = None

    def set_network_type(self, network_type: str):
        self.network_type = network_type

    def set_treatment_effect(self, treatment_effect_mean: float, treatment_effect_std: float):
        self.treatment_effect_mean = treatment_effect_mean
        self.treatment_effect_std = treatment_effect_std

    def set_group_assignment_type(self, group_assignment_type: str):
        if group_assignment_type not in ['random', 'individual_covariates', 'individual_and_neighbors']:
            raise ValueError(f"Unknown method '{self.group_assignment_type}'. Use random, 'ndividual_covariates, or individual_and_neighbors")
        self.group_assignment_type = group_assignment_type

    def set_p_edges(self, p_edges: float):
        self.p_edges = p_edges

    def set_neighbour_influence(self, neighbour_influence: float):
        self.neighbour_influence = neighbour_influence

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

    def generate_node_embeddings(self, adj_matrix: np.ndarray, dimensions: int = 64, walk_length: int = 30, num_walks: int = 200, p: float = 1, q: float = 1) -> pd.DataFrame:
        """
        Генерирует эмбеддинги для узлов в сети с помощью Node2Vec.
        
        Параметры:
        - dict_adj_matrix: словарь {sim: матрица смежности} для каждой симуляции.
        - dimensions: размерность эмбеддинга.
        - walk_length: длина одного случайного обхода.
        - num_walks: количество случайных обходов на узел.
        - p, q: параметры управления стратегией обхода.
        
        Возвращает:
        - DataFrame (n_sim, node, emb_1, emb_2, ..., emb_N).
        """
        embeddings_list = []

        G = nx.from_numpy_array(adj_matrix)

        # Обучение Node2Vec
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        for node in G.nodes():
            embeddings_list.append({
                "node": node,
                **{f"emb_{i}": value for i, value in enumerate(model.wv[str(node)])}
            })
        return pd.DataFrame(embeddings_list)

    def extract_network_features(self, adj_matrix: np.ndarray) -> (pd.DataFrame, pd.DataFrame):

        features_list = []
        community_list = []

        G = nx.from_numpy_array(adj_matrix)
        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(G)
        communities_generator = nx.algorithms.community.girvan_newman(G)
        try:
            first_partition = next(communities_generator)
        except StopIteration:
            first_partition = [set(G.nodes())] # no communities
        node_comm = {}
        for idx, community in enumerate(first_partition):
            for node in community:
                node_comm[node] = idx

        for node in G.nodes():
            features_list.append({
                "node": node,
                "degree_centrality": deg_cent[node],
                "betweenness_centrality": bet_cent[node]
            })
            community_list.append({
                "node": node,
                "community": node_comm.get(node, -1)
            })

        df_features = pd.DataFrame(features_list)
        df_communities = pd.DataFrame(community_list)
        return pd.merge(df_features, df_communities, on="node", how="left")

    def generate_group_assignment(self, covariates=None, adj_matrix=None):
        """
        Generate group assignments with different methods.
        
        Parameters:
            covariates (np.ndarray): Covariate matrix for individual assignment. Required for "individual_covariates" and "individual_and_neighbors".
            adj_matrix (np.ndarray): Adjacency matrix for neighborhood influence. Required for "individual_and_neighbors".
 
        Returns:
            group_assignment (np.ndarray): Generated group assignments.
        """
        if self.group_assignment_type == "random":
            n_treatment = int(self.n_nodes * self.share_treatment)
            n_control = self.n_nodes - n_treatment
            group_assignment = np.array([1] * n_treatment + [0] * n_control)
            np.random.shuffle(group_assignment)

        elif self.group_assignment_type == "individual_covariates":
            if covariates is None:
                raise ValueError("Covariates are required")
            scores = np.dot(covariates, np.random.normal(size=self.n_features)) # Linear combination of covariates
            scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize to [0, 1]
            group_assignment = (scores < self.share_treatment).astype(int)  # Assign based on threshold

        elif self.group_assignment_type == "individual_and_neighbors":
            if covariates is None or adj_matrix is None:
                raise ValueError("Covariates and adjacency matrix are required")
            neighbor_covariates = adj_matrix @ covariates
            combined_covariates = np.hstack([covariates, neighbor_covariates])  # Combine individual and neighbor covariates
            scores = np.dot(combined_covariates, np.random.normal(size=combined_covariates.shape[1]))  # Linear combination
            scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize to [0, 1]
            group_assignment = (scores < self.share_treatment).astype(int)  # Assign based on threshold

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

    def generate_simulations(
            self,
            network_type: str,
            neighbour_influence: float,
            n_jobs: int = 1
        ) -> dict:
        def simulate_single_data(i_sim, neighbour_influence):
            adj_matrix = self.generate_network(network_type)
            error = self.generate_random_error()
            beta = self.generate_random_beta_matrix()
            covariates = self.generate_random_covariates()
            group_assignment = self.generate_group_assignment(covariates=covariates, adj_matrix=adj_matrix)

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
            covariates_df.columns = [f"covariate_{i}" for i in range(self.n_features)]
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

    def write_results(individ_data_all, dict_beta, dict_adj_matrix):
        base_path = "data/"
        file_template = f"{base_path}{{folder}}/dict_{{name}}_{self.group_assignment_type}_{self.n_sim}_{self.network_type}_{np.round(self.neighbour_influence, 2)}.{{ext}}"
        individ_path = file_template.format(folder="individ_data", name="individ_data", ext="parquet")
        beta_path = file_template.format(folder="beta_matrix", name="beta", ext="pkl")
        adj_matrix_path = file_template.format(folder="adj_matrix", name="adj_matrix", ext="pkl")

        individ_data_all.to_parquet(individ_path)
        logging.info("Successfully saved simulation data")

        with open(beta_path, "wb") as file:
            pickle.dump(dict_beta, file)
        logging.info("Successfully saved beta dict data")

        with open(adj_matrix_path, "wb") as file:
            pickle.dump(dict_adj_matrix, file)
        logging.info("Successfully saved adjacency matrix data")

    def compute_neighbours_data_simulations(self, write_results=False):
        """Generate parallel simulations across multiple neigbour influence values"""
        logging.info(f"Simulating {self.n_sim} simulations for {round(self.neighbour_influence, 2)} neighbour influence")
        data_simulated = self.generate_simulations(network_type=self.network_type, neighbour_influence=self.neighbour_influence)
        dict_adj_matrix = {}
        dict_beta = {}
        list_df = []
        for i, (individ_data, beta, adj_matrix) in enumerate(data_simulated):
            dict_adj_matrix[i] = adj_matrix
            dict_beta[i] = beta
            individ_data["n_sim"] = i
            list_df.append(individ_data)
        individ_data_all = pd.concat(list_df)

        if write_results:
            write_results(individ_data_all, dict_beta, dict_adj_matrix)
        return individ_data_all, dict_beta, dict_adj_matrix
