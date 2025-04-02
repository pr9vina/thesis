import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from node2vec import Node2Vec
import itertools


class GraphGenerator:
    def __init__(self, n_nodes, n_edges, p_edges=None):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.p_edges = p_edges
    
    def generate_network(self, network_type: str) -> np.ndarray:
        if network_type == "barabasi_albert_graph":
            graph = nx.barabasi_albert_graph(n=self.n_nodes, m=self.n_edges, seed=1234)
        elif network_type == "erdos_renyi_graph":
            graph = nx.erdos_renyi_graph(n=self.n_nodes, p=self.p_edges)
        elif network_type == "watts_strogatz_graph":
            graph = nx.watts_strogatz_graph(n=self.n_nodes, k=self.n_edges, p=self.p_edges)
        else:
            raise ValueError("Unsupported network type")
      
        return nx.to_numpy_array(graph)


class NetworkFeatureExtractor:
    def extract_network_features(self, adj_matrix: np.ndarray) -> (pd.DataFrame, pd.DataFrame):
        features_list = []
        community_list = []
        G = nx.from_numpy_array(adj_matrix)
        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(G)
        communities_generator = nx.algorithms.community.girvan_newman(G)
        first_partition = next(communities_generator, [set(G.nodes())])
        node_comm = {node: idx for idx, community in enumerate(first_partition) for node in community}

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
        
        return pd.DataFrame(features_list), pd.DataFrame(community_list)
    
    def count_treated_neighbors(
        self,
        adj_matrix: np.ndarray,
        treatment_assignment: np.ndarray
    ) -> pd.DataFrame:
    
        G = nx.from_numpy_array(adj_matrix)
        treated_neighbors_list = [
            {
                "node": node, 
                "treated_neighbors_count": sum(1 for neighbor in G.neighbors(node) if treatment_assignment[neighbor] == 1),
                "treated_neighbors_exposure": sum(treatment_assignment[neighbor] for neighbor in G.neighbors(node)) / len(list(G.neighbors(node))) if len(list(G.neighbors(node))) > 0 else 0
            }
            for node in G.nodes()
        ]

        return pd.DataFrame(treated_neighbors_list)

    def generate_node_embeddings(
        self,
        adj_matrix: np.ndarray,
        dimensions: int,
        walk_length: int,
        num_walks: int,
        p: float,
        q: float,
        window: int,
        min_count: int,
        batch_words: int
    ) -> pd.DataFrame:
        embeddings_list = []
        G = nx.from_numpy_array(adj_matrix)
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=4)
        model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)

        for node in G.nodes():
            embeddings_list.append({
                "node": node,
                **{f"emb_{i}": value for i, value in enumerate(model.wv[str(node)])}
            })

        return pd.DataFrame(embeddings_list)


class StochasticBlockModel:
    def __init__(self, n_blocks: int, inter_block_probs: np.ndarray):
        """
        SBM where blocks are determined based on node covariates.
        
        :param n_blocks: Number of blocks (groups).
        :param inter_block_probs: A (n_blocks, n_blocks) matrix of connection probabilities between blocks.
        """
        self.n_blocks = n_blocks
        self.inter_block_probs = inter_block_probs

    def assign_blocks(self, covariates: np.ndarray):
        """
        Determines block assignments based on clustering nodes by their covariates.
        
        :param covariates: A (n_nodes, n_features) matrix of node covariates.
        :return: A vector of block assignments for the nodes.
        """
        kmeans = KMeans(n_clusters=self.n_blocks, random_state=42, n_init=10)
        return kmeans.fit_predict(covariates)

    def generate_network(self, covariates: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Generates a network according to the SBM model based on clustered covariates.
        
        :param covariates: A matrix of node covariates.
        :return: An adjacency matrix (n_nodes, n_nodes) and a vector of node block assignments.
        """
        n_nodes = covariates.shape[0]
        block_assignments = self.assign_blocks(covariates)  # Determine blocks

        adjacency_matrix = np.zeros((n_nodes, n_nodes))

        for i, j in itertools.combinations(range(n_nodes), 2):
            block_i = block_assignments[i]
            block_j = block_assignments[j]
            prob = self.inter_block_probs[block_i, block_j]  # Connection probability

            if np.random.rand() < prob:
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1  # Add edge

        return adjacency_matrix, block_assignments
