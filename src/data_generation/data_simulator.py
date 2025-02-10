import scipy
import numpy as np
import pandas as pd


class IndividualDataSimulator:
    def __init__(self, n_nodes, n_features, error_mean, error_std, beta_mean, beta_std, share_treatment, group_assignment_type):
        """
        group_assignment_type: 'random', 'individual_covariates' или 'individual_and_neighbors'
        """
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.error_mean = error_mean
        self.error_std = error_std
        self.beta_mean = beta_mean
        self.beta_std = beta_std
        self.share_treatment = share_treatment
        self.group_assignment_type = group_assignment_type

    def generate_random_covariates(self) -> np.ndarray:
        # Генерация ковариат, например, из нормального распределения
        return np.random.randn(self.n_nodes, self.n_features)

    def generate_random_error(self) -> np.ndarray:
        return np.random.normal(self.error_mean, self.error_std, self.n_nodes)

    def generate_random_beta_matrix(self) -> np.ndarray:
        return np.random.normal(self.beta_mean, self.beta_std, self.n_features)

    def set_treatment_effect(self, treatment_effect_mean: float, treatment_effect_std: float):
        self.treatment_effect_mean = treatment_effect_mean
        self.treatment_effect_std = treatment_effect_std

    def generate_treatment_effect(self) -> np.ndarray:
        return np.random.normal(self.treatment_effect_mean, self.treatment_effect_std)

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

    def generate_group_assignment(self, covariates: np.ndarray, adj_matrix: np.ndarray = None) -> np.ndarray:
        if self.group_assignment_type == "random":
            n_treatment = int(self.n_nodes * self.share_treatment)
            n_control = self.n_nodes - n_treatment
            group_assignment = np.array([1] * n_treatment + [0] * n_control)
            np.random.shuffle(group_assignment)
            return group_assignment
        elif self.group_assignment_type == "individual_covariates":
            scores = covariates @ np.random.normal(size=self.n_features)
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            return (scores < self.share_treatment).astype(int)
        elif self.group_assignment_type == "individual_and_neighbors":
            if adj_matrix is None:
                raise ValueError("Adjacency matrix required for individual_and_neighbors assignment.")
            neighbor_cov = adj_matrix @ covariates
            combined = np.hstack([covariates, neighbor_cov])
            scores = combined @ np.random.normal(size=combined.shape[1])
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            return (scores < self.share_treatment).astype(int)
        else:
            raise ValueError("Unknown group assignment type")

    def generate_individ_data(self, outcome, covariates, group_assignment):
        outcome_df = pd.DataFrame(outcome)
        outcome_df.columns = ["outcome"]
        covariates_df = pd.DataFrame(covariates)
        covariates_df.columns = [f"covariate_{i}" for i in range(self.n_features)]
        group_assignment_df = pd.DataFrame(group_assignment)
        group_assignment_df.columns = ["group"]
        return pd.concat([outcome_df, covariates_df, group_assignment_df], axis=1)
