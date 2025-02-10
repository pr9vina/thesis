import logging
import pickle
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from utils.logging_results import setup_logger
from utils.saving_results import save_to_pickle
from data_generation.data_simulator import IndividualDataSimulator
from data_generation.network_generator import GraphGenerator, NetworkFeatureExtractor, StochasticBlockModel
from methods.visualization import ResultsPlotter
from methods.methods_est import LinearRegressionEstimator
from utils.warnings import validate_argument


def run_data_simulations(
    assignment_types: List[str],
    treatment_effects: List[Tuple[float, int]],
    network_configs: List[tuple[str, float | None]],
    neighbour_influences: List[float],
    n_nodes: int,
    n_features: int,
    error_mean: float,
    error_std: float,
    beta_mean: float,
    beta_std: float,
    share_treatment: float,
    n_edges: int,
    n_sim: int,
    network_structure_type: str = "random",
    n_blocks: int = None,
    inter_block_probs: int = None,
    add_network_features: bool = False,
    add_embeddings: bool = False,
    add_count_treated: bool = False,
    dimensions: int = None,
    walk_length: int = None,
    num_walks: int = None,
    p_random_walk: float = None,
    q_random_walk: float = None,
    vec_window: int = None,
    min_count: int = None,
    batch_words: int = None,
    save_data: bool = True
) -> Dict[str, Any]:
    """Function for data generation simulations

    Args:
        assignment_types (List[str]): List of treatment assignment types 
        treatment_effects (List[Tuple[float, int]]): List of tuples representing treatment effects
        network_configs (List[Tuple[str, Optional[float]]]): List of network types and their parameters
            The first value is the network type (e.g., "barabasi_albert_graph"), 
            and the second is an optional parameter
        neighbour_influences (List[float]): List of influence values from neighbors
        n_nodes (int): Number of nodes in the network
        n_features (int): Number of features
        error_mean (float): Mean of the error term
        error_std (float): Standard deviation of the error term
        beta_mean (float): Mean of the beta
        beta_std (float): Standard deviation of the beta
        share_treatment (float): Proportion of treated nodes in the network
        n_edges (int): Number of edges in the network
        n_sim (int): Number of simulation runs to perform
        add_network_features (bool, optional): Whether to include network-based features. Defaults to False.
        add_embeddings (bool, optional): Whether to compute and include node embeddings. Defaults to False.
        dimensions (Optional[int], optional): Number of dimensions for node embeddings. Defaults to None.
        walk_length (Optional[int], optional): Length of random walks for node embeddings. Defaults to None.
        num_walks (Optional[int], optional): Number of random walks per node. Defaults to None.
        p_random_walk (Optional[float], optional): Return parameter for random walks. Defaults to None.
        q_random_walk (Optional[float], optional): In-out parameter for random walks. Defaults to None.
        vec_window (Optional[int], optional): Context window size for Node2Vec embeddings. Defaults to None.
        min_count (Optional[int], optional): Minimum word count for embeddings training. Defaults to None.
        batch_words (Optional[int], optional): Batch size for Node2Vec training. Defaults to None.
        save_data (bool, optional): Whether to save the generated data to a file. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing the generated simulation data, including network structures, 
            treatment assignments, feature values, and results.
    """
    setup_logger()
    logging.info("Checking validity...")
    validate_argument(
        value=network_structure_type,
        allowed_values=['random', 'covariate_block'],
        argument_name='network_structure_type',
    )
    logging.info("Starting data simulations...")
    results = []
    
    total_iterations = (
        len(assignment_types) * len(treatment_effects) * len(network_configs) * len(neighbour_influences) * n_sim
    )
    
    with tqdm(total=total_iterations, desc="Simulations Progress") as pbar:
        for assignment_type in assignment_types:
            individual_data_simulator = IndividualDataSimulator(
                n_nodes=n_nodes,
                n_features=n_features,
                error_mean=error_mean,
                error_std=error_std,
                beta_mean=beta_mean,
                beta_std=beta_std,
                share_treatment=share_treatment,
                group_assignment_type=assignment_type
            )
            
            for treatment_effect_mean, treatment_effect_std in treatment_effects:
                individual_data_simulator.set_treatment_effect(
                    treatment_effect_mean=treatment_effect_mean,
                    treatment_effect_std=treatment_effect_std
                )
                
                for network_type, p_edges in network_configs:
                    network_feature_extractor = NetworkFeatureExtractor()
                    if network_structure_type == "random":
                        graph_generator = GraphGenerator(
                            n_nodes=n_nodes,
                            n_edges=n_edges,
                            p_edges=p_edges
                        )
                    else:
                        graph_generator = StochasticBlockModel(
                            n_blocks=n_blocks,
                            inter_block_probs=inter_block_probs
                        )
                                            
                    for influence in neighbour_influences:
                        simulation_artifacts = []
                        
                        for i_sim in range(n_sim):
                            logging.info(f"Running simulation {i_sim+1}/{n_sim} for {assignment_type}, {network_type}, influence {influence}")                            
                            error = individual_data_simulator.generate_random_error()
                            beta = individual_data_simulator.generate_random_beta_matrix()
                            covariates = individual_data_simulator.generate_random_covariates()
                            if network_structure_type == "random":
                                adj_matrix = graph_generator.generate_network(network_type)
                            else:
                                adj_matrix, block_assignments = graph_generator.generate_network(covariates)

                            group_assignment = individual_data_simulator.generate_group_assignment(
                                covariates=covariates, 
                                adj_matrix=adj_matrix
                            )
                            
                            outcome = individual_data_simulator.generate_SAR_outcome(
                                adj_matrix=adj_matrix,
                                neighbour_influence=influence,
                                group_assignment=group_assignment,
                                beta=beta,
                                covariates=covariates,
                                error=error,
                                treatment_effect=individual_data_simulator.generate_treatment_effect()
                            )
                            
                            individ_data = individual_data_simulator.generate_individ_data(
                                outcome=outcome,
                                covariates=covariates,
                                group_assignment=group_assignment
                            )
                            
                            if add_network_features:
                                df_features, df_communities = network_feature_extractor.extract_network_features(adj_matrix=adj_matrix)
                                individ_data = individ_data.join(df_features, how="left")
                                individ_data = individ_data.merge(df_communities, on="node", how="left")

                            if add_count_treated:
                                df_treated_neighbors = network_feature_extractor.count_treated_neighbors(
                                    adj_matrix=adj_matrix, 
                                    treatment_assignment=group_assignment
                                )
                                individ_data = individ_data.merge(df_treated_neighbors, on="node", how="left")

                            if add_embeddings:
                                df_embeddings = network_feature_extractor.generate_node_embeddings(
                                    adj_matrix=adj_matrix,
                                    dimensions=dimensions,
                                    walk_length=walk_length,
                                    num_walks=num_walks,
                                    p=p_random_walk,
                                    q=q_random_walk,
                                    window=vec_window,
                                    min_count=min_count,
                                    batch_words=batch_words
                                )
                                individ_data = individ_data.merge(df_embeddings, on="node", how="left")
                            
                            artifact = {
                                "i_sim": i_sim,
                                "individ_data": individ_data,
                                "adj_matrix": adj_matrix
                            }
                            simulation_artifacts.append(artifact)
                            pbar.update(1)
                        
                        param_summary = {
                            "assignment_type": assignment_type,
                            "treatment_effect_mean": treatment_effect_mean,
                            "treatment_effect_std": treatment_effect_std,
                            "network_type": network_type,
                            "p_edges": p_edges,
                            "neighbour_influence": influence
                        }
                        
                        if save_data:
                            filename = f"sim_{network_structure_type}_{assignment_type}_{network_type}_influence{influence}.pkl"
                            filepath = save_to_pickle(simulation_artifacts, filename)
                            param_summary["filename"] = filepath
                        
                        results.append({"parameters": param_summary, "simulations": simulation_artifacts})
                        
    save_to_pickle(results, "all_simulations.pkl")
    logging.info("Data simulations completed!")
    return results


def process_simulation_results(
        network_configs: List[Tuple[str, Optional[float]]],
        neighbour_influences: List[float],
        dgp_settings: Dict[str, Any],
        feature_adj_sets: Dict[str, List[str]],
        lr_estimator: LinearRegressionEstimator,
        assignment_type: str,
        source_path_file: str = "/Users/polinarevina/Desktop/thesis_new/src/simulation_results"
    ) -> pd.DataFrame:
    """Processes simulation results, computes regression estimates, and generates result visualizations.

    Args:
        network_configs (List[Tuple[str, Optional[float]]]): List of network types and their parameters.
            The first value is the network type (e.g., "barabasi_albert_graph"), and
            the second is an optional network parameter such as edge probability.
        neighbour_influences (List[float]): List of influence values from neighbors in the network.
        dgp_settings (Dict[str, Any]): Dictionary containing simulation settings
        feature_adj_sets (Dict[str, List[str]]): Dictionary mapping feature set names to lists of feature names 
            used for regression analysis
        lr_estimator (LinearRegressionEstimator): Linear regression estimator used to compute treatment effects.
        assignment_type (str): Type of treatment assignment used in simulations (e.g., "random", "clustered").
        source_path_file (str, optional): Path to the directory where simulation results are stored. 
    Returns:
        pd.DataFrame: A DataFrame containing estimated treatment effects, p-values, and other computed results.
    """    

    results = defaultdict(lambda: defaultdict(dict))
    all_results_df = pd.DataFrame()
    
    for network_type, p_edges in tqdm(network_configs):
        for influence in neighbour_influences:
            file_path = f"{source_path_file}/sim_{assignment_type}_{network_type}_influence{influence}.pkl"
            
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            results_dict = defaultdict(lambda: {"pvalue": [], "coef": []})

            for i_sim in range(dgp_settings.n_sim):
                individ_data_sim = data[i_sim]["individ_data"]
                for feature_set, features in feature_adj_sets.items():
                    coef, pvalue = lr_estimator.calculate_results(
                        individ_data=individ_data_sim,
                        additional_feature_names=features
                    )
                    results_dict[feature_set]["pvalue"].append(pvalue)
                    results_dict[feature_set]["coef"].append(coef)

            results[network_type][influence] = dict(results_dict)

            for feature_set, features in feature_adj_sets.items():
                results_plotter = ResultsPlotter(
                    estimated_effect=results_dict[feature_set]["pvalue"],
                    true_effect=dgp_settings.treatment_effect_mean,
                    network_type=network_type,
                    assignment_type=assignment_type,
                    neighbour_influence=influence,
                    treatment_mean=dgp_settings.treatment_effect_mean,
                    additional_feature_names=features
                )
                results_plotter.calculate_results()
                results_df = results_plotter.create_results()
                results_plotter.create_plots()
                all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
    
    return all_results_df
