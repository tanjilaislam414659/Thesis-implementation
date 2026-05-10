"""
Node feature extraction utilities for the learning-based graph coloring pipeline.

This module computes simple structural node features for each graph.
These features will later be used as input to a Graph Neural Network.

Initial feature set:
1. degree
2. normalized degree
3. clustering coefficient
4. core number
5. constant bias feature
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def extract_node_features(graph: nx.Graph) -> np.ndarray:
    """
    Extract node-level structural features from a NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.

    Returns
    -------
    np.ndarray
        Feature matrix of shape [num_nodes, num_features].
        Rows follow the sorted node order.
    """

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot extract features from an empty graph.")

    nodes = sorted(graph.nodes())
    num_nodes = graph.number_of_nodes()

    degrees = dict(graph.degree())

    max_possible_degree = max(num_nodes - 1, 1)

    clustering = nx.clustering(graph)

    try:
        core_numbers = nx.core_number(graph)
    except nx.NetworkXError:
        core_numbers = {node: 0 for node in nodes}

    features = []

    for node in nodes:
        degree = degrees[node]
        normalized_degree = degree / max_possible_degree
        clustering_coefficient = clustering[node]
        core_number = core_numbers[node]
        constant_bias = 1.0

        features.append(
            [
                float(degree),
                float(normalized_degree),
                float(clustering_coefficient),
                float(core_number),
                float(constant_bias),
            ]
        )

    return np.asarray(features, dtype=np.float32)


def get_feature_names() -> list[str]:
    """
    Return the names of the node features in the same order as extract_node_features.
    """

    return [
        "degree",
        "normalized_degree",
        "clustering_coefficient",
        "core_number",
        "constant_bias",
    ]


def validate_feature_matrix(features: np.ndarray, expected_num_nodes: int) -> None:
    """
    Validate basic properties of a node feature matrix.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    expected_num_nodes : int
        Expected number of graph nodes.
    """

    if features.ndim != 2:
        raise ValueError(f"Feature matrix must be 2-dimensional, got shape {features.shape}.")

    if features.shape[0] != expected_num_nodes:
        raise ValueError(
            f"Expected {expected_num_nodes} feature rows, got {features.shape[0]}."
        )

    if np.isnan(features).any():
        raise ValueError("Feature matrix contains NaN values.")

    if np.isinf(features).any():
        raise ValueError("Feature matrix contains infinite values.")