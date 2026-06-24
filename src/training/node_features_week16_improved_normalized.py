"""
Normalized improved node feature extraction utilities for the Week 16 GNN experiment.

This version keeps the coloring-relevant structural features, but avoids large raw
feature scales. Most features are normalized to [0, 1] within each graph.

The motivation is to make training more stable than the first improved-feature
version, where raw degree, triangle count, and neighbor statistics could dominate
the input representation.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _safe_max(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(values))


def _safe_min(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.min(values))


def _safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(values))


def _safe_divide(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(value / denominator)


def _rank_normalized(values: dict[int, float], nodes: list[int]) -> dict[int, float]:
    """
    Compute normalized ranks in [0, 1].

    Larger feature values receive larger ranks.
    Ties are handled deterministically using sorted node order.
    """
    if len(nodes) <= 1:
        return {node: 0.0 for node in nodes}

    sorted_nodes = sorted(nodes, key=lambda node: (values[node], node))

    ranks = {}
    for rank, node in enumerate(sorted_nodes):
        ranks[node] = rank / (len(nodes) - 1)

    return ranks


def extract_node_features(graph: nx.Graph) -> np.ndarray:
    """
    Extract normalized improved node-level structural features from a NetworkX graph.

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

    max_degree_in_graph = max(degrees.values()) if degrees else 1
    max_degree_in_graph = max(max_degree_in_graph, 1)

    clustering = nx.clustering(graph)

    try:
        core_numbers = nx.core_number(graph)
    except nx.NetworkXError:
        core_numbers = {node: 0 for node in nodes}

    max_core_in_graph = max(core_numbers.values()) if core_numbers else 1
    max_core_in_graph = max(max_core_in_graph, 1)

    triangles = nx.triangles(graph)
    max_triangles_in_graph = max(triangles.values()) if triangles else 1
    max_triangles_in_graph = max(max_triangles_in_graph, 1)

    degree_rank = _rank_normalized(
        {node: float(degrees[node]) for node in nodes},
        nodes,
    )

    core_rank = _rank_normalized(
        {node: float(core_numbers[node]) for node in nodes},
        nodes,
    )

    features = []

    for node in nodes:
        degree = degrees[node]

        normalized_degree = _safe_divide(degree, max_possible_degree)
        graph_normalized_degree = _safe_divide(degree, max_degree_in_graph)

        clustering_coefficient = clustering[node]

        core_number = core_numbers[node]
        normalized_core_number = _safe_divide(core_number, max_core_in_graph)

        triangle_count = triangles[node]
        normalized_triangle_count = _safe_divide(
            triangle_count,
            max_triangles_in_graph,
        )

        neighbors = list(graph.neighbors(node))

        neighbor_degrees = [float(degrees[neighbor]) for neighbor in neighbors]
        neighbor_cores = [float(core_numbers[neighbor]) for neighbor in neighbors]

        average_neighbor_degree = _safe_mean(neighbor_degrees)
        max_neighbor_degree = _safe_max(neighbor_degrees)
        min_neighbor_degree = _safe_min(neighbor_degrees)
        neighbor_degree_std = _safe_std(neighbor_degrees)

        average_neighbor_core = _safe_mean(neighbor_cores)
        max_neighbor_core = _safe_max(neighbor_cores)

        average_neighbor_degree_normalized = _safe_divide(
            average_neighbor_degree,
            max_degree_in_graph,
        )
        max_neighbor_degree_normalized = _safe_divide(
            max_neighbor_degree,
            max_degree_in_graph,
        )
        min_neighbor_degree_normalized = _safe_divide(
            min_neighbor_degree,
            max_degree_in_graph,
        )
        neighbor_degree_std_normalized = _safe_divide(
            neighbor_degree_std,
            max_degree_in_graph,
        )

        average_neighbor_core_normalized = _safe_divide(
            average_neighbor_core,
            max_core_in_graph,
        )
        max_neighbor_core_normalized = _safe_divide(
            max_neighbor_core,
            max_core_in_graph,
        )

        if degree <= 1:
            local_edge_density = 0.0
        else:
            neighbor_subgraph = graph.subgraph(neighbors)
            possible_neighbor_edges = degree * (degree - 1) / 2
            local_edge_density = _safe_divide(
                neighbor_subgraph.number_of_edges(),
                possible_neighbor_edges,
            )

        constant_bias = 1.0

        features.append(
            [
                float(normalized_degree),
                float(graph_normalized_degree),
                float(clustering_coefficient),
                float(normalized_core_number),
                float(normalized_triangle_count),
                float(average_neighbor_degree_normalized),
                float(max_neighbor_degree_normalized),
                float(min_neighbor_degree_normalized),
                float(neighbor_degree_std_normalized),
                float(average_neighbor_core_normalized),
                float(max_neighbor_core_normalized),
                float(degree_rank[node]),
                float(core_rank[node]),
                float(local_edge_density),
                float(constant_bias),
            ]
        )

    return np.asarray(features, dtype=np.float32)


def get_feature_names() -> list[str]:
    """
    Return the names of the node features in the same order as extract_node_features.
    """

    return [
        "normalized_degree",
        "graph_normalized_degree",
        "clustering_coefficient",
        "normalized_core_number",
        "normalized_triangle_count",
        "average_neighbor_degree_normalized",
        "max_neighbor_degree_normalized",
        "min_neighbor_degree_normalized",
        "neighbor_degree_std_normalized",
        "average_neighbor_core_normalized",
        "max_neighbor_core_normalized",
        "degree_rank",
        "core_rank",
        "local_edge_density",
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