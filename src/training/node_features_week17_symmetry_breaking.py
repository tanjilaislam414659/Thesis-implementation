"""
Symmetry-breaking node feature extraction utilities for the Week 17 GNN experiment.

This module extends the Week 16 improved structural features with deterministic
position and component-based features.

The motivation is to help the GNN distinguish vertices that are structurally
similar under purely local graph features, especially in regular, grid-like,
cycle-like, and block-structured graphs.
"""

from __future__ import annotations

import math

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


def _component_features(graph: nx.Graph, nodes: list[int]):
    """
    Compute deterministic component-based features.

    Returns
    -------
    component_id_normalized : dict[int, float]
    component_size_normalized : dict[int, float]
    component_position_normalized : dict[int, float]
    """
    components = [sorted(component) for component in nx.connected_components(graph)]
    components = sorted(components, key=lambda comp: (min(comp), len(comp)))

    num_components = len(components)
    num_nodes = max(len(nodes), 1)

    component_id_normalized = {}
    component_size_normalized = {}
    component_position_normalized = {}

    for component_index, component_nodes in enumerate(components):
        component_size = len(component_nodes)

        if num_components > 1:
            comp_id_value = component_index / (num_components - 1)
        else:
            comp_id_value = 0.0

        comp_size_value = component_size / num_nodes

        if component_size > 1:
            for local_position, node in enumerate(component_nodes):
                component_position_normalized[node] = local_position / (
                    component_size - 1
                )
        else:
            node = component_nodes[0]
            component_position_normalized[node] = 0.0

        for node in component_nodes:
            component_id_normalized[node] = comp_id_value
            component_size_normalized[node] = comp_size_value

    return (
        component_id_normalized,
        component_size_normalized,
        component_position_normalized,
    )


def extract_node_features(graph: nx.Graph) -> np.ndarray:
    """
    Extract Week 17 symmetry-breaking node features from a NetworkX graph.

    Feature matrix rows follow the sorted node order.
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

    (
        component_id_normalized,
        component_size_normalized,
        component_position_normalized,
    ) = _component_features(graph, nodes)

    features = []

    for sorted_position, node in enumerate(nodes):
        degree = degrees[node]
        normalized_degree = degree / max_possible_degree
        graph_normalized_degree = degree / max_degree_in_graph

        clustering_coefficient = clustering[node]

        core_number = core_numbers[node]
        normalized_core_number = core_number / max_core_in_graph

        triangle_count = triangles[node]
        normalized_triangle_count = triangle_count / max_triangles_in_graph

        neighbors = list(graph.neighbors(node))
        neighbor_degrees = [float(degrees[neighbor]) for neighbor in neighbors]
        neighbor_cores = [float(core_numbers[neighbor]) for neighbor in neighbors]

        average_neighbor_degree = _safe_mean(neighbor_degrees)
        max_neighbor_degree = _safe_max(neighbor_degrees)
        min_neighbor_degree = _safe_min(neighbor_degrees)
        neighbor_degree_std = _safe_std(neighbor_degrees)

        average_neighbor_core = _safe_mean(neighbor_cores)
        max_neighbor_core = _safe_max(neighbor_cores)

        if degree <= 1:
            local_edge_density = 0.0
        else:
            neighbor_subgraph = graph.subgraph(neighbors)
            possible_neighbor_edges = degree * (degree - 1) / 2
            local_edge_density = (
                neighbor_subgraph.number_of_edges() / possible_neighbor_edges
            )

        # Deterministic symmetry-breaking / positional features.
        if num_nodes > 1:
            node_position_normalized = sorted_position / (num_nodes - 1)
        else:
            node_position_normalized = 0.0

        node_position_sin = math.sin(2.0 * math.pi * node_position_normalized)
        node_position_cos = math.cos(2.0 * math.pi * node_position_normalized)

        node_index_parity = float(int(node) % 2)

        constant_bias = 1.0

        features.append(
            [
                # Week 16 improved structural features
                float(degree),
                float(normalized_degree),
                float(graph_normalized_degree),
                float(clustering_coefficient),
                float(core_number),
                float(normalized_core_number),
                float(triangle_count),
                float(normalized_triangle_count),
                float(average_neighbor_degree),
                float(max_neighbor_degree),
                float(min_neighbor_degree),
                float(neighbor_degree_std),
                float(average_neighbor_core),
                float(max_neighbor_core),
                float(degree_rank[node]),
                float(core_rank[node]),
                float(local_edge_density),

                # Week 17 symmetry-breaking features
                float(node_position_normalized),
                float(node_position_sin),
                float(node_position_cos),
                float(node_index_parity),
                float(component_id_normalized[node]),
                float(component_size_normalized[node]),
                float(component_position_normalized[node]),

                # Bias
                float(constant_bias),
            ]
        )

    return np.asarray(features, dtype=np.float32)


def get_feature_names() -> list[str]:
    """
    Return feature names in the same order as extract_node_features.
    """

    return [
        # Week 16 improved structural features
        "degree",
        "normalized_degree",
        "graph_normalized_degree",
        "clustering_coefficient",
        "core_number",
        "normalized_core_number",
        "triangle_count",
        "normalized_triangle_count",
        "average_neighbor_degree",
        "max_neighbor_degree",
        "min_neighbor_degree",
        "neighbor_degree_std",
        "average_neighbor_core",
        "max_neighbor_core",
        "degree_rank",
        "core_rank",
        "local_edge_density",

        # Week 17 symmetry-breaking features
        "node_position_normalized",
        "node_position_sin",
        "node_position_cos",
        "node_index_parity",
        "component_id_normalized",
        "component_size_normalized",
        "component_position_normalized",

        # Bias
        "constant_bias",
    ]


def validate_feature_matrix(features: np.ndarray, expected_num_nodes: int) -> None:
    """
    Validate basic properties of a node feature matrix.
    """

    if features.ndim != 2:
        raise ValueError(
            f"Feature matrix must be 2-dimensional, got shape {features.shape}."
        )

    if features.shape[0] != expected_num_nodes:
        raise ValueError(
            f"Expected {expected_num_nodes} feature rows, got {features.shape[0]}."
        )

    if features.shape[1] != len(get_feature_names()):
        raise ValueError(
            f"Expected {len(get_feature_names())} feature columns, "
            f"got {features.shape[1]}."
        )

    if np.isnan(features).any():
        raise ValueError("Feature matrix contains NaN values.")

    if np.isinf(features).any():
        raise ValueError("Feature matrix contains infinite values.")