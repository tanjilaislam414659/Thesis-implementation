"""
Utility functions for graph coloring experiments.

This module provides reusable utilities for:
- validating node orderings,
- running greedy graph coloring,
- checking coloring validity,
- counting used colors,
- computing graph statistics.

These functions are designed to support:
- baseline heuristic experiments,
- ColPack comparison pipelines,
- later GNN-based ordering experiments.

Author: Tanjila Islam Thesis Project
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Mapping, Sequence, TypeAlias

import networkx as nx


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A graph node in NetworkX can be any hashable Python object.
Node: TypeAlias = Hashable

# A coloring maps each node to an integer color ID.
Coloring: TypeAlias = Dict[Node, int]

# Graph statistics dictionary.
GraphStats: TypeAlias = Dict[str, Any]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GraphColoringError(Exception):
    """Raised when graph coloring inputs or outputs are invalid."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_graph(G: nx.Graph) -> None:
    """
    Validate that the input is a NetworkX graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Raises
    ------
    TypeError
        If G is not a NetworkX graph.
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX graph.")


def normalize_node_order(G: nx.Graph, order: Sequence[Node]) -> List[Node]:
    """
    Validate and normalize a node ordering.

    A valid node order must contain each graph node exactly once.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    order : Sequence[Node]
        Proposed node processing order.

    Returns
    -------
    List[Node]
        Validated node order as a list.

    Raises
    ------
    GraphColoringError
        If the order is invalid.
    TypeError
        If G is not a NetworkX graph.
    """
    validate_graph(G)

    order_list = list(order)
    graph_nodes = list(G.nodes)

    if len(order_list) != len(graph_nodes):
        raise GraphColoringError(
            "Invalid node order: order length does not match number of graph nodes."
        )

    if len(set(order_list)) != len(order_list):
        raise GraphColoringError(
            "Invalid node order: duplicate nodes detected in the ordering."
        )

    graph_node_set = set(graph_nodes)
    order_node_set = set(order_list)

    unknown_nodes = order_node_set - graph_node_set
    missing_nodes = graph_node_set - order_node_set

    if unknown_nodes:
        raise GraphColoringError(
            f"Invalid node order: unknown nodes found: {sorted(map(str, unknown_nodes))}"
        )

    if missing_nodes:
        raise GraphColoringError(
            f"Invalid node order: missing nodes found: {sorted(map(str, missing_nodes))}"
        )

    return order_list


def validate_coloring_keys(G: nx.Graph, coloring: Mapping[Node, int]) -> None:
    """
    Validate that a coloring contains exactly the graph nodes.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    coloring : Mapping[Node, int]
        Node-to-color mapping.

    Raises
    ------
    GraphColoringError
        If the coloring keys do not match graph nodes exactly.
    """
    validate_graph(G)

    graph_nodes = set(G.nodes)
    coloring_nodes = set(coloring.keys())

    missing_nodes = graph_nodes - coloring_nodes
    unknown_nodes = coloring_nodes - graph_nodes

    if missing_nodes:
        raise GraphColoringError(
            f"Coloring is incomplete: missing nodes {sorted(map(str, missing_nodes))}"
        )

    if unknown_nodes:
        raise GraphColoringError(
            f"Coloring contains unknown nodes: {sorted(map(str, unknown_nodes))}"
        )


def validate_coloring_values(coloring: Mapping[Node, int]) -> None:
    """
    Validate that coloring values are non-negative integers.

    Parameters
    ----------
    coloring : Mapping[Node, int]
        Node-to-color mapping.

    Raises
    ------
    GraphColoringError
        If any color is not a non-negative integer.
    """
    for node, color in coloring.items():
        if not isinstance(color, int):
            raise GraphColoringError(
                f"Invalid color for node {node!r}: color must be an integer."
            )
        if color < 0:
            raise GraphColoringError(
                f"Invalid color for node {node!r}: color must be non-negative."
            )


# ---------------------------------------------------------------------------
# Core graph coloring functions
# ---------------------------------------------------------------------------

def greedy_coloring(G: nx.Graph, order: Sequence[Node]) -> Coloring:
    """
    Perform greedy graph coloring using a specified node order.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    order : Sequence[Node]
        Node processing order.

    Returns
    -------
    Coloring
        Dictionary mapping each node to its assigned color.

    Notes
    -----
    This implementation assigns to each node the smallest non-negative
    integer color that is not currently used by its already-colored neighbors.

    The resulting number of colors depends strongly on the given node order.
    """
    order_list = normalize_node_order(G, order)

    coloring: Coloring = {}

    for node in order_list:
        used_neighbor_colors = {
            coloring[neighbor]
            for neighbor in G.neighbors(node)
            if neighbor in coloring
        }

        color = 0
        while color in used_neighbor_colors:
            color += 1

        coloring[node] = color

    return coloring


def is_valid_coloring(G: nx.Graph, coloring: Mapping[Node, int]) -> bool:
    """
    Check whether a coloring is valid for the given graph.

    A coloring is valid if:
    - every graph node is colored,
    - all colors are non-negative integers,
    - no adjacent nodes share the same color.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    coloring : Mapping[Node, int]
        Node-to-color mapping.

    Returns
    -------
    bool
        True if the coloring is valid, otherwise False.
    """
    try:
        validate_coloring_keys(G, coloring)
        validate_coloring_values(coloring)
    except GraphColoringError:
        return False

    for u, v in G.edges:
        if coloring[u] == coloring[v]:
            return False

    return True


def assert_valid_coloring(G: nx.Graph, coloring: Mapping[Node, int]) -> None:
    """
    Assert that a coloring is valid.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    coloring : Mapping[Node, int]
        Node-to-color mapping.

    Raises
    ------
    GraphColoringError
        If the coloring is invalid.
    """
    validate_coloring_keys(G, coloring)
    validate_coloring_values(coloring)

    for u, v in G.edges:
        if coloring[u] == coloring[v]:
            raise GraphColoringError(
                f"Invalid coloring: adjacent nodes {u!r} and {v!r} share color {coloring[u]}."
            )


def count_colors(coloring: Mapping[Node, int]) -> int:
    """
    Count the number of distinct colors used in a coloring.

    Parameters
    ----------
    coloring : Mapping[Node, int]
        Node-to-color mapping.

    Returns
    -------
    int
        Number of distinct colors used.

    Raises
    ------
    GraphColoringError
        If the coloring is empty or contains invalid color values.
    """
    if not coloring:
        raise GraphColoringError("Cannot count colors: coloring is empty.")

    validate_coloring_values(coloring)
    return len(set(coloring.values()))


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def compute_graph_stats(G: nx.Graph) -> GraphStats:
    """
    Compute basic graph statistics.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    GraphStats
        Dictionary containing core graph statistics.
    """
    validate_graph(G)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    degrees = [degree for _, degree in G.degree()]
    avg_degree = sum(degrees) / n_nodes if n_nodes > 0 else 0.0
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    density = nx.density(G) if n_nodes > 1 else 0.0
    n_components = nx.number_connected_components(G) if n_nodes > 0 else 0

    stats: GraphStats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "avg_degree": avg_degree,
        "n_connected_components": n_components,
        "is_connected": n_components == 1 if n_nodes > 0 else False,
    }

    return stats


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def run_greedy_coloring_with_validation(
    G: nx.Graph,
    order: Sequence[Node],
) -> Coloring:
    """
    Run greedy coloring and verify correctness before returning the result.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    order : Sequence[Node]
        Node processing order.

    Returns
    -------
    Coloring
        Valid greedy coloring.

    Raises
    ------
    GraphColoringError
        If the resulting coloring is invalid.
    """
    coloring = greedy_coloring(G, order)
    assert_valid_coloring(G, coloring)
    return coloring