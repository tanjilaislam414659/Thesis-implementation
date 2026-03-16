"""
Node ordering strategies for greedy graph coloring experiments.

This module provides reusable ordering heuristics that can be used with
greedy coloring algorithms. These strategies are intended for baseline
experiments and later comparison with learning-based ordering methods.

Author: Tanjila Islam Thesis Project
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Hashable, List, TypeAlias

import networkx as nx

from src.utils.graph_utils import validate_graph


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Node: TypeAlias = Hashable
OrderingFunction: TypeAlias = Callable[[nx.Graph], List[Node]]


# ---------------------------------------------------------------------------
# Basic ordering strategies
# ---------------------------------------------------------------------------

def natural_order(G: nx.Graph) -> List[Node]:
    """
    Return nodes in the graph's natural iteration order.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    List[Node]
        Node order as provided by NetworkX iteration.
    """
    validate_graph(G)
    return list(G.nodes)


def reverse_order(G: nx.Graph) -> List[Node]:
    """
    Return nodes in reverse of the graph's natural iteration order.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    List[Node]
        Reverse node order.
    """
    validate_graph(G)
    return list(reversed(list(G.nodes)))


def random_order(G: nx.Graph, seed: int = 42) -> List[Node]:
    """
    Return a reproducible random node order.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    List[Node]
        Randomized node order.
    """
    validate_graph(G)
    nodes = list(G.nodes)
    rng = random.Random(seed)
    rng.shuffle(nodes)
    return nodes


# ---------------------------------------------------------------------------
# Degree-based strategies
# ---------------------------------------------------------------------------

def descending_degree_order(G: nx.Graph) -> List[Node]:
    """
    Order nodes by descending degree.

    Nodes with higher degree are processed earlier.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    List[Node]
        Nodes sorted by descending degree.
    """
    validate_graph(G)
    return [
        node
        for node, _ in sorted(
            G.degree(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def ascending_degree_order(G: nx.Graph) -> List[Node]:
    """
    Order nodes by ascending degree.

    Nodes with lower degree are processed earlier.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    List[Node]
        Nodes sorted by ascending degree.
    """
    validate_graph(G)
    return [
        node
        for node, _ in sorted(
            G.degree(),
            key=lambda item: (item[1], item[0]),
        )
    ]


# ---------------------------------------------------------------------------
# Classical coloring-inspired strategy
# ---------------------------------------------------------------------------

def smallest_last_order(G: nx.Graph) -> List[Node]:
    """
    Compute the smallest-last vertex ordering.

    This strategy repeatedly removes a node of minimum current degree,
    records the removal order, and then reverses it.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    List[Node]
        Smallest-last ordering.

    Notes
    -----
    Smallest-last ordering is a classical graph coloring heuristic and often
    performs better than simple static degree-based orderings.
    """
    validate_graph(G)

    working_graph = G.copy()
    elimination_order: List[Node] = []

    while working_graph.number_of_nodes() > 0:
        min_node = min(
            working_graph.degree(),
            key=lambda item: (item[1], item[0]),
        )[0]
        elimination_order.append(min_node)
        working_graph.remove_node(min_node)

    return list(reversed(elimination_order))


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

def get_ordering_strategies(seed: int = 42) -> Dict[str, OrderingFunction]:
    """
    Return a registry of available ordering strategies.

    Parameters
    ----------
    seed : int, default=42
        Seed used for the random ordering strategy.

    Returns
    -------
    Dict[str, OrderingFunction]
        Mapping from strategy name to callable ordering function.
    """
    return {
        "natural": natural_order,
        "reverse": reverse_order,
        "descending_degree": descending_degree_order,
        "ascending_degree": ascending_degree_order,
        "smallest_last": smallest_last_order,
        "random": lambda G: random_order(G, seed=seed),
    }