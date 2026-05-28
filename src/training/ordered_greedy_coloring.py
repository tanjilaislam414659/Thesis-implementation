"""
Greedy graph coloring using a fixed vertex ordering.

This utility will be used to evaluate learned GNN orderings.
"""

from __future__ import annotations

import networkx as nx


def greedy_color_with_ordering(
    graph: nx.Graph,
    ordering: list[int],
) -> dict[int, int]:
    """
    Color a graph greedily following a given vertex ordering.

    Parameters
    ----------
    graph:
        NetworkX graph.
    ordering:
        List of node IDs in the order they should be colored.

    Returns
    -------
    dict[int, int]
        Mapping from node ID to color ID.
    """

    graph_nodes = set(graph.nodes())
    ordering_nodes = set(ordering)

    if graph_nodes != ordering_nodes:
        raise ValueError(
            "Ordering must contain exactly the same nodes as the graph."
        )

    coloring: dict[int, int] = {}

    for node in ordering:
        neighbor_colors = {
            coloring[neighbor]
            for neighbor in graph.neighbors(node)
            if neighbor in coloring
        }

        color = 0
        while color in neighbor_colors:
            color += 1

        coloring[node] = color

    return coloring


def count_colors(coloring: dict[int, int]) -> int:
    """
    Count the number of colors used in a coloring.
    """

    if not coloring:
        return 0

    return len(set(coloring.values()))


def is_valid_coloring(
    graph: nx.Graph,
    coloring: dict[int, int],
) -> bool:
    """
    Check whether a coloring is valid for a graph.
    """

    if set(graph.nodes()) != set(coloring.keys()):
        return False

    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            return False

    return True


def main() -> None:
    graph = nx.path_graph(4)
    ordering = [0, 1, 2, 3]

    coloring = greedy_color_with_ordering(graph, ordering)

    print("Ordered greedy coloring utility check")
    print("-------------------------------------")
    print(f"Ordering: {ordering}")
    print(f"Coloring: {coloring}")
    print(f"Number of colors: {count_colors(coloring)}")
    print(f"Valid coloring: {is_valid_coloring(graph, coloring)}")

    if not is_valid_coloring(graph, coloring):
        raise ValueError("Coloring should be valid.")

    print("Status: OK")


if __name__ == "__main__":
    main()