from pathlib import Path
from typing import Dict, Any

import networkx as nx
from scipy.io import mmread
from scipy.sparse import coo_matrix, issparse


def load_matrix_market(path: str | Path):
    """Load a sparse matrix from a Matrix Market file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    matrix = mmread(str(path))
    if not issparse(matrix):
        raise ValueError("Loaded object is not a sparse matrix.")
    return matrix.tocoo()


def matrix_to_undirected_graph(matrix: coo_matrix) -> nx.Graph:
    """Convert a square sparse matrix into an undirected graph from its sparsity pattern."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Only square matrices are supported in version 1.")

    n = matrix.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    rows = matrix.row
    cols = matrix.col

    edges = set()
    for i, j in zip(rows, cols):
        if i == j:
            continue
        u, v = sorted((int(i), int(j)))
        edges.add((u, v))

    graph.add_edges_from(edges)
    return graph


def greedy_coloring_summary(graph: nx.Graph, strategy: str = "largest_first") -> Dict[str, Any]:
    """Run greedy coloring and return summary stats."""
    coloring = nx.coloring.greedy_color(graph, strategy=strategy)
    num_colors = len(set(coloring.values())) if coloring else 0

    return {
    "num_nodes": graph.number_of_nodes(),
    "num_edges": graph.number_of_edges(),
    "density": nx.density(graph),
    "max_degree": max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 0,
    "num_colors": num_colors,
    "strategy": strategy,
}


def load_graph_from_mtx(path: str | Path) -> nx.Graph:
    matrix = load_matrix_market(path)
    return matrix_to_undirected_graph(matrix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert sparse matrix to graph and run greedy coloring.")
    parser.add_argument("matrix_path", type=str, help="Path to Matrix Market .mtx file")
    parser.add_argument("--strategy", type=str, default="largest_first", help="Greedy coloring strategy")
    args = parser.parse_args()

    graph = load_graph_from_mtx(args.matrix_path)
    summary = greedy_coloring_summary(graph, strategy=args.strategy)

    print("Graph summary")
    print("-------------")
    print(f"matrix_path: {args.matrix_path}")
    for key, value in summary.items():
        print(f"{key}: {value}")