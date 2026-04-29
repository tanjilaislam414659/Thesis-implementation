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


def matrix_to_column_intersection_graph(matrix: coo_matrix) -> nx.Graph:
    """
    Convert a sparse matrix into a column intersection graph.

    Vertices represent matrix columns.
    Two columns are connected if they have nonzero entries in the same row.

    This is useful for rectangular Jacobian sparsity patterns.
    """
    n_cols = matrix.shape[1]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_cols))

    row_to_cols: dict[int, set[int]] = {}

    for row, col in zip(matrix.row, matrix.col):
        row_to_cols.setdefault(int(row), set()).add(int(col))

    edges = set()
    for cols in row_to_cols.values():
        cols_list = sorted(cols)
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                edges.add((cols_list[i], cols_list[j]))

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
        "is_randomized": strategy == "random_sequential",
    }

def run_multiple_strategies(graph: nx.Graph, strategies: list[str]) -> list[Dict[str, Any]]:
    """Run greedy coloring for multiple strategies and return a list of summaries."""
    results = []
    for strategy in strategies:
        summary = greedy_coloring_summary(graph, strategy=strategy)
        results.append(summary)
    return results


def print_strategy_comparison(summaries: list[Dict[str, Any]]) -> None:
    """Print a compact comparison of strategies and color counts."""
    print("Strategy comparison")
    print("-------------------")
    for summary in summaries:
        print(
            f"{summary['strategy']}: "
            f"{summary['num_colors']} colors "
            f"(randomized={summary['is_randomized']})"
        )


def load_graph_from_mtx(path: str | Path) -> nx.Graph:
    """
    Load a Matrix Market file and convert it to a graph.

    Square matrices are converted using the undirected sparsity pattern.
    Rectangular matrices are converted using the column intersection graph.
    """
    matrix = load_matrix_market(path)

    if matrix.shape[0] == matrix.shape[1]:
        return matrix_to_undirected_graph(matrix)

    return matrix_to_column_intersection_graph(matrix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert sparse matrix to graph and run greedy coloring.")
    parser.add_argument("matrix_path", type=str, help="Path to Matrix Market .mtx file")
    parser.add_argument(
    "--strategy",
    type=str,
    nargs="+",
    default=["largest_first"],
    help="One or more greedy coloring strategies",
)
    args = parser.parse_args()

    graph = load_graph_from_mtx(args.matrix_path)
    summaries = run_multiple_strategies(graph, args.strategy)

    for summary in summaries:
        print("Graph summary")
        print("-------------")
        print(f"matrix_path: {args.matrix_path}")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print()


    print_strategy_comparison(summaries)