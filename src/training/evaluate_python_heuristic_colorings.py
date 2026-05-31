"""
Evaluate Python/NetworkX greedy coloring heuristics on the test graph.

This provides an additional baseline next to ColPack and the learned GNN ordering.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import networkx as nx

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx
from src.training.ordered_greedy_coloring import is_valid_coloring


OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/python_heuristic_coloring_jac_pat.csv"
)

TEST_GRAPH_ID = "jac_pat"
TEST_MATRIX_PATH = Path("data/raw/matrices/jac_pat.mtx")

STRATEGIES = [
    "largest_first",
    "smallest_last",
    "random_sequential",
]


def main() -> None:
    graph = load_graph_from_mtx(TEST_MATRIX_PATH)

    rows = []

    print("Python heuristic coloring evaluation")
    print("------------------------------------")
    print(f"Graph ID: {TEST_GRAPH_ID}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print()

    for strategy in STRATEGIES:
        start_time = time.perf_counter()

        coloring = nx.coloring.greedy_color(
            graph,
            strategy=strategy,
        )

        runtime_seconds = time.perf_counter() - start_time

        num_colors = len(set(coloring.values())) if coloring else 0
        valid = is_valid_coloring(graph, coloring)

        rows.append(
            {
                "graph_id": TEST_GRAPH_ID,
                "method_family": "networkx",
                "method_name": "greedy_color",
                "ordering_name": strategy,
                "num_vertices": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "num_colors": num_colors,
                "valid": valid,
                "runtime_seconds": runtime_seconds,
            }
        )

        print(
            f"{strategy}: "
            f"{num_colors} colors | "
            f"valid={valid} | "
            f"runtime={runtime_seconds:.6f}s"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved Python heuristic results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()