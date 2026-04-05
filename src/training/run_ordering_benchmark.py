"""
Benchmark greedy graph coloring under different node ordering strategies.

This script runs a collection of ordering heuristics on one or more graphs,
applies greedy coloring, validates the results, and stores summary metrics.

Author: Tanjila Islam Thesis Project
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import networkx as nx

from src.graphs.ordering_strategies import get_ordering_strategies
from src.utils.graph_utils import (
    compute_graph_stats,
    count_colors,
    greedy_coloring,
    is_valid_coloring,
)


def build_graph_family_benchmarks() -> Dict[str, nx.Graph]:
    """
    Create graph families for Week 3 ordering benchmark experiments.

    Returns
    -------
    Dict[str, nx.Graph]
        Mapping from graph name to NetworkX graph.
    """
    graphs: Dict[str, nx.Graph] = {}

    # Wheel graphs
    graphs["wheel_8"] = nx.wheel_graph(8)
    graphs["wheel_12"] = nx.wheel_graph(12)

    # Ladder graphs
    graphs["ladder_6"] = nx.ladder_graph(6)
    graphs["ladder_10"] = nx.ladder_graph(10)

    # Balanced trees
    graphs["balanced_tree_2_3"] = nx.balanced_tree(2, 3)
    graphs["balanced_tree_2_4"] = nx.balanced_tree(2, 4)

    # Barbell graphs
    graphs["barbell_5_2"] = nx.barbell_graph(5, 2)
    graphs["barbell_6_3"] = nx.barbell_graph(6, 3)

    # Grid graphs
    grid_4_4 = nx.grid_2d_graph(4, 4)
    graphs["grid_4_4"] = nx.convert_node_labels_to_integers(grid_4_4)

    grid_5_5 = nx.grid_2d_graph(5, 5)
    graphs["grid_5_5"] = nx.convert_node_labels_to_integers(grid_5_5)

    return graphs


def benchmark_graph(
    graph_name: str,
    G: nx.Graph,
    seed: int = 42,
) -> List[dict]:
    """
    Benchmark all ordering strategies on a single graph.

    Parameters
    ----------
    graph_name : str
        Name of the graph.
    G : nx.Graph
        Input graph.
    seed : int, default=42
        Random seed used in the random ordering strategy.

    Returns
    -------
    List[dict]
        List of benchmark result rows.
    """
    strategies = get_ordering_strategies(seed=seed)
    graph_stats = compute_graph_stats(G)

    rows: List[dict] = []

    for strategy_name, strategy_fn in strategies.items():
        start_time = time.perf_counter()

        order = strategy_fn(G)
        coloring = greedy_coloring(G, order)
        valid = is_valid_coloring(G, coloring)
        n_colors = count_colors(coloring)

        runtime_sec = time.perf_counter() - start_time

        row = {
            "graph_name": graph_name,
            "strategy": strategy_name,
            "n_nodes": graph_stats["n_nodes"],
            "n_edges": graph_stats["n_edges"],
            "density": graph_stats["density"],
            "min_degree": graph_stats["min_degree"],
            "max_degree": graph_stats["max_degree"],
            "avg_degree": graph_stats["avg_degree"],
            "n_connected_components": graph_stats["n_connected_components"],
            "is_connected": graph_stats["is_connected"],
            "n_colors": n_colors,
            "valid_coloring": valid,
            "runtime_sec": runtime_sec,
            "order": str(order),
            "coloring": str(coloring),
        }
        rows.append(row)

    return rows


def save_results_to_csv(rows: List[dict], output_path: Path) -> None:
    """
    Save benchmark rows to a CSV file.

    Parameters
    ----------
    rows : List[dict]
        Benchmark result rows.
    output_path : Path
        Destination CSV path.
    """
    if not rows:
        raise ValueError("No benchmark rows to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[dict]) -> None:
    """
    Print a readable summary of benchmark results.

    Parameters
    ----------
    rows : List[dict]
        Benchmark result rows.
    """
    print("=" * 90)
    print("Greedy Coloring Benchmark Summary")
    print("=" * 90)

    current_graph = None
    for row in rows:
        if row["graph_name"] != current_graph:
            current_graph = row["graph_name"]
            print(f"\nGraph: {current_graph}")
            print("-" * 90)

        print(
            f"Strategy: {row['strategy']:<18} | "
            f"Colors: {row['n_colors']:<3} | "
            f"Valid: {str(row['valid_coloring']):<5} | "
            f"Runtime (s): {row['runtime_sec']:.6f}"
        )


def main() -> None:
    """
    Run the ordering benchmark on toy graphs and save results.
    """
    graphs = build_graph_family_benchmarks()

    all_rows: List[dict] = []
    for graph_name, G in graphs.items():
        rows = benchmark_graph(graph_name, G, seed=42)
        all_rows.extend(rows)

    output_path = Path("results/tables/ordering_benchmark_graph_families.csv")
    save_results_to_csv(all_rows, output_path)
    print_summary(all_rows)

    print("\nResults saved to:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()