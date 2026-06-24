"""
Evaluate Week 15 learned GNN orderings from multiple training-run checkpoints.

For each saved checkpoint, this script:
1. loads the trained model,
2. predicts node scores on each test graph,
3. converts scores into a learned ordering,
4. applies greedy coloring,
5. stores the number of colors and validity.

This Week 15 version evaluates all test graphs from the expanded dataset.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import networkx as nx
import torch

from src.models.gnn_node_scorer import GNNNodeScorer
from src.training.learned_ordering import scores_to_ordering
from src.training.load_pyg_splits import load_all_pyg_graphs, group_dataset_by_split
from src.training.ordered_greedy_coloring import (
    count_colors,
    greedy_color_with_ordering,
    is_valid_coloring,
)


PYG_DATA_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "pyg_data_week16_larger_extension_normalized_features_best_available_of_5"
)


TRAINING_SUMMARY_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week16_larger_extension_training_summary.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week16_larger_extension_learned_coloring_evaluation.csv"
)


def load_training_run_rows() -> list[dict[str, str]]:
    if not TRAINING_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Training summary CSV not found: {TRAINING_SUMMARY_CSV}"
        )

    with TRAINING_SUMMARY_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("No rows found in Week 16 normalized-feature training summary.")

    return rows


def pyg_data_to_networkx_graph(data) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))

    edge_index = data.edge_index.detach().cpu()

    for source, target in edge_index.t().tolist():
        graph.add_edge(int(source), int(target))

    return graph


def evaluate_checkpoint_on_graph(
    checkpoint_path: Path,
    graph_data,
) -> tuple[int, bool, float]:
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model = GNNNodeScorer(
        in_channels=checkpoint["input_dim"],
        hidden_channels=checkpoint["hidden_channels"],
        out_channels=checkpoint["out_channels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    graph = pyg_data_to_networkx_graph(graph_data)

    start_time = time.perf_counter()

    with torch.no_grad():
        predicted_scores = model(graph_data.x, graph_data.edge_index)

    learned_ordering = scores_to_ordering(predicted_scores)

    learned_coloring = greedy_color_with_ordering(
        graph=graph,
        ordering=learned_ordering,
    )

    runtime_seconds = time.perf_counter() - start_time

    num_colors = count_colors(learned_coloring)
    valid = is_valid_coloring(graph, learned_coloring)

    return num_colors, valid, runtime_seconds


def main() -> None:
    training_rows = load_training_run_rows()

    dataset = load_all_pyg_graphs(PYG_DATA_DIR)
    grouped = group_dataset_by_split(dataset)

    test_graphs = grouped["test"]

    if not test_graphs:
        raise ValueError("No test graphs found in expanded Week 16 dataset.")

    rows = []

    print("Week 16 larger-extension best-available-of-5 learned coloring evaluation")
    print("-------------------------------------------")
    print(f"Test graphs: {[data.graph_id for data in test_graphs]}")
    print()

    for graph_data in test_graphs:
        for training_row in training_rows:
            seed = training_row["seed"]
            checkpoint_path = Path(training_row["checkpoint_path"])

            num_colors, valid, runtime_seconds = evaluate_checkpoint_on_graph(
                checkpoint_path=checkpoint_path,
                graph_data=graph_data,
            )

            result_row = {
                "graph_id": graph_data.graph_id,
                "seed": seed,
                "method_family": "gnn",
                "method_name": "GNNNodeScorer",
                "target_ordering": "BEST_AVAILABLE_OF_5_Larger_Extension",
                "ordering_name": "learned_ordering_from_predicted_scores",
                "num_vertices": graph_data.num_nodes,
                "num_edges": graph_data.edge_index.shape[1] // 2,
                "num_colors": num_colors,
                "valid": valid,
                "runtime_seconds": runtime_seconds,
                "checkpoint_path": str(checkpoint_path),
            }

            rows.append(result_row)

            print(
                f"{graph_data.graph_id} | "
                f"seed {seed} | "
                f"colors: {num_colors} | "
                f"valid: {valid} | "
                f"runtime: {runtime_seconds:.6f}s"
            )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved Week 16 larger-extension learned coloring evaluation to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()