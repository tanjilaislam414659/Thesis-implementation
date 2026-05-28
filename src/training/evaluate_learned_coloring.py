"""
Evaluate the learned GNN ordering by applying greedy coloring.

This script loads the best trained GNN checkpoint, predicts node scores
for the test graph, converts scores into a learned vertex ordering,
and applies greedy coloring using that ordering.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx
from src.models.gnn_node_scorer import GNNNodeScorer
from src.training.learned_ordering import scores_to_ordering
from src.training.load_pyg_splits import load_all_pyg_graphs, group_dataset_by_split
from src.training.ordered_greedy_coloring import (
    count_colors,
    greedy_color_with_ordering,
    is_valid_coloring,
)


CHECKPOINT_PATH = Path(
    "results/models/gnn_node_scorer/best_gnn_node_scorer.pt"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/learned_coloring_evaluation.csv"
)

TEST_MATRIX_PATHS = {
    "jac_pat": Path("data/raw/matrices/jac_pat.mtx"),
}


def main() -> None:
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

    model = GNNNodeScorer(
        in_channels=checkpoint["input_dim"],
        hidden_channels=checkpoint["hidden_channels"],
        out_channels=checkpoint["out_channels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = load_all_pyg_graphs()
    grouped = group_dataset_by_split(dataset)

    test_graph_data = grouped["test"][0]
    graph_id = test_graph_data.graph_id

    if graph_id not in TEST_MATRIX_PATHS:
        raise ValueError(f"No matrix path configured for test graph: {graph_id}")

    graph = load_graph_from_mtx(TEST_MATRIX_PATHS[graph_id])

    with torch.no_grad():
        predicted_scores = model(test_graph_data.x, test_graph_data.edge_index)

    learned_ordering = scores_to_ordering(predicted_scores)

    learned_coloring = greedy_color_with_ordering(
        graph=graph,
        ordering=learned_ordering,
    )

    num_colors = count_colors(learned_coloring)
    valid = is_valid_coloring(graph, learned_coloring)

    if not valid:
        raise ValueError("Learned coloring is invalid.")

    result_row = {
        "graph_id": graph_id,
        "method_family": "gnn",
        "method_name": "GNNNodeScorer",
        "ordering_name": "learned_ordering_from_predicted_scores",
        "num_vertices": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_colors": num_colors,
        "valid": valid,
        "checkpoint_path": str(CHECKPOINT_PATH),
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(result_row.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result_row)

    print("Learned coloring evaluation")
    print("---------------------------")
    print(f"Graph ID: {graph_id}")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Learned ordering colors: {num_colors}")
    print(f"Valid coloring: {valid}")
    print()
    print(f"Saved evaluation CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()