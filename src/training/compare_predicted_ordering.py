"""
Compare the GNN-predicted node ordering with the target SMALLEST_LAST ordering.

The GNN produces one scalar score per node.
Higher predicted score means the node is placed earlier in the predicted ordering.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.models.gnn_node_scorer import GNNNodeScorer
from src.training.load_pyg_splits import load_all_pyg_graphs, group_dataset_by_split


CHECKPOINT_PATH = Path(
    "results/models/gnn_node_scorer/best_gnn_node_scorer.pt"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/jac_pat_predicted_ordering_comparison.csv"
)


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

    graph = grouped["test"][0]

    with torch.no_grad():
        predictions = model(graph.x, graph.edge_index).view(-1)

    targets = graph.y.view(-1)

    predicted_order = torch.argsort(predictions, descending=True).tolist()
    target_order = torch.argsort(targets, descending=True).tolist()

    predicted_position_by_node = {
        node_id: position
        for position, node_id in enumerate(predicted_order)
    }

    target_position_by_node = {
        node_id: position
        for position, node_id in enumerate(target_order)
    }

    rows = []

    for node_id in range(graph.num_nodes):
        rows.append(
            {
                "graph_id": graph.graph_id,
                "node_id": node_id,
                "predicted_score": float(predictions[node_id].item()),
                "target_score": float(targets[node_id].item()),
                "predicted_position": predicted_position_by_node[node_id],
                "target_position": target_position_by_node[node_id],
                "absolute_position_error": abs(
                    predicted_position_by_node[node_id]
                    - target_position_by_node[node_id]
                ),
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "graph_id",
            "node_id",
            "predicted_score",
            "target_score",
            "predicted_position",
            "target_position",
            "absolute_position_error",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overlap_top_10 = set(predicted_order[:10]).intersection(set(target_order[:10]))
    overlap_top_15 = set(predicted_order[:15]).intersection(set(target_order[:15]))

    print("Predicted ordering comparison")
    print("-----------------------------")
    print(f"Graph ID: {graph.graph_id}")
    print(f"Number of nodes: {graph.num_nodes}")
    print()

    print("Top 15 nodes by predicted GNN score:")
    print(predicted_order[:15])
    print()

    print("Top 15 nodes by target SMALLEST_LAST score:")
    print(target_order[:15])
    print()

    print("Top-k overlap")
    print("-------------")
    print(f"Top-10 overlap: {len(overlap_top_10)} / 10")
    print(f"Top-15 overlap: {len(overlap_top_15)} / 15")
    print()
    print(f"Saved comparison CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()