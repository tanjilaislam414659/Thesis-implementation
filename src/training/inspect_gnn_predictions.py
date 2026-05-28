"""
Inspect predicted node scores from the trained GNN node scorer.

This script loads the best saved checkpoint and prints predicted scores
together with target scores for one selected graph.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.gnn_node_scorer import GNNNodeScorer
from src.training.load_pyg_splits import load_all_pyg_graphs, group_dataset_by_split


CHECKPOINT_PATH = Path(
    "results/models/gnn_node_scorer/best_gnn_node_scorer.pt"
)


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

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
        predictions = model(graph.x, graph.edge_index)

    print("GNN prediction inspection")
    print("-------------------------")
    print(f"Graph ID: {graph.graph_id}")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Prediction shape: {tuple(predictions.shape)}")
    print()

    print("First 15 nodes:")
    print("node_id | predicted_score | target_score")
    print("----------------------------------------")

    for node_id in range(min(15, graph.num_nodes)):
        predicted_score = float(predictions[node_id].item())
        target_score = float(graph.y[node_id].item())

        print(
            f"{node_id:7d} | "
            f"{predicted_score:15.6f} | "
            f"{target_score:12.6f}"
        )


if __name__ == "__main__":
    main()