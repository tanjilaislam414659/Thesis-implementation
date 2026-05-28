"""
Check that the saved GNN node scorer checkpoint can be loaded
and used for prediction.
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

    test_graph = grouped["test"][0]

    with torch.no_grad():
        scores = model(test_graph.x, test_graph.edge_index)

    expected_shape = (test_graph.num_nodes, 1)

    if tuple(scores.shape) != expected_shape:
        raise ValueError(
            f"Expected score shape {expected_shape}, got {tuple(scores.shape)}."
        )

    print("GNN checkpoint loading check")
    print("----------------------------")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Best validation loss: {checkpoint['best_validation_loss']:.6f}")
    print(f"Test graph: {test_graph.graph_id}")
    print(f"Score shape: {tuple(scores.shape)}")
    print("Status: OK")


if __name__ == "__main__":
    main()