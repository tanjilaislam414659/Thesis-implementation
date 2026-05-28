"""
Train the first GNN node-scoring model on SMALLEST_LAST ordering targets.

This is the first supervised training script for the learning-based
graph coloring pipeline.

The model learns:

    graph node features + edge_index
    -> node-level scalar scores

using normalized ColPack SMALLEST_LAST ordering scores as targets.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.models.gnn_node_scorer import GNNNodeScorer
from src.training.load_pyg_splits import load_all_pyg_graphs, group_dataset_by_split


CHECKPOINT_PATH = Path(
    "results/models/gnn_node_scorer/best_gnn_node_scorer.pt"
)


RESULTS_CSV = Path(
    "results/tables/gnn_node_scorer/initial_training_summary.csv"
)


def compute_average_loss(
    model: GNNNodeScorer,
    graphs: list,
    loss_fn: nn.Module,
) -> float:
    """
    Compute average loss over a list of graphs.
    """

    if not graphs:
        raise ValueError("Cannot compute loss on an empty graph list.")

    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for data in graphs:
            predictions = model(data.x, data.edge_index)
            loss = loss_fn(predictions, data.y)
            total_loss += float(loss.item())

    return total_loss / len(graphs)


def train() -> None:
    dataset = load_all_pyg_graphs()
    grouped = group_dataset_by_split(dataset)

    train_graphs = grouped["train"]
    validation_graphs = grouped["validation"]
    test_graphs = grouped["test"]

    if not train_graphs:
        raise ValueError("No training graphs found.")

    input_dim = train_graphs[0].x.shape[1]
    hidden_channels = 32
    out_channels = 1

    model = GNNNodeScorer(
        in_channels=input_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    num_epochs = 200

    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    print("GNN node scorer training")
    print("------------------------")
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(validation_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    print(f"Input feature dimension: {input_dim}")
    print(f"Epochs: {num_epochs}")
    print()

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_train_loss = 0.0

        for data in train_graphs:
            optimizer.zero_grad()

            predictions = model(data.x, data.edge_index)
            loss = loss_fn(predictions, data.y)

            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item())

        avg_train_loss = total_train_loss / len(train_graphs)
        avg_validation_loss = compute_average_loss(
            model=model,
            graphs=validation_graphs,
            loss_fn=loss_fn,
        )

        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            best_epoch = epoch
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {avg_train_loss:.6f} | "
                f"validation loss: {avg_validation_loss:.6f}"
            )

    if best_model_state is None:
        raise RuntimeError("No best model state was recorded during training.")

    model.load_state_dict(best_model_state)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "input_dim": input_dim,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        CHECKPOINT_PATH,
    )

    final_train_loss = compute_average_loss(
        model=model,
        graphs=train_graphs,
        loss_fn=loss_fn,
    )
    final_validation_loss = compute_average_loss(
        model=model,
        graphs=validation_graphs,
        loss_fn=loss_fn,
    )
    final_test_loss = compute_average_loss(
        model=model,
        graphs=test_graphs,
        loss_fn=loss_fn,
    )

    print()
    print("Final losses using best validation model")
    print("----------------------------------------")
    print(f"Train loss: {final_train_loss:.6f}")
    print(f"Validation loss: {final_validation_loss:.6f}")
    print(f"Test loss: {final_test_loss:.6f}")

    print()
    print("Best validation result")
    print("----------------------")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Saved best model checkpoint to: {CHECKPOINT_PATH}")


    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "model_name",
            "target_ordering",
            "num_train_graphs",
            "num_validation_graphs",
            "num_test_graphs",
            "input_dim",
            "hidden_channels",
            "num_epochs",
            "best_epoch",
            "best_validation_loss",
            "final_train_loss_best_model",
            "final_validation_loss_best_model",
            "final_test_loss_best_model",
            "checkpoint_path",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model_name": "GNNNodeScorer",
                "target_ordering": "SMALLEST_LAST",
                "num_train_graphs": len(train_graphs),
                "num_validation_graphs": len(validation_graphs),
                "num_test_graphs": len(test_graphs),
                "input_dim": input_dim,
                "hidden_channels": hidden_channels,
                "num_epochs": num_epochs,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "final_train_loss_best_model": final_train_loss,
                "final_validation_loss_best_model": final_validation_loss,
                "final_test_loss_best_model": final_test_loss,
                "checkpoint_path": str(CHECKPOINT_PATH),
            }
        )

    print(f"Saved training summary to: {RESULTS_CSV}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()