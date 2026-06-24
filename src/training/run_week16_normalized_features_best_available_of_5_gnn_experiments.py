"""
Run repeated GNN node scorer training experiments with different random seeds.

This script is used for Week 12 evaluation. It trains the same GNN model
multiple times and saves one summary row per run.

The goal is to check whether the learned ordering result is stable across
different random initializations.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.models.gnn_node_scorer import GNNNodeScorer
from src.training.load_pyg_splits import load_all_pyg_graphs, group_dataset_by_split


OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week16_normalized_features_best_available_of_5_training_summary.csv"
)

CHECKPOINT_DIR = Path(
    "results/models/gnn_node_scorer/week16_normalized_features_best_available_of_5_runs"
)


SEEDS = [0, 1, 2, 3, 4]


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training runs.
    """

    random.seed(seed)
    torch.manual_seed(seed)


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
            loss = loss_fn(predictions.view(-1), data.y.view(-1))
            total_loss += float(loss.item())

    return total_loss / len(graphs)


def train_single_run(seed: int) -> dict[str, object]:
    """
    Train one GNN model using one random seed.
    """

    set_random_seed(seed)

    dataset = load_all_pyg_graphs(
    "data/processed/initial_graph_coloring_dataset/"
    "pyg_data_week16_normalized_features_best_available_of_5"
)

    grouped = group_dataset_by_split(dataset)

    train_graphs = grouped["train"]
    validation_graphs = grouped["validation"]
    test_graphs = grouped["test"]

    input_dim = train_graphs[0].x.shape[1]
    hidden_channels = 32
    out_channels = 1
    num_epochs = 200

    model = GNNNodeScorer(
        in_channels=input_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()

        for data in train_graphs:
            optimizer.zero_grad()

            predictions = model(data.x, data.edge_index)
            loss = loss_fn(predictions.view(-1), data.y.view(-1))

            loss.backward()
            optimizer.step()

        validation_loss = compute_average_loss(
            model=model,
            graphs=validation_graphs,
            loss_fn=loss_fn,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_model_state is None:
        raise RuntimeError(f"Seed {seed}: no best model state recorded.")

    model.load_state_dict(best_model_state)

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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"best_gnn_node_scorer_seed_{seed}.pt"

    torch.save(
        {
            "model_state_dict": best_model_state,
            "input_dim": input_dim,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        checkpoint_path,
    )

    return {
        "seed": seed,
        "model_name": "GNNNodeScorer",
        "target_ordering": "BEST_AVAILABLE_OF_5_NORMALIZED_FEATURES",
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
        "checkpoint_path": str(checkpoint_path),
    }


def main() -> None:
    print("Week 16 normalized-feature best-available-of-5 GNN node scorer experiment")
    print("---------------------------------------")
    print(f"Seeds: {SEEDS}")
    print()

    rows = []

    for seed in SEEDS:
        row = train_single_run(seed)
        rows.append(row)

        print(
            f"Seed {seed} | "
            f"best epoch: {row['best_epoch']} | "
            f"validation loss: {row['best_validation_loss']:.6f} | "
            f"test loss: {row['final_test_loss_best_model']:.6f}"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved multiple-run training summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()