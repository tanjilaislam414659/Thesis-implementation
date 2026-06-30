from __future__ import annotations

import csv
import random
from pathlib import Path

import networkx as nx
import torch
from torch import nn
from torch.optim import Adam

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
    "pyg_data_week17_best_available_of_5_symmetry_breaking_features"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_symmetry_breaking_validation_color_selection_training_summary.csv"
)

CHECKPOINT_DIR = Path(
    "results/models/gnn_node_scorer/"
    "week17_symmetry_breaking_validation_color_selection_runs"
)

SEEDS = [0, 1, 2, 3, 4]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pyg_data_to_networkx_graph(data) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))

    edge_index = data.edge_index.detach().cpu()

    for source, target in edge_index.t().tolist():
        graph.add_edge(int(source), int(target))

    return graph


def compute_average_loss(
    model: GNNNodeScorer,
    graphs: list,
    loss_fn: nn.Module,
) -> float:
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


def evaluate_coloring_quality(
    model: GNNNodeScorer,
    graphs: list,
) -> dict[str, object]:
    if not graphs:
        raise ValueError("Cannot evaluate coloring quality on an empty graph list.")

    model.eval()

    total_colors = 0
    total_target_colors = 0
    total_gap_from_target = 0
    all_valid = True
    per_graph_parts = []

    with torch.no_grad():
        for data in graphs:
            graph = pyg_data_to_networkx_graph(data)

            predicted_scores = model(data.x, data.edge_index)
            learned_ordering = scores_to_ordering(predicted_scores)

            coloring = greedy_color_with_ordering(
                graph=graph,
                ordering=learned_ordering,
            )

            num_colors = count_colors(coloring)
            valid = is_valid_coloring(graph, coloring)

            target_colors = int(data.selected_num_colors)
            gap_from_target = num_colors - target_colors

            total_colors += int(num_colors)
            total_target_colors += int(target_colors)
            total_gap_from_target += int(gap_from_target)
            all_valid = all_valid and bool(valid)

            per_graph_parts.append(
                f"{data.graph_id}:{num_colors}/{target_colors}"
            )

    return {
        "total_colors": total_colors,
        "average_colors": total_colors / len(graphs),
        "total_target_colors": total_target_colors,
        "total_gap_from_target": total_gap_from_target,
        "average_gap_from_target": total_gap_from_target / len(graphs),
        "all_valid": all_valid,
        "per_graph_colors": "; ".join(per_graph_parts),
    }


def is_better_checkpoint(
    current_validation_total_colors: int,
    current_validation_loss: float,
    best_validation_total_colors: int | None,
    best_validation_loss: float,
) -> bool:
    if best_validation_total_colors is None:
        return True

    if current_validation_total_colors < best_validation_total_colors:
        return True

    if current_validation_total_colors == best_validation_total_colors:
        return current_validation_loss < best_validation_loss

    return False


def train_single_run(seed: int) -> dict[str, object]:
    set_random_seed(seed)

    dataset = load_all_pyg_graphs(PYG_DATA_DIR)
    grouped = group_dataset_by_split(dataset)

    train_graphs = grouped["train"]
    validation_graphs = grouped["validation"]
    test_graphs = grouped["test"]

    if not train_graphs:
        raise ValueError("No training graphs found.")

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

    best_epoch = 0
    best_model_state = None

    best_validation_total_colors = None
    best_validation_loss = float("inf")
    best_validation_average_colors = None
    best_validation_total_gap = None
    best_validation_average_gap = None
    best_validation_all_valid = None
    best_validation_per_graph_colors = None

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

        validation_coloring = evaluate_coloring_quality(
            model=model,
            graphs=validation_graphs,
        )

        validation_total_colors = int(validation_coloring["total_colors"])

        if is_better_checkpoint(
            current_validation_total_colors=validation_total_colors,
            current_validation_loss=validation_loss,
            best_validation_total_colors=best_validation_total_colors,
            best_validation_loss=best_validation_loss,
        ):
            best_epoch = epoch
            best_validation_total_colors = validation_total_colors
            best_validation_loss = validation_loss
            best_validation_average_colors = validation_coloring["average_colors"]
            best_validation_total_gap = validation_coloring["total_gap_from_target"]
            best_validation_average_gap = validation_coloring["average_gap_from_target"]
            best_validation_all_valid = validation_coloring["all_valid"]
            best_validation_per_graph_colors = validation_coloring["per_graph_colors"]

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Seed {seed} | Epoch {epoch:03d} | "
                f"validation loss: {validation_loss:.6f} | "
                f"validation colors: {validation_total_colors} | "
                f"validation gap: {validation_coloring['total_gap_from_target']}"
            )

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

    train_coloring = evaluate_coloring_quality(model, train_graphs)
    validation_coloring = evaluate_coloring_quality(model, validation_graphs)
    test_coloring = evaluate_coloring_quality(model, test_graphs)

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
            "selection_metric": "validation_total_colors_then_validation_loss",
            "best_validation_total_colors": best_validation_total_colors,
            "best_validation_average_colors": best_validation_average_colors,
            "best_validation_total_gap_from_target": best_validation_total_gap,
            "best_validation_average_gap_from_target": best_validation_average_gap,
            "best_validation_loss": best_validation_loss,
            "best_validation_all_valid": best_validation_all_valid,
            "best_validation_per_graph_colors": best_validation_per_graph_colors,
        },
        checkpoint_path,
    )

    return {
        "seed": seed,
        "model_name": "GNNNodeScorer",
        "target_ordering": "WEEK17_BEST_AVAILABLE_OF_5",
        "feature_set": "WEEK17_SYMMETRY_BREAKING_25",
        "checkpoint_selection": "validation_total_colors_then_validation_loss",
        "num_train_graphs": len(train_graphs),
        "num_validation_graphs": len(validation_graphs),
        "num_test_graphs": len(test_graphs),
        "input_dim": input_dim,
        "hidden_channels": hidden_channels,
        "num_epochs": num_epochs,
        "best_epoch": best_epoch,
        "best_validation_total_colors": best_validation_total_colors,
        "best_validation_average_colors": best_validation_average_colors,
        "best_validation_total_gap_from_target": best_validation_total_gap,
        "best_validation_average_gap_from_target": best_validation_average_gap,
        "best_validation_loss": best_validation_loss,
        "best_validation_all_valid": best_validation_all_valid,
        "best_validation_per_graph_colors": best_validation_per_graph_colors,
        "final_train_loss_best_model": final_train_loss,
        "final_validation_loss_best_model": final_validation_loss,
        "final_test_loss_best_model": final_test_loss,
        "final_train_total_colors": train_coloring["total_colors"],
        "final_train_total_gap_from_target": train_coloring["total_gap_from_target"],
        "final_validation_total_colors": validation_coloring["total_colors"],
        "final_validation_total_gap_from_target": validation_coloring[
            "total_gap_from_target"
        ],
        "final_test_total_colors": test_coloring["total_colors"],
        "final_test_total_gap_from_target": test_coloring["total_gap_from_target"],
        "final_test_per_graph_colors": test_coloring["per_graph_colors"],
        "checkpoint_path": str(checkpoint_path),
    }


def main() -> None:
    print("Week 17 symmetry-breaking GNN training with validation color-count selection")
    print("--------------------------------------------------------------------------")
    print(f"PyG data directory: {PYG_DATA_DIR}")
    print(f"Seeds: {SEEDS}")
    print()

    rows = []

    for seed in SEEDS:
        print(f"Training seed {seed}")
        print("----------------")

        row = train_single_run(seed)
        rows.append(row)

        print(
            f"Seed {seed} selected epoch {row['best_epoch']} | "
            f"validation colors: {row['best_validation_total_colors']} | "
            f"validation gap: {row['best_validation_total_gap_from_target']} | "
            f"test colors: {row['final_test_total_colors']} | "
            f"test gap: {row['final_test_total_gap_from_target']}"
        )
        print()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved Week 17 training summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()