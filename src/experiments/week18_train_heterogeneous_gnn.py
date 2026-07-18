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
from src.training.load_pyg_splits import (
    group_dataset_by_split,
    load_all_pyg_graphs,
)
from src.training.ordered_greedy_coloring import (
    count_colors,
    greedy_color_with_ordering,
    is_valid_coloring,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PYG_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "pyg_data_week18_heterogeneous_generalization"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_heterogeneous_gnn_training_summary.csv"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gnn_node_scorer"
    / "week18_heterogeneous_gnn_runs"
)


SEEDS = [0, 1, 2, 3, 4]

EXPECTED_SPLIT_COUNTS = {
    "train": 48,
    "validation": 8,
    "test": 12,
}

EXPECTED_NUM_FEATURES = 25


def set_random_seed(seed: int) -> None:
    """
    Set the random seeds used by Python and PyTorch.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def pyg_data_to_networkx_graph(data) -> nx.Graph:
    """
    Convert a PyG graph into an undirected NetworkX graph.
    """
    graph = nx.Graph()

    graph.add_nodes_from(
        range(int(data.num_nodes))
    )

    edge_index = (
        data.edge_index
        .detach()
        .cpu()
    )

    for source, target in edge_index.t().tolist():
        graph.add_edge(
            int(source),
            int(target),
        )

    return graph


def validate_loaded_dataset(
    dataset: list,
    grouped: dict[str, list],
) -> None:
    """
    Validate the Week 18 dataset before training begins.
    """
    if len(dataset) != 68:
        raise ValueError(
            f"Expected 68 Week 18 graphs, found {len(dataset)}."
        )

    graph_ids: list[str] = []

    for data in dataset:
        graph_id = str(data.graph_id)
        graph_ids.append(graph_id)

        if data.x.ndim != 2:
            raise ValueError(
                f"{graph_id}: expected a two-dimensional "
                f"feature matrix."
            )

        if int(data.x.shape[1]) != EXPECTED_NUM_FEATURES:
            raise ValueError(
                f"{graph_id}: expected "
                f"{EXPECTED_NUM_FEATURES} features, found "
                f"{data.x.shape[1]}."
            )

        if data.y.ndim != 1:
            raise ValueError(
                f"{graph_id}: expected a one-dimensional "
                f"target vector."
            )

        if int(data.y.shape[0]) != int(data.num_nodes):
            raise ValueError(
                f"{graph_id}: target length "
                f"{data.y.shape[0]} does not match "
                f"{data.num_nodes} nodes."
            )

        required_attributes = [
            "split",
            "graph_family",
            "selected_teacher_ordering",
            "selected_num_colors",
            "best_colpack5_colors",
            "worst_colpack5_colors",
            "ordering_gap",
        ]

        missing_attributes = [
            attribute
            for attribute in required_attributes
            if not hasattr(data, attribute)
        ]

        if missing_attributes:
            raise ValueError(
                f"{graph_id}: missing PyG attributes "
                f"{missing_attributes}."
            )

        target_colors = int(
            data.selected_num_colors
        )

        best_colpack5_colors = int(
            data.best_colpack5_colors
        )

        if target_colors != best_colpack5_colors:
            raise ValueError(
                f"{graph_id}: selected teacher uses "
                f"{target_colors} colors, but the best "
                f"ColPack-5 count is {best_colpack5_colors}."
            )

    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError(
            "Duplicate graph IDs found in the loaded PyG dataset."
        )

    for split, expected_count in (
        EXPECTED_SPLIT_COUNTS.items()
    ):
        actual_count = len(
            grouped.get(split, [])
        )

        if actual_count != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} graphs, "
                f"found {actual_count}."
            )


def compute_average_loss(
    model: GNNNodeScorer,
    graphs: list,
    loss_fn: nn.Module,
) -> float:
    """
    Compute mean graph-level MSE over a collection of graphs.
    """
    if not graphs:
        raise ValueError(
            "Cannot compute loss on an empty graph list."
        )

    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for data in graphs:
            predictions = model(
                data.x,
                data.edge_index,
            )

            loss = loss_fn(
                predictions.view(-1),
                data.y.view(-1),
            )

            total_loss += float(
                loss.item()
            )

    return total_loss / len(graphs)


def evaluate_coloring_quality(
    model: GNNNodeScorer,
    graphs: list,
) -> dict[str, object]:
    """
    Convert GNN node scores into vertex orderings and evaluate
    the resulting greedy colorings.
    """
    if not graphs:
        raise ValueError(
            "Cannot evaluate coloring quality on an empty "
            "graph list."
        )

    model.eval()

    total_colors = 0
    total_target_colors = 0
    total_colpack5_colors = 0

    total_gap_from_target = 0
    total_gap_from_colpack5 = 0

    all_valid = True
    per_graph_parts: list[str] = []

    with torch.no_grad():
        for data in graphs:
            graph = pyg_data_to_networkx_graph(
                data
            )

            predicted_scores = model(
                data.x,
                data.edge_index,
            )

            learned_ordering = scores_to_ordering(
                predicted_scores
            )

            coloring = greedy_color_with_ordering(
                graph=graph,
                ordering=learned_ordering,
            )

            num_colors = int(
                count_colors(coloring)
            )

            valid = bool(
                is_valid_coloring(
                    graph,
                    coloring,
                )
            )

            graph_id = str(data.graph_id)
            family = str(data.graph_family)

            teacher = str(
                data.selected_teacher_ordering
            )

            target_colors = int(
                data.selected_num_colors
            )

            best_colpack5_colors = int(
                data.best_colpack5_colors
            )

            ordering_gap = int(
                data.ordering_gap
            )

            gap_from_target = (
                num_colors - target_colors
            )

            gap_from_colpack5 = (
                num_colors
                - best_colpack5_colors
            )

            total_colors += num_colors
            total_target_colors += target_colors
            total_colpack5_colors += (
                best_colpack5_colors
            )

            total_gap_from_target += (
                gap_from_target
            )

            total_gap_from_colpack5 += (
                gap_from_colpack5
            )

            all_valid = (
                all_valid and valid
            )

            per_graph_parts.append(
                f"{graph_id}:"
                f"family={family}/"
                f"teacher={teacher}/"
                f"ordering_gap={ordering_gap}/"
                f"gnn={num_colors}/"
                f"target={target_colors}/"
                f"colpack5={best_colpack5_colors}/"
                f"valid={valid}"
            )

    number_of_graphs = len(graphs)

    return {
        "total_colors": total_colors,
        "average_colors": (
            total_colors / number_of_graphs
        ),
        "total_target_colors": (
            total_target_colors
        ),
        "average_target_colors": (
            total_target_colors
            / number_of_graphs
        ),
        "total_colpack5_colors": (
            total_colpack5_colors
        ),
        "average_colpack5_colors": (
            total_colpack5_colors
            / number_of_graphs
        ),
        "total_gap_from_target": (
            total_gap_from_target
        ),
        "average_gap_from_target": (
            total_gap_from_target
            / number_of_graphs
        ),
        "total_gap_from_colpack5": (
            total_gap_from_colpack5
        ),
        "average_gap_from_colpack5": (
            total_gap_from_colpack5
            / number_of_graphs
        ),
        "colors_saved_vs_colpack5": (
            total_colpack5_colors
            - total_colors
        ),
        "all_valid": all_valid,
        "per_graph_colors": "; ".join(
            per_graph_parts
        ),
    }


def is_better_checkpoint(
    current_validation_total_colors: int,
    current_validation_loss: float,
    best_validation_total_colors: int | None,
    best_validation_loss: float,
) -> bool:
    """
    Prefer fewer validation colors. Use validation loss only
    as the tie-breaker.
    """
    if best_validation_total_colors is None:
        return True

    if (
        current_validation_total_colors
        < best_validation_total_colors
    ):
        return True

    if (
        current_validation_total_colors
        == best_validation_total_colors
    ):
        return (
            current_validation_loss
            < best_validation_loss
        )

    return False


def train_single_run(
    seed: int,
) -> dict[str, object]:
    """
    Train and evaluate one deterministic model seed.
    """
    set_random_seed(seed)

    dataset = load_all_pyg_graphs(
        PYG_DATA_DIR
    )

    grouped = group_dataset_by_split(
        dataset
    )

    validate_loaded_dataset(
        dataset=dataset,
        grouped=grouped,
    )

    train_graphs = grouped["train"]
    validation_graphs = grouped["validation"]
    test_graphs = grouped["test"]

    input_dim = int(
        train_graphs[0].x.shape[1]
    )

    hidden_channels = 32
    out_channels = 1
    num_epochs = 500
    learning_rate = 0.01

    model = GNNNodeScorer(
        in_channels=input_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )

    loss_fn = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_epoch = 0
    best_model_state = None

    best_validation_total_colors: int | None = None
    best_validation_loss = float("inf")
    best_validation_coloring = None

    for epoch in range(
        1,
        num_epochs + 1,
    ):
        model.train()

        for data in train_graphs:
            optimizer.zero_grad()

            predictions = model(
                data.x,
                data.edge_index,
            )

            loss = loss_fn(
                predictions.view(-1),
                data.y.view(-1),
            )

            loss.backward()
            optimizer.step()

        validation_loss = compute_average_loss(
            model=model,
            graphs=validation_graphs,
            loss_fn=loss_fn,
        )

        validation_coloring = (
            evaluate_coloring_quality(
                model=model,
                graphs=validation_graphs,
            )
        )

        validation_total_colors = int(
            validation_coloring[
                "total_colors"
            ]
        )

        if is_better_checkpoint(
            current_validation_total_colors=(
                validation_total_colors
            ),
            current_validation_loss=(
                validation_loss
            ),
            best_validation_total_colors=(
                best_validation_total_colors
            ),
            best_validation_loss=(
                best_validation_loss
            ),
        ):
            best_epoch = epoch

            best_validation_total_colors = (
                validation_total_colors
            )

            best_validation_loss = (
                validation_loss
            )

            best_validation_coloring = (
                validation_coloring
            )

            best_model_state = {
                key: (
                    value
                    .detach()
                    .cpu()
                    .clone()
                )
                for key, value in (
                    model.state_dict().items()
                )
            }

        if (
            epoch == 1
            or epoch % 50 == 0
        ):
            print(
                f"Seed {seed} | "
                f"Epoch {epoch:03d} | "
                f"validation loss: "
                f"{validation_loss:.6f} | "
                f"validation colors: "
                f"{validation_total_colors} | "
                f"validation target: "
                f"{validation_coloring['total_target_colors']} | "
                f"validation gap: "
                f"{validation_coloring['total_gap_from_target']} | "
                f"colors saved vs ColPack-5: "
                f"{validation_coloring['colors_saved_vs_colpack5']} | "
                f"valid: "
                f"{validation_coloring['all_valid']}"
            )

    if best_model_state is None:
        raise RuntimeError(
            f"Seed {seed}: no best model state was recorded."
        )

    model.load_state_dict(
        best_model_state
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

    train_coloring = evaluate_coloring_quality(
        model=model,
        graphs=train_graphs,
    )

    validation_coloring = (
        evaluate_coloring_quality(
            model=model,
            graphs=validation_graphs,
        )
    )

    test_coloring = evaluate_coloring_quality(
        model=model,
        graphs=test_graphs,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / (
            "week18_best_gnn_node_scorer_"
            f"seed_{seed}.pt"
        )
    )

    torch.save(
        {
            "model_state_dict": (
                best_model_state
            ),
            "input_dim": input_dim,
            "hidden_channels": (
                hidden_channels
            ),
            "out_channels": out_channels,
            "learning_rate": (
                learning_rate
            ),
            "num_epochs": num_epochs,
            "seed": seed,
            "best_epoch": best_epoch,
            "selection_metric": (
                "validation_total_colors_"
                "then_validation_loss"
            ),
            "best_validation_total_colors": (
                best_validation_total_colors
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "best_validation_coloring": (
                best_validation_coloring
            ),
            "experiment": (
                "WEEK18_HETEROGENEOUS_"
                "GENERALIZATION"
            ),
            "target_ordering": (
                "BALANCED_BEST_OF_5_"
                "COLPACK_TEACHER"
            ),
            "feature_set": (
                "WEEK18_SYMMETRY_"
                "BREAKING_25"
            ),
        },
        checkpoint_path,
    )

    return {
        "seed": seed,
        "model_name": "GNNNodeScorer",
        "experiment": (
            "WEEK18_HETEROGENEOUS_GENERALIZATION"
        ),
        "target_ordering": (
            "BALANCED_BEST_OF_5_COLPACK_TEACHER"
        ),
        "feature_set": (
            "WEEK18_SYMMETRY_BREAKING_25"
        ),
        "checkpoint_selection": (
            "validation_total_colors_"
            "then_validation_loss"
        ),
        "num_train_graphs": len(
            train_graphs
        ),
        "num_validation_graphs": len(
            validation_graphs
        ),
        "num_test_graphs": len(
            test_graphs
        ),
        "input_dim": input_dim,
        "hidden_channels": (
            hidden_channels
        ),
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "best_epoch": best_epoch,
        "best_validation_total_colors": (
            best_validation_total_colors
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "final_train_loss_best_model": (
            final_train_loss
        ),
        "final_validation_loss_best_model": (
            final_validation_loss
        ),
        "final_test_loss_best_model": (
            final_test_loss
        ),
        "final_train_total_colors": (
            train_coloring["total_colors"]
        ),
        "final_train_target_colors": (
            train_coloring[
                "total_target_colors"
            ]
        ),
        "final_train_colpack5_colors": (
            train_coloring[
                "total_colpack5_colors"
            ]
        ),
        "final_train_gap_from_target": (
            train_coloring[
                "total_gap_from_target"
            ]
        ),
        "final_train_gap_from_colpack5": (
            train_coloring[
                "total_gap_from_colpack5"
            ]
        ),
        "final_train_colors_saved_vs_colpack5": (
            train_coloring[
                "colors_saved_vs_colpack5"
            ]
        ),
        "final_train_all_valid": (
            train_coloring["all_valid"]
        ),
        "final_validation_total_colors": (
            validation_coloring[
                "total_colors"
            ]
        ),
        "final_validation_target_colors": (
            validation_coloring[
                "total_target_colors"
            ]
        ),
        "final_validation_colpack5_colors": (
            validation_coloring[
                "total_colpack5_colors"
            ]
        ),
        "final_validation_gap_from_target": (
            validation_coloring[
                "total_gap_from_target"
            ]
        ),
        "final_validation_gap_from_colpack5": (
            validation_coloring[
                "total_gap_from_colpack5"
            ]
        ),
        "final_validation_colors_saved_vs_colpack5": (
            validation_coloring[
                "colors_saved_vs_colpack5"
            ]
        ),
        "final_validation_all_valid": (
            validation_coloring[
                "all_valid"
            ]
        ),
        "final_test_total_colors": (
            test_coloring["total_colors"]
        ),
        "final_test_target_colors": (
            test_coloring[
                "total_target_colors"
            ]
        ),
        "final_test_colpack5_colors": (
            test_coloring[
                "total_colpack5_colors"
            ]
        ),
        "final_test_gap_from_target": (
            test_coloring[
                "total_gap_from_target"
            ]
        ),
        "final_test_gap_from_colpack5": (
            test_coloring[
                "total_gap_from_colpack5"
            ]
        ),
        "final_test_colors_saved_vs_colpack5": (
            test_coloring[
                "colors_saved_vs_colpack5"
            ]
        ),
        "final_test_all_valid": (
            test_coloring["all_valid"]
        ),
        "final_test_per_graph_colors": (
            test_coloring[
                "per_graph_colors"
            ]
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }


def main() -> None:
    if not PYG_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Week 18 PyG directory not found: "
            f"{PYG_DATA_DIR}"
        )

    print(
        "Week 18 heterogeneous GNN training"
    )
    print(
        "----------------------------------"
    )
    print(
        f"PyG data directory: "
        f"{PYG_DATA_DIR}"
    )
    print(f"Seeds: {SEEDS}")
    print(
        "Architecture: existing "
        "GNNNodeScorer"
    )
    print(
        "Training: 500 epochs, "
        "hidden channels 32, "
        "Adam learning rate 0.01"
    )
    print()

    rows: list[
        dict[str, object]
    ] = []

    for seed in SEEDS:
        print(f"Training seed {seed}")
        print("----------------")

        row = train_single_run(
            seed
        )

        rows.append(row)

        print(
            f"Seed {seed} selected epoch "
            f"{row['best_epoch']} | "
            f"validation colors: "
            f"{row['final_validation_total_colors']} | "
            f"validation target: "
            f"{row['final_validation_target_colors']} | "
            f"validation gap: "
            f"{row['final_validation_gap_from_target']} | "
            f"test colors: "
            f"{row['final_test_total_colors']} | "
            f"test target: "
            f"{row['final_test_target_colors']} | "
            f"test gap: "
            f"{row['final_test_gap_from_target']} | "
            f"test valid: "
            f"{row['final_test_all_valid']}"
        )
        print()

    if not rows:
        raise RuntimeError(
            "No training rows were produced."
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = list(
            rows[0].keys()
        )

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "Week 18 heterogeneous GNN "
        "training completed."
    )
    print(
        "--------------------------------"
    )
    print(
        f"Completed seeds: "
        f"{len(rows)}"
    )
    print(
        f"Saved checkpoints to: "
        f"{CHECKPOINT_DIR}"
    )
    print(
        f"Saved training summary to: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()