from __future__ import annotations

import csv
import random
from pathlib import Path

import networkx as nx
import pandas as pd
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

ABLATION_ASSIGNMENT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week18_training_size_ablation_assignment.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_training_size_ablation_gnn_training_summary.csv"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gnn_node_scorer"
    / "week18_training_size_ablation_gnn_runs"
)


TRAINING_SIZES = [12, 24, 36, 48]
SEEDS = [0, 1, 2, 3, 4]

EXPECTED_VALIDATION_GRAPHS = 8
EXPECTED_TEST_GRAPHS = 12

HIDDEN_CHANNELS = 32
OUT_CHANNELS = 1
NUM_EPOCHS = 500
LEARNING_RATE = 0.01


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pyg_data_to_networkx_graph(data) -> nx.Graph:
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


def load_experiment_data() -> tuple[
    dict[str, object],
    list,
    list,
    pd.DataFrame,
]:
    if not PYG_DATA_DIR.exists():
        raise FileNotFoundError(
            f"PyG directory not found: {PYG_DATA_DIR}"
        )

    if not ABLATION_ASSIGNMENT_CSV.exists():
        raise FileNotFoundError(
            "Ablation assignment CSV not found: "
            f"{ABLATION_ASSIGNMENT_CSV}"
        )

    dataset = load_all_pyg_graphs(
        PYG_DATA_DIR
    )

    grouped = group_dataset_by_split(
        dataset
    )

    all_train_graphs = sorted(
        grouped.get("train", []),
        key=lambda data: str(data.graph_id),
    )

    validation_graphs = sorted(
        grouped.get("validation", []),
        key=lambda data: str(data.graph_id),
    )

    test_graphs = sorted(
        grouped.get("test", []),
        key=lambda data: str(data.graph_id),
    )

    if len(all_train_graphs) != 48:
        raise ValueError(
            f"Expected 48 training graphs, "
            f"found {len(all_train_graphs)}."
        )

    if len(validation_graphs) != EXPECTED_VALIDATION_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_GRAPHS} "
            f"validation graphs, found "
            f"{len(validation_graphs)}."
        )

    if len(test_graphs) != EXPECTED_TEST_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_GRAPHS} test graphs, "
            f"found {len(test_graphs)}."
        )

    train_graph_by_id = {
        str(data.graph_id): data
        for data in all_train_graphs
    }

    if len(train_graph_by_id) != 48:
        raise ValueError(
            "Duplicate graph IDs found in the PyG "
            "training dataset."
        )

    assignment_df = pd.read_csv(
        ABLATION_ASSIGNMENT_CSV
    )

    required_columns = {
        "graph_id",
        "minimum_training_size",
        "family",
        "selected_teacher_ordering",
        "split_group_id",
    }

    missing_columns = (
        required_columns
        - set(assignment_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Ablation assignment is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(assignment_df) != 48:
        raise ValueError(
            f"Expected 48 assignment rows, "
            f"found {len(assignment_df)}."
        )

    if assignment_df["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in the "
            "ablation assignment."
        )

    assignment_graph_ids = set(
        assignment_df["graph_id"].astype(str)
    )

    pyg_train_graph_ids = set(
        train_graph_by_id
    )

    if assignment_graph_ids != pyg_train_graph_ids:
        raise ValueError(
            "Training graph IDs differ between the PyG "
            "dataset and ablation assignment.\n"
            f"Only in assignment: "
            f"{sorted(assignment_graph_ids - pyg_train_graph_ids)}\n"
            f"Only in PyG data: "
            f"{sorted(pyg_train_graph_ids - assignment_graph_ids)}"
        )

    return (
        train_graph_by_id,
        validation_graphs,
        test_graphs,
        assignment_df,
    )


def build_training_subset(
    training_size: int,
    train_graph_by_id: dict[str, object],
    assignment_df: pd.DataFrame,
) -> list:
    selected_rows = assignment_df[
        assignment_df["minimum_training_size"]
        <= training_size
    ].copy()

    if len(selected_rows) != training_size:
        raise ValueError(
            f"Training size {training_size}: expected "
            f"{training_size} graphs, found "
            f"{len(selected_rows)}."
        )

    selected_graph_ids = sorted(
        selected_rows["graph_id"]
        .astype(str)
        .tolist()
    )

    train_graphs = [
        train_graph_by_id[graph_id]
        for graph_id in selected_graph_ids
    ]

    teacher_counts = (
        selected_rows["selected_teacher_ordering"]
        .value_counts()
    )

    expected_per_teacher = (
        training_size // 4
    )

    teachers = [
        "LARGEST_FIRST",
        "DYNAMIC_LARGEST_FIRST",
        "INCIDENCE_DEGREE",
        "SMALLEST_LAST",
    ]

    for teacher in teachers:
        actual_count = int(
            teacher_counts.get(teacher, 0)
        )

        if actual_count != expected_per_teacher:
            raise ValueError(
                f"Training size {training_size}, "
                f"{teacher}: expected "
                f"{expected_per_teacher}, found "
                f"{actual_count}."
            )

    return train_graphs


def compute_average_loss(
    model: GNNNodeScorer,
    graphs: list,
    loss_fn: nn.Module,
) -> float:
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
    if not graphs:
        raise ValueError(
            "Cannot evaluate an empty graph list."
        )

    model.eval()

    total_colors = 0
    total_target_colors = 0
    total_gap = 0

    exact_matches = 0
    better_than_target = 0
    worse_than_target = 0

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

            gnn_colors = int(
                count_colors(coloring)
            )

            valid = bool(
                is_valid_coloring(
                    graph,
                    coloring,
                )
            )

            target_colors = int(
                data.best_colpack5_colors
            )

            gap = (
                gnn_colors
                - target_colors
            )

            total_colors += gnn_colors
            total_target_colors += target_colors
            total_gap += gap

            exact_matches += int(
                gap == 0
            )

            better_than_target += int(
                gap < 0
            )

            worse_than_target += int(
                gap > 0
            )

            all_valid = (
                all_valid and valid
            )

            per_graph_parts.append(
                f"{data.graph_id}:"
                f"family={data.graph_family}/"
                f"gnn={gnn_colors}/"
                f"target={target_colors}/"
                f"gap={gap}/"
                f"valid={valid}"
            )

    return {
        "total_colors": total_colors,
        "total_target_colors": total_target_colors,
        "total_gap_from_target": total_gap,
        "average_gap_per_graph": (
            total_gap / len(graphs)
        ),
        "exact_matches": exact_matches,
        "better_than_target_count": (
            better_than_target
        ),
        "worse_than_target_count": (
            worse_than_target
        ),
        "all_valid": all_valid,
        "per_graph_results": "; ".join(
            per_graph_parts
        ),
    }


def is_better_checkpoint(
    current_validation_colors: int,
    current_validation_loss: float,
    best_validation_colors: int | None,
    best_validation_loss: float,
) -> bool:
    if best_validation_colors is None:
        return True

    if (
        current_validation_colors
        < best_validation_colors
    ):
        return True

    if (
        current_validation_colors
        == best_validation_colors
    ):
        return (
            current_validation_loss
            < best_validation_loss
        )

    return False


def train_single_run(
    training_size: int,
    seed: int,
    train_graphs: list,
    validation_graphs: list,
    test_graphs: list,
) -> dict[str, object]:
    set_random_seed(seed)

    input_dim = int(
        train_graphs[0].x.shape[1]
    )

    model = GNNNodeScorer(
        in_channels=input_dim,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
    )

    loss_fn = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_epoch = 0
    best_model_state = None

    best_validation_total_colors: int | None = None
    best_validation_loss = float("inf")

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
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

        validation_result = (
            evaluate_coloring_quality(
                model=model,
                graphs=validation_graphs,
            )
        )

        validation_colors = int(
            validation_result[
                "total_colors"
            ]
        )

        if is_better_checkpoint(
            current_validation_colors=(
                validation_colors
            ),
            current_validation_loss=(
                validation_loss
            ),
            best_validation_colors=(
                best_validation_total_colors
            ),
            best_validation_loss=(
                best_validation_loss
            ),
        ):
            best_epoch = epoch
            best_validation_total_colors = (
                validation_colors
            )
            best_validation_loss = (
                validation_loss
            )

            best_model_state = {
                key: (
                    value
                    .detach()
                    .cpu()
                    .clone()
                )
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 100 == 0
        ):
            print(
                f"Train {training_size:02d} | "
                f"Seed {seed} | "
                f"Epoch {epoch:03d} | "
                f"validation colors="
                f"{validation_colors} | "
                f"target="
                f"{validation_result['total_target_colors']} | "
                f"gap="
                f"{validation_result['total_gap_from_target']} | "
                f"loss={validation_loss:.6f}"
            )

    if best_model_state is None:
        raise RuntimeError(
            f"Training size {training_size}, seed {seed}: "
            "no checkpoint was recorded."
        )

    model.load_state_dict(
        best_model_state
    )

    train_loss = compute_average_loss(
        model=model,
        graphs=train_graphs,
        loss_fn=loss_fn,
    )

    validation_loss = compute_average_loss(
        model=model,
        graphs=validation_graphs,
        loss_fn=loss_fn,
    )

    test_loss = compute_average_loss(
        model=model,
        graphs=test_graphs,
        loss_fn=loss_fn,
    )

    train_result = evaluate_coloring_quality(
        model=model,
        graphs=train_graphs,
    )

    validation_result = (
        evaluate_coloring_quality(
            model=model,
            graphs=validation_graphs,
        )
    )

    test_result = evaluate_coloring_quality(
        model=model,
        graphs=test_graphs,
    )

    run_checkpoint_dir = (
        CHECKPOINT_DIR
        / f"train_{training_size}"
    )

    run_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        run_checkpoint_dir
        / (
            "week18_ablation_"
            f"train_{training_size}_"
            f"seed_{seed}.pt"
        )
    )

    torch.save(
        {
            "model_state_dict": best_model_state,
            "training_size": training_size,
            "seed": seed,
            "input_dim": input_dim,
            "hidden_channels": HIDDEN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "best_epoch": best_epoch,
            "best_validation_total_colors": (
                best_validation_total_colors
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "selection_metric": (
                "validation_total_colors_"
                "then_validation_loss"
            ),
        },
        checkpoint_path,
    )

    return {
        "training_size": training_size,
        "seed": seed,
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
        "hidden_channels": HIDDEN_CHANNELS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_total_colors": (
            best_validation_total_colors
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "final_train_loss": train_loss,
        "final_validation_loss": (
            validation_loss
        ),
        "final_test_loss": test_loss,
        "final_train_total_colors": (
            train_result["total_colors"]
        ),
        "final_train_target_colors": (
            train_result[
                "total_target_colors"
            ]
        ),
        "final_train_gap_from_target": (
            train_result[
                "total_gap_from_target"
            ]
        ),
        "final_validation_total_colors": (
            validation_result[
                "total_colors"
            ]
        ),
        "final_validation_target_colors": (
            validation_result[
                "total_target_colors"
            ]
        ),
        "final_validation_gap_from_target": (
            validation_result[
                "total_gap_from_target"
            ]
        ),
        "final_validation_exact_matches": (
            validation_result[
                "exact_matches"
            ]
        ),
        "final_test_total_colors": (
            test_result["total_colors"]
        ),
        "final_test_target_colors": (
            test_result[
                "total_target_colors"
            ]
        ),
        "final_test_gap_from_target": (
            test_result[
                "total_gap_from_target"
            ]
        ),
        "final_test_average_gap_per_graph": (
            test_result[
                "average_gap_per_graph"
            ]
        ),
        "final_test_exact_matches": (
            test_result[
                "exact_matches"
            ]
        ),
        "final_test_better_than_target_count": (
            test_result[
                "better_than_target_count"
            ]
        ),
        "final_test_worse_than_target_count": (
            test_result[
                "worse_than_target_count"
            ]
        ),
        "final_test_all_valid": (
            test_result["all_valid"]
        ),
        "final_test_per_graph_results": (
            test_result[
                "per_graph_results"
            ]
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }


def print_ablation_summary(
    results_df: pd.DataFrame,
) -> None:
    aggregate_df = (
        results_df
        .groupby(
            "training_size",
            as_index=False,
        )
        .agg(
            mean_test_colors=(
                "final_test_total_colors",
                "mean",
            ),
            std_test_colors=(
                "final_test_total_colors",
                "std",
            ),
            minimum_test_colors=(
                "final_test_total_colors",
                "min",
            ),
            maximum_test_colors=(
                "final_test_total_colors",
                "max",
            ),
            mean_test_gap=(
                "final_test_gap_from_target",
                "mean",
            ),
            mean_exact_matches=(
                "final_test_exact_matches",
                "mean",
            ),
            all_colorings_valid=(
                "final_test_all_valid",
                "all",
            ),
        )
    )

    representative_rows: list[
        dict[str, object]
    ] = []

    for training_size in TRAINING_SIZES:
        size_rows = (
            results_df[
                results_df["training_size"]
                == training_size
            ]
            .sort_values(
                [
                    "best_validation_total_colors",
                    "best_validation_loss",
                    "seed",
                ]
            )
            .reset_index(drop=True)
        )

        representative = (
            size_rows.iloc[0]
        )

        representative_rows.append(
            {
                "training_size": training_size,
                "representative_seed": int(
                    representative["seed"]
                ),
                "representative_validation_colors": int(
                    representative[
                        "final_validation_total_colors"
                    ]
                ),
                "representative_test_colors": int(
                    representative[
                        "final_test_total_colors"
                    ]
                ),
                "representative_test_gap": int(
                    representative[
                        "final_test_gap_from_target"
                    ]
                ),
                "representative_exact_matches": int(
                    representative[
                        "final_test_exact_matches"
                    ]
                ),
            }
        )

    representative_df = pd.DataFrame(
        representative_rows
    )

    display_df = aggregate_df.merge(
        representative_df,
        on="training_size",
        how="inner",
        validate="one_to_one",
    )

    print()
    print("Training-size ablation summary")
    print("------------------------------")
    print(
        display_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )


def main() -> None:
    (
        train_graph_by_id,
        validation_graphs,
        test_graphs,
        assignment_df,
    ) = load_experiment_data()

    print(
        "Week 18 training-size ablation"
    )
    print(
        "------------------------------"
    )
    print(
        f"Training sizes: {TRAINING_SIZES}"
    )
    print(f"Seeds: {SEEDS}")
    print(
        "Training configuration: "
        f"{NUM_EPOCHS} epochs, "
        f"{HIDDEN_CHANNELS} hidden channels, "
        f"learning rate {LEARNING_RATE}"
    )
    print()

    result_rows: list[
        dict[str, object]
    ] = []

    for training_size in TRAINING_SIZES:
        train_graphs = build_training_subset(
            training_size=training_size,
            train_graph_by_id=(
                train_graph_by_id
            ),
            assignment_df=assignment_df,
        )

        print()
        print(
            f"Training-size condition: "
            f"{training_size} graphs"
        )
        print(
            "--------------------------------"
        )

        for seed in SEEDS:
            print(
                f"Starting training size "
                f"{training_size}, seed {seed}"
            )

            result = train_single_run(
                training_size=training_size,
                seed=seed,
                train_graphs=train_graphs,
                validation_graphs=(
                    validation_graphs
                ),
                test_graphs=test_graphs,
            )

            result_rows.append(
                result
            )

            print(
                f"Completed train={training_size}, "
                f"seed={seed} | "
                f"best epoch={result['best_epoch']} | "
                f"validation colors="
                f"{result['final_validation_total_colors']} | "
                f"test colors="
                f"{result['final_test_total_colors']} | "
                f"test gap="
                f"{result['final_test_gap_from_target']} | "
                f"exact matches="
                f"{result['final_test_exact_matches']} | "
                f"valid="
                f"{result['final_test_all_valid']}"
            )
            print()

    if len(result_rows) != (
        len(TRAINING_SIZES)
        * len(SEEDS)
    ):
        raise RuntimeError(
            "Unexpected number of ablation runs."
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
            result_rows[0].keys()
        )

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            result_rows
        )

    results_df = pd.DataFrame(
        result_rows
    )

    print_ablation_summary(
        results_df
    )

    print()
    print(
        "Week 18 training-size ablation completed."
    )
    print(
        "----------------------------------------"
    )
    print(
        f"Completed runs: {len(result_rows)}"
    )
    print(
        f"Saved checkpoints to: "
        f"{CHECKPOINT_DIR}"
    )
    print(
        f"Saved summary to: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()