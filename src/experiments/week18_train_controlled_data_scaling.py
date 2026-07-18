from __future__ import annotations

import argparse
import random
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from src.models.gnn_node_scorer import GNNNodeScorer
from src.training.learned_ordering import scores_to_ordering
from src.training.ordered_greedy_coloring import (
    count_colors,
    greedy_color_with_ordering,
    is_valid_coloring,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week18_controlled_data_scaling_manifest.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_gnn_training_summary.csv"
)

AGGREGATE_OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_gnn_aggregate.csv"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_gnn_runs"
)

CONDITIONS = [
    "train_20_baseline",
    "train_32_plus12",
    "train_44_plus24",
    "train_125_plus105",
]

EXPECTED_TRAIN_COUNTS = {
    "train_20_baseline": 20,
    "train_32_plus12": 32,
    "train_44_plus24": 44,
    "train_125_plus105": 125,
}

SEEDS = [0, 1, 2, 3, 4]

HIDDEN_CHANNELS = 32
OUT_CHANNELS = 1
NUM_EPOCHS = 500
LEARNING_RATE = 0.01

EXPECTED_FEATURE_COUNT = 25
EXPECTED_VALIDATION_GRAPHS = 5
EXPECTED_TEST_GRAPHS = 5
EXPECTED_EVALUATION_TARGET_TOTAL = 60
EXPECTED_EVALUATION_COLPACK_TOTAL = 75


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the controlled-data scaling experiment."
        )
    )

    parser.add_argument(
        "--condition",
        choices=CONDITIONS + ["all"],
        default="all",
        help=(
            "Training condition to run. "
            "The default runs all four conditions."
        ),
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore existing summary rows and rerun the "
            "requested condition or conditions."
        ),
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_torch_data(path: Path):
    try:
        return torch.load(
            path,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path)


def scalar_to_int(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(
            value.detach().cpu().item()
        )

    return int(value)


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest_df = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "condition",
        "graph_id",
        "split",
        "source_dataset",
        "source_pt_path",
        "is_added_mixed_graph",
        "gap_level",
        "num_nodes",
        "target_colors",
        "best_colpack5_colors",
    }

    missing_columns = (
        required_columns
        - set(manifest_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    actual_conditions = set(
        manifest_df["condition"]
    )

    if actual_conditions != set(CONDITIONS):
        raise ValueError(
            "Manifest conditions differ from the "
            "expected conditions.\n"
            f"Expected: {CONDITIONS}\n"
            f"Found: {sorted(actual_conditions)}"
        )

    duplicate_mask = manifest_df[
        [
            "condition",
            "graph_id",
        ]
    ].duplicated()

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate graph entries found within "
            "a training condition."
        )

    for condition in CONDITIONS:
        condition_df = manifest_df[
            manifest_df["condition"]
            == condition
        ]

        split_counts = (
            condition_df["split"]
            .value_counts()
            .to_dict()
        )

        train_count = int(
            split_counts.get("train", 0)
        )

        validation_count = int(
            split_counts.get(
                "validation",
                0,
            )
        )

        test_count = int(
            split_counts.get("test", 0)
        )

        if train_count != EXPECTED_TRAIN_COUNTS[condition]:
            raise ValueError(
                f"{condition}: expected "
                f"{EXPECTED_TRAIN_COUNTS[condition]} "
                f"training graphs, found {train_count}."
            )

        if validation_count != EXPECTED_VALIDATION_GRAPHS:
            raise ValueError(
                f"{condition}: expected "
                f"{EXPECTED_VALIDATION_GRAPHS} validation "
                f"graphs, found {validation_count}."
            )

        if test_count != EXPECTED_TEST_GRAPHS:
            raise ValueError(
                f"{condition}: expected "
                f"{EXPECTED_TEST_GRAPHS} test graphs, "
                f"found {test_count}."
            )

    validate_frozen_evaluation_sets(
        manifest_df
    )

    return manifest_df


def validate_frozen_evaluation_sets(
    manifest_df: pd.DataFrame,
) -> None:
    reference_validation_ids: set[str] | None = None
    reference_test_ids: set[str] | None = None

    for condition in CONDITIONS:
        condition_df = manifest_df[
            manifest_df["condition"]
            == condition
        ]

        validation_ids = set(
            condition_df[
                condition_df["split"]
                == "validation"
            ]["graph_id"].astype(str)
        )

        test_ids = set(
            condition_df[
                condition_df["split"]
                == "test"
            ]["graph_id"].astype(str)
        )

        if reference_validation_ids is None:
            reference_validation_ids = (
                validation_ids
            )

            reference_test_ids = (
                test_ids
            )

        if (
            validation_ids
            != reference_validation_ids
        ):
            raise ValueError(
                f"{condition}: validation graph IDs "
                "are not frozen."
            )

        if test_ids != reference_test_ids:
            raise ValueError(
                f"{condition}: test graph IDs "
                "are not frozen."
            )


def load_condition_dataset(
    manifest_df: pd.DataFrame,
    condition: str,
) -> tuple[
    list,
    list,
    list,
    pd.DataFrame,
]:
    condition_df = manifest_df[
        manifest_df["condition"]
        == condition
    ].copy()

    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    condition_df[
        "_split_order"
    ] = condition_df[
        "split"
    ].map(split_order)

    condition_df = condition_df.sort_values(
        [
            "_split_order",
            "is_added_mixed_graph",
            "graph_id",
        ]
    ).reset_index(drop=True)

    grouped_graphs = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for row in condition_df.itertuples(
        index=False
    ):
        source_path = (
            PROJECT_ROOT
            / str(row.source_pt_path)
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"PyG file not found: {source_path}"
            )

        data = load_torch_data(
            source_path
        )

        if str(data.graph_id) != str(
            row.graph_id
        ):
            raise ValueError(
                f"Graph ID mismatch for {source_path}: "
                f"manifest={row.graph_id}, "
                f"data={data.graph_id}"
            )

        data.split = str(row.split)
        data.source_dataset = str(
            row.source_dataset
        )

        if data.x.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"{data.graph_id}: expected "
                f"{EXPECTED_FEATURE_COUNT} features, "
                f"found {data.x.shape[1]}."
            )

        if data.y.shape[0] != data.num_nodes:
            raise ValueError(
                f"{data.graph_id}: target count does "
                "not match number of nodes."
            )

        if scalar_to_int(
            data.selected_num_colors
        ) != int(row.target_colors):
            raise ValueError(
                f"{data.graph_id}: target-color "
                "mismatch between manifest and data."
            )

        if scalar_to_int(
            data.best_colpack5_colors
        ) != int(
            row.best_colpack5_colors
        ):
            raise ValueError(
                f"{data.graph_id}: ColPack-color "
                "mismatch between manifest and data."
            )

        grouped_graphs[
            str(row.split)
        ].append(data)

    train_graphs = grouped_graphs["train"]
    validation_graphs = grouped_graphs[
        "validation"
    ]
    test_graphs = grouped_graphs["test"]

    if len(train_graphs) != (
        EXPECTED_TRAIN_COUNTS[
            condition
        ]
    ):
        raise ValueError(
            f"{condition}: incorrect training count."
        )

    if len(validation_graphs) != (
        EXPECTED_VALIDATION_GRAPHS
    ):
        raise ValueError(
            f"{condition}: incorrect validation count."
        )

    if len(test_graphs) != (
        EXPECTED_TEST_GRAPHS
    ):
        raise ValueError(
            f"{condition}: incorrect test count."
        )

    return (
        train_graphs,
        validation_graphs,
        test_graphs,
        condition_df,
    )


def pyg_data_to_networkx_graph(
    data,
) -> nx.Graph:
    graph = nx.Graph()

    graph.add_nodes_from(
        range(data.num_nodes)
    )

    edge_index = (
        data.edge_index
        .detach()
        .cpu()
    )

    for source, target in (
        edge_index.t().tolist()
    ):
        graph.add_edge(
            int(source),
            int(target),
        )

    return graph


def compute_average_loss(
    model: GNNNodeScorer,
    graphs: list,
    loss_fn: nn.Module,
) -> float:
    if not graphs:
        raise ValueError(
            "Cannot compute loss on an empty "
            "graph list."
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

    return (
        total_loss
        / len(graphs)
    )


def graph_description(data) -> str:
    gap_level = scalar_to_int(
        data.gap_level
    )

    if hasattr(
        data,
        "base_cycle_size",
    ):
        base_cycle_size = scalar_to_int(
            data.base_cycle_size
        )

        structure = (
            f"base{base_cycle_size}"
        )

    elif hasattr(
        data,
        "component_cycle_sizes",
    ):
        component_sizes = str(
            data.component_cycle_sizes
        )

        structure = (
            f"mixed[{component_sizes}]"
        )

    else:
        structure = "structure_unknown"

    return (
        f"{structure}/gap{gap_level}"
    )


def evaluate_coloring_quality(
    model: GNNNodeScorer,
    graphs: list,
) -> dict[str, object]:
    if not graphs:
        raise ValueError(
            "Cannot evaluate coloring quality "
            "on an empty graph list."
        )

    model.eval()

    total_colors = 0
    total_target_colors = 0
    total_colpack5_colors = 0

    total_gap_from_target = 0
    total_gap_from_colpack5 = 0

    exact_target_graphs = 0
    better_than_colpack_graphs = 0
    tied_with_colpack_graphs = 0
    worse_than_colpack_graphs = 0

    all_valid = True
    per_graph_parts: list[str] = []

    with torch.no_grad():
        for data in graphs:
            graph = (
                pyg_data_to_networkx_graph(
                    data
                )
            )

            predicted_scores = model(
                data.x,
                data.edge_index,
            )

            learned_ordering = (
                scores_to_ordering(
                    predicted_scores
                )
            )

            coloring = (
                greedy_color_with_ordering(
                    graph=graph,
                    ordering=learned_ordering,
                )
            )

            num_colors = int(
                count_colors(
                    coloring
                )
            )

            valid = bool(
                is_valid_coloring(
                    graph,
                    coloring,
                )
            )

            target_colors = scalar_to_int(
                data.selected_num_colors
            )

            best_colpack5_colors = (
                scalar_to_int(
                    data.best_colpack5_colors
                )
            )

            gap_from_target = (
                num_colors
                - target_colors
            )

            gap_from_colpack5 = (
                num_colors
                - best_colpack5_colors
            )

            total_colors += num_colors
            total_target_colors += (
                target_colors
            )

            total_colpack5_colors += (
                best_colpack5_colors
            )

            total_gap_from_target += (
                gap_from_target
            )

            total_gap_from_colpack5 += (
                gap_from_colpack5
            )

            if num_colors == target_colors:
                exact_target_graphs += 1

            if num_colors < best_colpack5_colors:
                better_than_colpack_graphs += 1

            elif num_colors == best_colpack5_colors:
                tied_with_colpack_graphs += 1

            else:
                worse_than_colpack_graphs += 1

            all_valid = (
                all_valid
                and valid
            )

            per_graph_parts.append(
                f"{data.graph_id}:"
                f"{graph_description(data)}/"
                f"gnn{num_colors}/"
                f"target{target_colors}/"
                f"colpack5{best_colpack5_colors}"
            )

    return {
        "total_colors": total_colors,
        "average_colors": (
            total_colors
            / len(graphs)
        ),
        "total_target_colors": (
            total_target_colors
        ),
        "total_colpack5_colors": (
            total_colpack5_colors
        ),
        "total_gap_from_target": (
            total_gap_from_target
        ),
        "average_gap_from_target": (
            total_gap_from_target
            / len(graphs)
        ),
        "total_gap_from_colpack5": (
            total_gap_from_colpack5
        ),
        "average_gap_from_colpack5": (
            total_gap_from_colpack5
            / len(graphs)
        ),
        "colors_saved_vs_colpack5": (
            total_colpack5_colors
            - total_colors
        ),
        "exact_target_graphs": (
            exact_target_graphs
        ),
        "better_than_colpack_graphs": (
            better_than_colpack_graphs
        ),
        "tied_with_colpack_graphs": (
            tied_with_colpack_graphs
        ),
        "worse_than_colpack_graphs": (
            worse_than_colpack_graphs
        ),
        "all_valid": all_valid,
        "per_graph_colors": (
            "; ".join(
                per_graph_parts
            )
        ),
    }


def is_better_checkpoint(
    current_validation_total_colors: int,
    current_validation_loss: float,
    best_validation_total_colors: int | None,
    best_validation_loss: float,
) -> bool:
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


def validate_frozen_evaluation_totals(
    validation_graphs: list,
    test_graphs: list,
) -> None:
    for split_name, graphs in [
        (
            "validation",
            validation_graphs,
        ),
        (
            "test",
            test_graphs,
        ),
    ]:
        target_total = sum(
            scalar_to_int(
                data.selected_num_colors
            )
            for data in graphs
        )

        colpack_total = sum(
            scalar_to_int(
                data.best_colpack5_colors
            )
            for data in graphs
        )

        if (
            target_total
            != EXPECTED_EVALUATION_TARGET_TOTAL
        ):
            raise ValueError(
                f"{split_name}: expected target total "
                f"{EXPECTED_EVALUATION_TARGET_TOTAL}, "
                f"found {target_total}."
            )

        if (
            colpack_total
            != EXPECTED_EVALUATION_COLPACK_TOTAL
        ):
            raise ValueError(
                f"{split_name}: expected ColPack total "
                f"{EXPECTED_EVALUATION_COLPACK_TOTAL}, "
                f"found {colpack_total}."
            )


def train_single_run(
    condition: str,
    seed: int,
    train_graphs: list,
    validation_graphs: list,
    test_graphs: list,
    condition_manifest_df: pd.DataFrame,
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

    best_validation_total_colors = None
    best_validation_loss = float("inf")
    best_validation_coloring = None

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

        validation_loss = (
            compute_average_loss(
                model=model,
                graphs=validation_graphs,
                loss_fn=loss_fn,
            )
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
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 50 == 0
        ):
            print(
                f"{condition} | seed {seed} | "
                f"epoch {epoch:03d} | "
                f"validation loss "
                f"{validation_loss:.6f} | "
                f"validation colors "
                f"{validation_total_colors} | "
                f"target gap "
                f"{validation_coloring['total_gap_from_target']} | "
                f"saved vs ColPack "
                f"{validation_coloring['colors_saved_vs_colpack5']}"
            )

    if best_model_state is None:
        raise RuntimeError(
            f"{condition}, seed {seed}: "
            "no checkpoint was recorded."
        )

    model.load_state_dict(
        best_model_state
    )

    final_train_loss = (
        compute_average_loss(
            model=model,
            graphs=train_graphs,
            loss_fn=loss_fn,
        )
    )

    final_validation_loss = (
        compute_average_loss(
            model=model,
            graphs=validation_graphs,
            loss_fn=loss_fn,
        )
    )

    final_test_loss = (
        compute_average_loss(
            model=model,
            graphs=test_graphs,
            loss_fn=loss_fn,
        )
    )

    train_coloring = (
        evaluate_coloring_quality(
            model=model,
            graphs=train_graphs,
        )
    )

    validation_coloring = (
        evaluate_coloring_quality(
            model=model,
            graphs=validation_graphs,
        )
    )

    test_coloring = (
        evaluate_coloring_quality(
            model=model,
            graphs=test_graphs,
        )
    )

    if not (
        train_coloring["all_valid"]
        and validation_coloring[
            "all_valid"
        ]
        and test_coloring["all_valid"]
    ):
        raise ValueError(
            f"{condition}, seed {seed}: "
            "an invalid coloring was produced."
        )

    condition_checkpoint_dir = (
        CHECKPOINT_ROOT
        / condition
    )

    condition_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        condition_checkpoint_dir
        / f"best_gnn_node_scorer_seed_{seed}.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                best_model_state
            ),
            "condition": condition,
            "input_dim": input_dim,
            "hidden_channels": (
                HIDDEN_CHANNELS
            ),
            "out_channels": (
                OUT_CHANNELS
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "num_epochs": (
                NUM_EPOCHS
            ),
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
        },
        checkpoint_path,
    )

    train_manifest_df = (
        condition_manifest_df[
            condition_manifest_df["split"]
            == "train"
        ]
    )

    num_added_mixed_graphs = int(
        train_manifest_df[
            "is_added_mixed_graph"
        ].sum()
    )

    total_train_nodes = int(
        train_manifest_df[
            "num_nodes"
        ].sum()
    )

    return {
        "condition": condition,
        "seed": seed,
        "experiment": (
            "WEEK18_CONTROLLED_DATA_SCALING"
        ),
        "model_name": "GNNNodeScorer",
        "target_ordering": (
            "EXACT_OPTIMAL_COLOR_CLASS_ORDER"
        ),
        "feature_set": (
            "WEEK17_SYMMETRY_BREAKING_25"
        ),
        "checkpoint_selection": (
            "validation_total_colors_"
            "then_validation_loss"
        ),
        "num_train_graphs": (
            len(train_graphs)
        ),
        "num_original_train_graphs": (
            len(train_graphs)
            - num_added_mixed_graphs
        ),
        "num_added_mixed_graphs": (
            num_added_mixed_graphs
        ),
        "total_train_nodes": (
            total_train_nodes
        ),
        "num_validation_graphs": (
            len(validation_graphs)
        ),
        "num_test_graphs": (
            len(test_graphs)
        ),
        "input_dim": input_dim,
        "hidden_channels": (
            HIDDEN_CHANNELS
        ),
        "learning_rate": (
            LEARNING_RATE
        ),
        "num_epochs": (
            NUM_EPOCHS
        ),
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
            train_coloring[
                "total_colors"
            ]
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
        "final_train_colors_saved_vs_colpack5": (
            train_coloring[
                "colors_saved_vs_colpack5"
            ]
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
        "final_validation_colors_saved_vs_colpack5": (
            validation_coloring[
                "colors_saved_vs_colpack5"
            ]
        ),
        "final_test_total_colors": (
            test_coloring[
                "total_colors"
            ]
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
        "final_test_colors_saved_vs_colpack5": (
            test_coloring[
                "colors_saved_vs_colpack5"
            ]
        ),
        "final_test_exact_target_graphs": (
            test_coloring[
                "exact_target_graphs"
            ]
        ),
        "final_test_better_than_colpack_graphs": (
            test_coloring[
                "better_than_colpack_graphs"
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
            checkpoint_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }


def load_existing_results() -> pd.DataFrame:
    if not OUTPUT_CSV.exists():
        return pd.DataFrame()

    existing_df = pd.read_csv(
        OUTPUT_CSV
    )

    if existing_df.empty:
        return existing_df

    required_columns = {
        "condition",
        "seed",
    }

    if not required_columns.issubset(
        existing_df.columns
    ):
        raise ValueError(
            "Existing result file does not have "
            "condition and seed columns."
        )

    return existing_df


def save_run_results(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    results_df = pd.DataFrame(
        rows
    )

    results_df = (
        results_df
        .drop_duplicates(
            subset=[
                "condition",
                "seed",
            ],
            keep="last",
        )
        .sort_values(
            [
                "condition",
                "seed",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    return results_df


def create_aggregate_summary(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    aggregate_rows: list[
        dict[str, object]
    ] = []

    condition_order = {
        condition: index
        for index, condition
        in enumerate(CONDITIONS)
    }

    for condition in CONDITIONS:
        condition_df = results_df[
            results_df["condition"]
            == condition
        ].copy()

        if condition_df.empty:
            continue

        representative_row = (
            condition_df
            .sort_values(
                [
                    "final_validation_total_colors",
                    "final_validation_loss_best_model",
                    "seed",
                ]
            )
            .iloc[0]
        )

        aggregate_rows.append(
            {
                "condition": condition,
                "num_completed_seeds": (
                    len(condition_df)
                ),
                "num_train_graphs": int(
                    condition_df[
                        "num_train_graphs"
                    ].iloc[0]
                ),
                "num_added_mixed_graphs": int(
                    condition_df[
                        "num_added_mixed_graphs"
                    ].iloc[0]
                ),
                "total_train_nodes": int(
                    condition_df[
                        "total_train_nodes"
                    ].iloc[0]
                ),
                "mean_test_colors": float(
                    condition_df[
                        "final_test_total_colors"
                    ].mean()
                ),
                "std_test_colors": float(
                    condition_df[
                        "final_test_total_colors"
                    ].std(ddof=0)
                ),
                "minimum_test_colors": int(
                    condition_df[
                        "final_test_total_colors"
                    ].min()
                ),
                "maximum_test_colors": int(
                    condition_df[
                        "final_test_total_colors"
                    ].max()
                ),
                "mean_test_gap_from_target": float(
                    condition_df[
                        "final_test_gap_from_target"
                    ].mean()
                ),
                "mean_colors_saved_vs_colpack5": float(
                    condition_df[
                        "final_test_colors_saved_vs_colpack5"
                    ].mean()
                ),
                "mean_exact_target_graphs": float(
                    condition_df[
                        "final_test_exact_target_graphs"
                    ].mean()
                ),
                "representative_seed": int(
                    representative_row[
                        "seed"
                    ]
                ),
                "representative_validation_colors": int(
                    representative_row[
                        "final_validation_total_colors"
                    ]
                ),
                "representative_validation_loss": float(
                    representative_row[
                        "final_validation_loss_best_model"
                    ]
                ),
                "representative_test_colors": int(
                    representative_row[
                        "final_test_total_colors"
                    ]
                ),
                "representative_test_gap_from_target": int(
                    representative_row[
                        "final_test_gap_from_target"
                    ]
                ),
                "representative_colors_saved_vs_colpack5": int(
                    representative_row[
                        "final_test_colors_saved_vs_colpack5"
                    ]
                ),
                "all_test_colorings_valid": bool(
                    condition_df[
                        "final_test_all_valid"
                    ].all()
                ),
            }
        )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )

    if not aggregate_df.empty:
        aggregate_df[
            "_condition_order"
        ] = aggregate_df[
            "condition"
        ].map(condition_order)

        aggregate_df = (
            aggregate_df
            .sort_values(
                "_condition_order"
            )
            .drop(
                columns=[
                    "_condition_order"
                ]
            )
            .reset_index(drop=True)
        )

    AGGREGATE_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_df.to_csv(
        AGGREGATE_OUTPUT_CSV,
        index=False,
    )

    return aggregate_df


def main() -> None:
    arguments = parse_arguments()

    manifest_df = load_manifest()

    if arguments.condition == "all":
        conditions_to_run = CONDITIONS
    else:
        conditions_to_run = [
            arguments.condition
        ]

    existing_df = (
        load_existing_results()
    )

    if arguments.restart:
        if existing_df.empty:
            retained_df = existing_df
        else:
            retained_df = existing_df[
                ~existing_df[
                    "condition"
                ].isin(
                    conditions_to_run
                )
            ].copy()

    else:
        retained_df = existing_df.copy()

    result_rows = (
        retained_df
        .to_dict(orient="records")
        if not retained_df.empty
        else []
    )

    completed_pairs = {
        (
            str(row["condition"]),
            int(row["seed"]),
        )
        for row in result_rows
    }

    print(
        "Week 18 controlled data-scaling "
        "GNN training"
    )
    print(
        "------------------------------------------"
    )
    print(
        f"Conditions: {conditions_to_run}"
    )
    print(
        f"Seeds: {SEEDS}"
    )
    print(
        f"Epochs per run: {NUM_EPOCHS}"
    )
    print()

    for condition in conditions_to_run:
        (
            train_graphs,
            validation_graphs,
            test_graphs,
            condition_manifest_df,
        ) = load_condition_dataset(
            manifest_df=manifest_df,
            condition=condition,
        )

        validate_frozen_evaluation_totals(
            validation_graphs=(
                validation_graphs
            ),
            test_graphs=test_graphs,
        )

        print(
            f"Condition: {condition}"
        )
        print(
            f"Training graphs: "
            f"{len(train_graphs)}"
        )
        print(
            f"Validation graphs: "
            f"{len(validation_graphs)}"
        )
        print(
            f"Test graphs: "
            f"{len(test_graphs)}"
        )
        print()

        for seed in SEEDS:
            pair = (
                condition,
                seed,
            )

            if (
                pair in completed_pairs
                and not arguments.restart
            ):
                print(
                    f"Skipping {condition}, seed {seed}: "
                    "result already exists."
                )
                continue

            print(
                f"Training {condition}, seed {seed}"
            )
            print(
                "----------------------------------"
            )

            row = train_single_run(
                condition=condition,
                seed=seed,
                train_graphs=train_graphs,
                validation_graphs=(
                    validation_graphs
                ),
                test_graphs=test_graphs,
                condition_manifest_df=(
                    condition_manifest_df
                ),
            )

            result_rows.append(row)
            completed_pairs.add(pair)

            current_results_df = (
                save_run_results(
                    result_rows
                )
            )

            print()
            print(
                f"Selected epoch: "
                f"{row['best_epoch']}"
            )
            print(
                f"Validation colors: "
                f"{row['final_validation_total_colors']} "
                f"(target 60, ColPack 75)"
            )
            print(
                f"Test colors: "
                f"{row['final_test_total_colors']} "
                f"(target 60, ColPack 75)"
            )
            print(
                f"Test colors saved vs ColPack: "
                f"{row['final_test_colors_saved_vs_colpack5']}"
            )
            print(
                f"Test valid: "
                f"{row['final_test_all_valid']}"
            )
            print()

    final_results_df = (
        save_run_results(
            result_rows
        )
    )

    aggregate_df = (
        create_aggregate_summary(
            final_results_df
        )
    )

    print()
    print(
        "Controlled data-scaling training complete."
    )
    print(
        "------------------------------------------"
    )

    display_columns = [
        "condition",
        "num_completed_seeds",
        "num_train_graphs",
        "mean_test_colors",
        "std_test_colors",
        "minimum_test_colors",
        "maximum_test_colors",
        "mean_test_gap_from_target",
        "mean_colors_saved_vs_colpack5",
        "representative_seed",
        "representative_test_colors",
    ]

    print(
        aggregate_df[
            display_columns
        ].round(3).to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved run summary to: "
        f"{OUTPUT_CSV}"
    )
    print(
        f"Saved aggregate summary to: "
        f"{AGGREGATE_OUTPUT_CSV}"
    )
    print(
        f"Saved checkpoints under: "
        f"{CHECKPOINT_ROOT}"
    )


if __name__ == "__main__":
    main()