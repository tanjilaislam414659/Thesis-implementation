from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
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

COLOR_CLASS_TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week19_controlled_color_class_targets.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week19_equivalence_aware_objective_run_summary.csv"
)

AGGREGATE_OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week19_equivalence_aware_objective_aggregate.csv"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gnn_node_scorer"
    / "week19_equivalence_aware_objective_runs"
)

CONDITION = "train_125_plus105"
OBJECTIVES = ["mse", "ranking"]
SEEDS = [0, 1, 2, 3, 4]

HIDDEN_CHANNELS = 32
OUT_CHANNELS = 1
NUM_EPOCHS = 500
LEARNING_RATE = 0.01
RANKING_PAIRS_PER_GRAPH = 256

EXPECTED_FEATURE_COUNT = 25
EXPECTED_TRAIN_GRAPHS = 125
EXPECTED_VALIDATION_GRAPHS = 5
EXPECTED_TEST_GRAPHS = 5
EXPECTED_TOTAL_GRAPHS = 135
EXPECTED_COLOR_CLASS_ROWS = 15045
EXPECTED_EVALUATION_TARGET_TOTAL = 60
EXPECTED_EVALUATION_COLPACK_TOTAL = 75


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Week 19 MSE-versus-equivalence-aware "
            "ranking-objective comparison."
        )
    )

    parser.add_argument(
        "--objective",
        choices=OBJECTIVES + ["all"],
        default="all",
        help="Objective to run. The default runs both.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        default=None,
        help="Run only one seed. The default runs seeds 0-4.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Rerun the requested objective and seed "
            "combinations even if results already exist."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate data, labels, pair sampling, losses, "
            "and coloring without training or saving results."
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


def load_full_condition_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest_df = pd.read_csv(MANIFEST_PATH)

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
    missing = required_columns - set(
        manifest_df.columns
    )

    if missing:
        raise ValueError(
            "Manifest is missing columns: "
            f"{sorted(missing)}"
        )

    condition_df = manifest_df[
        manifest_df["condition"] == CONDITION
    ].copy()

    if len(condition_df) != EXPECTED_TOTAL_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_GRAPHS} graphs for "
            f"{CONDITION}, found {len(condition_df)}."
        )

    if condition_df["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in the full condition."
        )

    expected_split_counts = {
        "train": EXPECTED_TRAIN_GRAPHS,
        "validation": EXPECTED_VALIDATION_GRAPHS,
        "test": EXPECTED_TEST_GRAPHS,
    }
    actual_split_counts = (
        condition_df["split"]
        .value_counts()
        .to_dict()
    )

    if actual_split_counts != expected_split_counts:
        raise ValueError(
            "Unexpected full-condition split counts: "
            f"{actual_split_counts}."
        )

    return condition_df


def load_color_class_targets() -> pd.DataFrame:
    if not COLOR_CLASS_TARGET_PATH.exists():
        raise FileNotFoundError(
            "Week 19 color-class target file not found: "
            f"{COLOR_CLASS_TARGET_PATH}"
        )

    target_df = pd.read_csv(
        COLOR_CLASS_TARGET_PATH
    )
    required_columns = {
        "graph_id",
        "node_id",
        "known_color",
        "existing_target_score",
        "split",
        "label_source",
    }
    missing = required_columns - set(
        target_df.columns
    )

    if missing:
        raise ValueError(
            "Color-class target file is missing columns: "
            f"{sorted(missing)}"
        )

    if len(target_df) != EXPECTED_COLOR_CLASS_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_COLOR_CLASS_ROWS} color-class "
            f"rows, found {len(target_df)}."
        )

    if target_df[
        ["graph_id", "node_id"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate graph/node rows found in color targets."
        )

    return target_df


def attach_color_classes(
    data,
    graph_target_df: pd.DataFrame,
) -> None:
    graph_target_df = (
        graph_target_df
        .sort_values("node_id")
        .reset_index(drop=True)
    )
    expected_node_ids = list(
        range(data.num_nodes)
    )
    actual_node_ids = (
        graph_target_df["node_id"]
        .astype(int)
        .tolist()
    )

    if actual_node_ids != expected_node_ids:
        raise ValueError(
            f"{data.graph_id}: color-class target node IDs "
            "do not match 0..num_nodes-1."
        )

    existing_scores = torch.tensor(
        graph_target_df[
            "existing_target_score"
        ].values,
        dtype=torch.float32,
    )

    if not torch.allclose(
        existing_scores,
        data.y.detach().cpu().view(-1),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(
            f"{data.graph_id}: Week 19 target rows do not "
            "match the existing MSE scores."
        )

    data.week19_color_classes = torch.tensor(
        graph_target_df["known_color"].values,
        dtype=torch.long,
    )
    data.week19_label_source = str(
        graph_target_df["label_source"].iloc[0]
    )


def load_dataset() -> tuple[
    list,
    list,
    list,
    pd.DataFrame,
]:
    condition_df = load_full_condition_manifest()
    target_df = load_color_class_targets()

    manifest_ids = set(
        condition_df["graph_id"].astype(str)
    )
    target_ids = set(
        target_df["graph_id"].astype(str)
    )

    if manifest_ids != target_ids:
        raise ValueError(
            "Manifest and Week 19 color-target graph IDs differ."
        )

    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }
    condition_df["_split_order"] = (
        condition_df["split"].map(split_order)
    )
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

    for row in condition_df.itertuples(index=False):
        source_path = (
            PROJECT_ROOT
            / str(row.source_pt_path)
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"PyG file not found: {source_path}"
            )

        data = load_torch_data(source_path)

        if str(data.graph_id) != str(row.graph_id):
            raise ValueError(
                f"Graph-ID mismatch for {source_path}."
            )

        if int(data.x.shape[1]) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"{data.graph_id}: expected "
                f"{EXPECTED_FEATURE_COUNT} features, found "
                f"{data.x.shape[1]}."
            )

        if data.y.numel() != data.num_nodes:
            raise ValueError(
                f"{data.graph_id}: MSE target count does not "
                "match the node count."
            )

        graph_target_df = target_df[
            target_df["graph_id"].astype(str)
            == str(row.graph_id)
        ]
        attach_color_classes(
            data=data,
            graph_target_df=graph_target_df,
        )

        data.split = str(row.split)
        data.source_dataset = str(row.source_dataset)

        if scalar_to_int(
            data.selected_num_colors
        ) != int(row.target_colors):
            raise ValueError(
                f"{data.graph_id}: target-color mismatch."
            )

        if scalar_to_int(
            data.best_colpack5_colors
        ) != int(row.best_colpack5_colors):
            raise ValueError(
                f"{data.graph_id}: ColPack-color mismatch."
            )

        grouped_graphs[str(row.split)].append(data)

    train_graphs = grouped_graphs["train"]
    validation_graphs = grouped_graphs["validation"]
    test_graphs = grouped_graphs["test"]

    if len(train_graphs) != EXPECTED_TRAIN_GRAPHS:
        raise ValueError("Incorrect training-graph count.")
    if len(validation_graphs) != EXPECTED_VALIDATION_GRAPHS:
        raise ValueError("Incorrect validation-graph count.")
    if len(test_graphs) != EXPECTED_TEST_GRAPHS:
        raise ValueError("Incorrect test-graph count.")

    validate_frozen_evaluation_totals(
        validation_graphs=validation_graphs,
        test_graphs=test_graphs,
    )

    return (
        train_graphs,
        validation_graphs,
        test_graphs,
        condition_df,
    )


def validate_frozen_evaluation_totals(
    validation_graphs: list,
    test_graphs: list,
) -> None:
    for split_name, graphs in [
        ("validation", validation_graphs),
        ("test", test_graphs),
    ]:
        target_total = sum(
            scalar_to_int(data.selected_num_colors)
            for data in graphs
        )
        colpack_total = sum(
            scalar_to_int(data.best_colpack5_colors)
            for data in graphs
        )

        if target_total != EXPECTED_EVALUATION_TARGET_TOTAL:
            raise ValueError(
                f"{split_name}: expected target total "
                f"{EXPECTED_EVALUATION_TARGET_TOTAL}, "
                f"found {target_total}."
            )
        if colpack_total != EXPECTED_EVALUATION_COLPACK_TOTAL:
            raise ValueError(
                f"{split_name}: expected ColPack total "
                f"{EXPECTED_EVALUATION_COLPACK_TOTAL}, "
                f"found {colpack_total}."
            )


def build_balanced_ranking_pair_indices(
    colors: torch.Tensor,
    pair_budget: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, int], int]]:
    colors = colors.detach().cpu().view(-1)
    unique_colors = sorted(
        int(value)
        for value in colors.unique().tolist()
    )

    if len(unique_colors) < 2:
        raise ValueError(
            "Ranking loss requires at least two color classes."
        )

    nodes_by_color = {
        color: torch.nonzero(
            colors == color,
            as_tuple=False,
        ).view(-1).tolist()
        for color in unique_colors
    }
    class_pairs = list(
        itertools.combinations(unique_colors, 2)
    )

    full_repeats, remainder = divmod(
        pair_budget,
        len(class_pairs),
    )
    sampled_class_pairs = (
        class_pairs * full_repeats
    )

    if remainder:
        sampled_class_pairs.extend(
            rng.sample(class_pairs, remainder)
        )

    rng.shuffle(sampled_class_pairs)

    earlier_nodes: list[int] = []
    later_nodes: list[int] = []
    class_pair_counts = {
        pair: 0
        for pair in class_pairs
    }

    for earlier_color, later_color in sampled_class_pairs:
        earlier_nodes.append(
            rng.choice(nodes_by_color[earlier_color])
        )
        later_nodes.append(
            rng.choice(nodes_by_color[later_color])
        )
        class_pair_counts[
            (earlier_color, later_color)
        ] += 1

    return (
        torch.tensor(earlier_nodes, dtype=torch.long),
        torch.tensor(later_nodes, dtype=torch.long),
        class_pair_counts,
    )


def sampled_ranking_loss(
    predictions: torch.Tensor,
    colors: torch.Tensor,
    rng: random.Random,
) -> torch.Tensor:
    earlier_indices, later_indices, _ = (
        build_balanced_ranking_pair_indices(
            colors=colors,
            pair_budget=RANKING_PAIRS_PER_GRAPH,
            rng=rng,
        )
    )
    scores = predictions.view(-1)

    return F.softplus(
        scores[later_indices]
        - scores[earlier_indices]
    ).mean()


def complete_ranking_loss(
    predictions: torch.Tensor,
    colors: torch.Tensor,
) -> torch.Tensor:
    scores = predictions.view(-1)
    colors = colors.view(-1)
    earlier_indices, later_indices = torch.where(
        colors[:, None] < colors[None, :]
    )

    if earlier_indices.numel() == 0:
        raise ValueError(
            "No cross-class validation pairs were found."
        )

    return F.softplus(
        scores[later_indices]
        - scores[earlier_indices]
    ).mean()


def compute_average_loss(
    model: GNNNodeScorer,
    graphs: list,
    objective: str,
) -> float:
    if not graphs:
        raise ValueError(
            "Cannot compute loss on an empty graph list."
        )

    model.eval()
    total_loss = 0.0
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for data in graphs:
            predictions = model(
                data.x,
                data.edge_index,
            )

            if objective == "mse":
                loss = mse_loss(
                    predictions.view(-1),
                    data.y.view(-1),
                )
            elif objective == "ranking":
                loss = complete_ranking_loss(
                    predictions=predictions,
                    colors=data.week19_color_classes,
                )
            else:
                raise ValueError(
                    f"Unknown objective: {objective}"
                )

            total_loss += float(loss.item())

    return total_loss / len(graphs)


def pyg_data_to_networkx_graph(data) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.detach().cpu()

    for source, target in edge_index.t().tolist():
        graph.add_edge(int(source), int(target))

    return graph


def graph_description(data) -> str:
    gap_level = scalar_to_int(data.gap_level)

    if hasattr(data, "base_cycle_size"):
        structure = (
            f"base{scalar_to_int(data.base_cycle_size)}"
        )
    elif hasattr(data, "component_cycle_sizes"):
        structure = (
            f"mixed[{str(data.component_cycle_sizes)}]"
        )
    else:
        structure = "structure_unknown"

    return f"{structure}/gap{gap_level}"


def evaluate_coloring_quality(
    model: GNNNodeScorer,
    graphs: list,
) -> dict[str, object]:
    if not graphs:
        raise ValueError(
            "Cannot evaluate coloring on an empty graph list."
        )

    model.eval()
    total_colors = 0
    total_target_colors = 0
    total_colpack5_colors = 0
    exact_target_graphs = 0
    all_valid = True
    per_graph_parts: list[str] = []

    with torch.no_grad():
        for data in graphs:
            graph = pyg_data_to_networkx_graph(data)
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
            num_colors = int(count_colors(coloring))
            valid = bool(
                is_valid_coloring(graph, coloring)
            )
            target_colors = scalar_to_int(
                data.selected_num_colors
            )
            colpack_colors = scalar_to_int(
                data.best_colpack5_colors
            )

            total_colors += num_colors
            total_target_colors += target_colors
            total_colpack5_colors += colpack_colors
            exact_target_graphs += int(
                num_colors == target_colors
            )
            all_valid = all_valid and valid
            per_graph_parts.append(
                f"{data.graph_id}:"
                f"{graph_description(data)}/"
                f"gnn{num_colors}/"
                f"target{target_colors}/"
                f"colpack5{colpack_colors}"
            )

    return {
        "total_colors": total_colors,
        "average_colors": total_colors / len(graphs),
        "total_target_colors": total_target_colors,
        "total_colpack5_colors": total_colpack5_colors,
        "total_gap_from_target": (
            total_colors - total_target_colors
        ),
        "colors_saved_vs_colpack5": (
            total_colpack5_colors - total_colors
        ),
        "exact_target_graphs": exact_target_graphs,
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
    if (
        current_validation_total_colors
        == best_validation_total_colors
    ):
        return current_validation_loss < best_validation_loss
    return False


def train_single_run(
    objective: str,
    seed: int,
    train_graphs: list,
    validation_graphs: list,
    test_graphs: list,
    condition_manifest_df: pd.DataFrame,
) -> dict[str, object]:
    set_random_seed(seed)

    input_dim = int(train_graphs[0].x.shape[1])
    model = GNNNodeScorer(
        in_channels=input_dim,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
    )
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    mse_loss = nn.MSELoss()

    # Separate RNGs keep graph shuffling identical across objectives.
    shuffle_rng = random.Random(seed)
    pair_rng = random.Random(seed + 100_000)

    best_epoch = 0
    best_model_state = None
    best_validation_total_colors = None
    best_validation_loss = float("inf")
    best_validation_coloring = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_graphs = list(train_graphs)
        shuffle_rng.shuffle(epoch_graphs)

        for data in epoch_graphs:
            optimizer.zero_grad()
            predictions = model(
                data.x,
                data.edge_index,
            )

            if objective == "mse":
                loss = mse_loss(
                    predictions.view(-1),
                    data.y.view(-1),
                )
            elif objective == "ranking":
                loss = sampled_ranking_loss(
                    predictions=predictions,
                    colors=data.week19_color_classes,
                    rng=pair_rng,
                )
            else:
                raise ValueError(
                    f"Unknown objective: {objective}"
                )

            loss.backward()
            optimizer.step()

        validation_loss = compute_average_loss(
            model=model,
            graphs=validation_graphs,
            objective=objective,
        )
        validation_coloring = evaluate_coloring_quality(
            model=model,
            graphs=validation_graphs,
        )
        validation_total_colors = int(
            validation_coloring["total_colors"]
        )

        if is_better_checkpoint(
            current_validation_total_colors=(
                validation_total_colors
            ),
            current_validation_loss=validation_loss,
            best_validation_total_colors=(
                best_validation_total_colors
            ),
            best_validation_loss=best_validation_loss,
        ):
            best_epoch = epoch
            best_validation_total_colors = (
                validation_total_colors
            )
            best_validation_loss = validation_loss
            best_validation_coloring = validation_coloring
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"{objective} | seed {seed} | "
                f"epoch {epoch:03d} | "
                f"validation loss {validation_loss:.6f} | "
                f"validation colors {validation_total_colors} | "
                f"target gap "
                f"{validation_coloring['total_gap_from_target']}"
            )

    if best_model_state is None:
        raise RuntimeError(
            f"{objective}, seed {seed}: no checkpoint recorded."
        )

    model.load_state_dict(best_model_state)

    final_train_loss = compute_average_loss(
        model, train_graphs, objective
    )
    final_validation_loss = compute_average_loss(
        model, validation_graphs, objective
    )
    final_test_loss = compute_average_loss(
        model, test_graphs, objective
    )
    train_coloring = evaluate_coloring_quality(
        model, train_graphs
    )
    validation_coloring = evaluate_coloring_quality(
        model, validation_graphs
    )
    test_coloring = evaluate_coloring_quality(
        model, test_graphs
    )

    if not (
        train_coloring["all_valid"]
        and validation_coloring["all_valid"]
        and test_coloring["all_valid"]
    ):
        raise ValueError(
            f"{objective}, seed {seed}: invalid coloring produced."
        )

    checkpoint_dir = CHECKPOINT_ROOT / objective
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir
        / f"week19_best_gnn_node_scorer_seed_{seed}.pt"
    )

    torch.save(
        {
            "model_state_dict": best_model_state,
            "condition": CONDITION,
            "objective": objective,
            "input_dim": input_dim,
            "hidden_channels": HIDDEN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "ranking_pairs_per_graph": (
                RANKING_PAIRS_PER_GRAPH
                if objective == "ranking"
                else None
            ),
            "graph_order_shuffled_each_epoch": True,
            "seed": seed,
            "best_epoch": best_epoch,
            "selection_metric": (
                "validation_total_colors_then_"
                "objective_specific_validation_loss"
            ),
            "best_validation_total_colors": (
                best_validation_total_colors
            ),
            "best_validation_loss": best_validation_loss,
            "best_validation_coloring": (
                best_validation_coloring
            ),
        },
        checkpoint_path,
    )

    train_manifest_df = condition_manifest_df[
        condition_manifest_df["split"] == "train"
    ]

    return {
        "condition": CONDITION,
        "objective": objective,
        "seed": seed,
        "experiment": (
            "WEEK19_EQUIVALENCE_AWARE_OBJECTIVE_COMPARISON"
        ),
        "model_name": "GNNNodeScorer",
        "target_ordering": (
            "EXACT_OPTIMAL_COLOR_CLASS_ORDER"
        ),
        "feature_set": "WEEK17_SYMMETRY_BREAKING_25",
        "loss_definition": (
            "node_score_mse"
            if objective == "mse"
            else "mean_softplus_later_minus_earlier"
        ),
        "ranking_pairs_per_train_graph": (
            RANKING_PAIRS_PER_GRAPH
            if objective == "ranking"
            else 0
        ),
        "validation_ranking_pairs": (
            "all_cross_class_pairs"
            if objective == "ranking"
            else "not_applicable"
        ),
        "graph_order_shuffled_each_epoch": True,
        "checkpoint_selection": (
            "validation_total_colors_then_"
            "objective_specific_validation_loss"
        ),
        "num_train_graphs": len(train_graphs),
        "num_original_train_graphs": int(
            (~train_manifest_df[
                "is_added_mixed_graph"
            ].astype(bool)).sum()
        ),
        "num_added_mixed_graphs": int(
            train_manifest_df[
                "is_added_mixed_graph"
            ].astype(bool).sum()
        ),
        "num_validation_graphs": len(validation_graphs),
        "num_test_graphs": len(test_graphs),
        "input_dim": input_dim,
        "hidden_channels": HIDDEN_CHANNELS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_total_colors": (
            best_validation_total_colors
        ),
        "best_validation_loss": best_validation_loss,
        "final_train_loss_best_model": final_train_loss,
        "final_validation_loss_best_model": (
            final_validation_loss
        ),
        "final_test_loss_best_model": final_test_loss,
        "final_train_total_colors": (
            train_coloring["total_colors"]
        ),
        "final_train_target_colors": (
            train_coloring["total_target_colors"]
        ),
        "final_train_gap_from_target": (
            train_coloring["total_gap_from_target"]
        ),
        "final_validation_total_colors": (
            validation_coloring["total_colors"]
        ),
        "final_validation_target_colors": (
            validation_coloring["total_target_colors"]
        ),
        "final_validation_gap_from_target": (
            validation_coloring["total_gap_from_target"]
        ),
        "final_test_total_colors": (
            test_coloring["total_colors"]
        ),
        "final_test_target_colors": (
            test_coloring["total_target_colors"]
        ),
        "final_test_colpack5_colors": (
            test_coloring["total_colpack5_colors"]
        ),
        "final_test_gap_from_target": (
            test_coloring["total_gap_from_target"]
        ),
        "final_test_colors_saved_vs_colpack5": (
            test_coloring["colors_saved_vs_colpack5"]
        ),
        "final_test_exact_target_graphs": (
            test_coloring["exact_target_graphs"]
        ),
        "final_test_all_valid": test_coloring["all_valid"],
        "final_test_per_graph_colors": (
            test_coloring["per_graph_colors"]
        ),
        "checkpoint_path": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ),
    }


def validate_experiment_setup(
    train_graphs: list,
    validation_graphs: list,
    test_graphs: list,
) -> None:
    set_random_seed(0)
    model = GNNNodeScorer(
        in_channels=int(train_graphs[0].x.shape[1]),
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
    )
    pair_rng = random.Random(100_000)

    for data in train_graphs:
        earlier, later, counts = (
            build_balanced_ranking_pair_indices(
                colors=data.week19_color_classes,
                pair_budget=RANKING_PAIRS_PER_GRAPH,
                rng=pair_rng,
            )
        )
        if earlier.numel() != RANKING_PAIRS_PER_GRAPH:
            raise ValueError(
                f"{data.graph_id}: wrong ranking-pair count."
            )
        colors = data.week19_color_classes
        if not bool(
            (colors[earlier] < colors[later]).all().item()
        ):
            raise ValueError(
                f"{data.graph_id}: invalid ranking-pair direction."
            )
        count_values = list(counts.values())
        if max(count_values) - min(count_values) > 1:
            raise ValueError(
                f"{data.graph_id}: class-pair sampling is unbalanced."
            )

    sample_data = train_graphs[0]
    predictions = model(
        sample_data.x,
        sample_data.edge_index,
    )
    mse_value = nn.MSELoss()(
        predictions.view(-1),
        sample_data.y.view(-1),
    )
    sampled_rank_value = sampled_ranking_loss(
        predictions=predictions,
        colors=sample_data.week19_color_classes,
        rng=pair_rng,
    )
    complete_rank_value = complete_ranking_loss(
        predictions=predictions,
        colors=sample_data.week19_color_classes,
    )

    for name, value in [
        ("MSE", mse_value),
        ("sampled ranking", sampled_rank_value),
        ("complete ranking", complete_rank_value),
    ]:
        if not bool(torch.isfinite(value).item()):
            raise ValueError(
                f"Non-finite {name} loss during validation."
            )

    validation_coloring = evaluate_coloring_quality(
        model, validation_graphs
    )
    test_coloring = evaluate_coloring_quality(
        model, test_graphs
    )

    if not (
        validation_coloring["all_valid"]
        and test_coloring["all_valid"]
    ):
        raise ValueError(
            "Untrained-model coloring validation failed."
        )

    print(
        "Week 19 objective-comparison setup "
        "validated successfully."
    )
    print("------------------------------------------")
    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(validation_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    print(
        f"Ranking pairs per training graph: "
        f"{RANKING_PAIRS_PER_GRAPH}"
    )
    print("Class-pair balance: maximum difference 1")
    print(f"Sample MSE loss: {float(mse_value.item()):.6f}")
    print(
        "Sample sampled-ranking loss: "
        f"{float(sampled_rank_value.item()):.6f}"
    )
    print(
        "Sample complete-ranking loss: "
        f"{float(complete_rank_value.item()):.6f}"
    )
    print("All validation checks passed.")


def load_existing_results() -> pd.DataFrame:
    if not OUTPUT_CSV.exists():
        return pd.DataFrame()
    existing_df = pd.read_csv(OUTPUT_CSV)

    if existing_df.empty:
        return existing_df
    if not {"objective", "seed"}.issubset(
        existing_df.columns
    ):
        raise ValueError(
            "Existing Week 19 result file lacks objective/seed."
        )
    return existing_df


def save_run_results(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    results_df = pd.DataFrame(rows)
    results_df = (
        results_df
        .drop_duplicates(
            subset=["objective", "seed"],
            keep="last",
        )
        .sort_values(["objective", "seed"])
        .reset_index(drop=True)
    )
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    results_df.to_csv(OUTPUT_CSV, index=False)
    return results_df


def create_aggregate_summary(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    aggregate_rows: list[dict[str, object]] = []

    for objective in OBJECTIVES:
        objective_df = results_df[
            results_df["objective"] == objective
        ].copy()

        if objective_df.empty:
            continue

        representative = (
            objective_df
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
                "objective": objective,
                "num_completed_seeds": len(objective_df),
                "num_train_graphs": int(
                    objective_df["num_train_graphs"].iloc[0]
                ),
                "mean_test_colors": float(
                    objective_df[
                        "final_test_total_colors"
                    ].mean()
                ),
                "std_test_colors": float(
                    objective_df[
                        "final_test_total_colors"
                    ].std(ddof=0)
                ),
                "minimum_test_colors": int(
                    objective_df[
                        "final_test_total_colors"
                    ].min()
                ),
                "maximum_test_colors": int(
                    objective_df[
                        "final_test_total_colors"
                    ].max()
                ),
                "mean_test_gap_from_target": float(
                    objective_df[
                        "final_test_gap_from_target"
                    ].mean()
                ),
                "mean_colors_saved_vs_colpack5": float(
                    objective_df[
                        "final_test_colors_saved_vs_colpack5"
                    ].mean()
                ),
                "mean_exact_target_graphs": float(
                    objective_df[
                        "final_test_exact_target_graphs"
                    ].mean()
                ),
                "total_exact_target_runs": int(
                    objective_df[
                        "final_test_exact_target_graphs"
                    ].sum()
                ),
                "representative_seed": int(
                    representative["seed"]
                ),
                "representative_validation_colors": int(
                    representative[
                        "final_validation_total_colors"
                    ]
                ),
                "representative_validation_loss": float(
                    representative[
                        "final_validation_loss_best_model"
                    ]
                ),
                "representative_test_colors": int(
                    representative[
                        "final_test_total_colors"
                    ]
                ),
                "representative_test_gap_from_target": int(
                    representative[
                        "final_test_gap_from_target"
                    ]
                ),
                "representative_exact_target_graphs": int(
                    representative[
                        "final_test_exact_target_graphs"
                    ]
                ),
                "all_test_colorings_valid": bool(
                    objective_df[
                        "final_test_all_valid"
                    ].all()
                ),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows)

    if not aggregate_df.empty:
        objective_order = {
            objective: index
            for index, objective in enumerate(OBJECTIVES)
        }
        aggregate_df["_order"] = (
            aggregate_df["objective"].map(objective_order)
        )
        aggregate_df = (
            aggregate_df
            .sort_values("_order")
            .drop(columns=["_order"])
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
    (
        train_graphs,
        validation_graphs,
        test_graphs,
        condition_manifest_df,
    ) = load_dataset()

    if arguments.validate_only:
        validate_experiment_setup(
            train_graphs=train_graphs,
            validation_graphs=validation_graphs,
            test_graphs=test_graphs,
        )
        return

    objectives_to_run = (
        OBJECTIVES
        if arguments.objective == "all"
        else [arguments.objective]
    )
    seeds_to_run = (
        SEEDS
        if arguments.seed is None
        else [arguments.seed]
    )

    existing_df = load_existing_results()
    requested_pairs = {
        (objective, seed)
        for objective in objectives_to_run
        for seed in seeds_to_run
    }

    if arguments.restart and not existing_df.empty:
        keep_mask = [
            (str(row.objective), int(row.seed))
            not in requested_pairs
            for row in existing_df.itertuples(index=False)
        ]
        retained_df = existing_df[keep_mask].copy()
    else:
        retained_df = existing_df.copy()

    result_rows = (
        retained_df.to_dict(orient="records")
        if not retained_df.empty
        else []
    )
    completed_pairs = {
        (str(row["objective"]), int(row["seed"]))
        for row in result_rows
    }

    print("Week 19 equivalence-aware objective comparison")
    print("------------------------------------------")
    print(f"Condition: {CONDITION}")
    print(f"Objectives: {objectives_to_run}")
    print(f"Seeds: {seeds_to_run}")
    print(f"Epochs per run: {NUM_EPOCHS}")
    print(
        f"Ranking pairs per graph: "
        f"{RANKING_PAIRS_PER_GRAPH}"
    )
    print("Graph shuffling: enabled for both objectives")
    print()

    for objective in objectives_to_run:
        for seed in seeds_to_run:
            pair = (objective, seed)

            if pair in completed_pairs and not arguments.restart:
                print(
                    f"Skipping {objective}, seed {seed}: "
                    "result already exists."
                )
                continue

            print(f"Training {objective}, seed {seed}")
            print("----------------------------------")
            row = train_single_run(
                objective=objective,
                seed=seed,
                train_graphs=train_graphs,
                validation_graphs=validation_graphs,
                test_graphs=test_graphs,
                condition_manifest_df=condition_manifest_df,
            )
            result_rows.append(row)
            completed_pairs.add(pair)
            save_run_results(result_rows)

            print()
            print(f"Selected epoch: {row['best_epoch']}")
            print(
                "Validation colors: "
                f"{row['final_validation_total_colors']} "
                "(target 60, ColPack 75)"
            )
            print(
                f"Test colors: {row['final_test_total_colors']} "
                "(target 60, ColPack 75)"
            )
            print(
                "Exact-target test graphs: "
                f"{row['final_test_exact_target_graphs']}/5"
            )
            print(f"Test valid: {row['final_test_all_valid']}")
            print()

    final_results_df = save_run_results(result_rows)
    aggregate_df = create_aggregate_summary(
        final_results_df
    )

    print()
    print("Week 19 objective comparison complete.")
    print("------------------------------------------")
    display_columns = [
        "objective",
        "num_completed_seeds",
        "mean_test_colors",
        "std_test_colors",
        "minimum_test_colors",
        "maximum_test_colors",
        "mean_test_gap_from_target",
        "mean_exact_target_graphs",
        "representative_seed",
        "representative_test_colors",
    ]
    print(
        aggregate_df[display_columns]
        .round(3)
        .to_string(index=False)
    )
    print()
    print(f"Saved run summary to: {OUTPUT_CSV}")
    print(
        f"Saved aggregate summary to: "
        f"{AGGREGATE_OUTPUT_CSV}"
    )
    print(f"Saved checkpoints under: {CHECKPOINT_ROOT}")


if __name__ == "__main__":
    main()