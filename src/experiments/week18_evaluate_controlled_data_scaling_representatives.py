from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import torch

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

TRAINING_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_gnn_training_summary.csv"
)

PER_GRAPH_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_representative_test_per_graph.csv"
)

GAP_SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_representative_gap_summary.csv"
)

COMPARISON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_representative_comparison_summary.csv"
)


CONDITIONS = [
    "train_20_baseline",
    "train_32_plus12",
    "train_44_plus24",
    "train_125_plus105",
]

CONDITION_LABELS = {
    "train_20_baseline": "20 graphs",
    "train_32_plus12": "32 graphs",
    "train_44_plus24": "44 graphs",
    "train_125_plus105": "125 graphs",
}

BASELINE_CONDITION = "train_20_baseline"

EXPECTED_TEST_GRAPH_COUNT = 5
EXPECTED_TEST_TARGET_TOTAL = 60
EXPECTED_TEST_COLPACK_TOTAL = 75


def load_torch_object(path: Path):
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
        "source_pt_path",
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
            f"Manifest missing columns: {sorted(missing)}"
        )

    return manifest_df


def load_training_summary() -> pd.DataFrame:
    if not TRAINING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Training summary not found: "
            f"{TRAINING_SUMMARY_PATH}"
        )

    summary_df = pd.read_csv(
        TRAINING_SUMMARY_PATH
    )

    required_columns = {
        "condition",
        "seed",
        "final_validation_total_colors",
        "final_validation_loss_best_model",
        "final_test_total_colors",
        "checkpoint_path",
    }

    missing = required_columns - set(
        summary_df.columns
    )

    if missing:
        raise ValueError(
            f"Training summary missing columns: "
            f"{sorted(missing)}"
        )

    for condition in CONDITIONS:
        condition_rows = summary_df[
            summary_df["condition"]
            == condition
        ]

        if len(condition_rows) != 5:
            raise ValueError(
                f"{condition}: expected 5 completed seeds, "
                f"found {len(condition_rows)}."
            )

    return summary_df


def select_representative_models(
    training_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one model per condition using validation only:

    1. Fewest validation colors.
    2. Lowest validation loss.
    3. Lowest seed as deterministic final tie-breaker.
    """
    representative_rows = []

    for condition in CONDITIONS:
        condition_df = training_df[
            training_df["condition"]
            == condition
        ].copy()

        representative = (
            condition_df
            .sort_values(
                [
                    "final_validation_total_colors",
                    "final_validation_loss_best_model",
                    "seed",
                ]
            )
            .iloc[0]
            .copy()
        )

        representative_rows.append(
            representative
        )

    representative_df = pd.DataFrame(
        representative_rows
    ).reset_index(drop=True)

    return representative_df


def get_frozen_test_manifest(
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Verify that every training condition uses exactly the same test
    graph IDs and source files.
    """
    reference_df = (
        manifest_df[
            (manifest_df["condition"] == BASELINE_CONDITION)
            & (manifest_df["split"] == "test")
        ]
        .sort_values("graph_id")
        .reset_index(drop=True)
    )

    if len(reference_df) != EXPECTED_TEST_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TEST_GRAPH_COUNT} test graphs, "
            f"found {len(reference_df)}."
        )

    reference_pairs = set(
        zip(
            reference_df["graph_id"].astype(str),
            reference_df["source_pt_path"].astype(str),
        )
    )

    for condition in CONDITIONS:
        condition_test_df = manifest_df[
            (manifest_df["condition"] == condition)
            & (manifest_df["split"] == "test")
        ]

        condition_pairs = set(
            zip(
                condition_test_df["graph_id"].astype(str),
                condition_test_df["source_pt_path"].astype(str),
            )
        )

        if condition_pairs != reference_pairs:
            raise ValueError(
                f"{condition}: test set is not identical "
                "to the baseline test set."
            )

    target_total = int(
        reference_df["target_colors"].sum()
    )

    colpack_total = int(
        reference_df["best_colpack5_colors"].sum()
    )

    if target_total != EXPECTED_TEST_TARGET_TOTAL:
        raise ValueError(
            f"Expected test target total "
            f"{EXPECTED_TEST_TARGET_TOTAL}, "
            f"found {target_total}."
        )

    if colpack_total != EXPECTED_TEST_COLPACK_TOTAL:
        raise ValueError(
            f"Expected test ColPack total "
            f"{EXPECTED_TEST_COLPACK_TOTAL}, "
            f"found {colpack_total}."
        )

    return reference_df


def pyg_data_to_networkx_graph(data) -> nx.Graph:
    graph = nx.Graph()

    graph.add_nodes_from(
        range(data.num_nodes)
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


def load_model(
    checkpoint_path: Path,
) -> tuple[GNNNodeScorer, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = load_torch_object(
        checkpoint_path
    )

    required_keys = {
        "model_state_dict",
        "input_dim",
        "hidden_channels",
        "out_channels",
        "seed",
        "condition",
        "best_epoch",
    }

    missing = required_keys - set(
        checkpoint.keys()
    )

    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing keys: "
            f"{sorted(missing)}"
        )

    model = GNNNodeScorer(
        in_channels=int(
            checkpoint["input_dim"]
        ),
        hidden_channels=int(
            checkpoint["hidden_channels"]
        ),
        out_channels=int(
            checkpoint["out_channels"]
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def describe_graph(data) -> str:
    gap_level = scalar_to_int(
        data.gap_level
    )

    if hasattr(data, "base_cycle_size"):
        base_size = scalar_to_int(
            data.base_cycle_size
        )

        if gap_level == 1:
            return f"C{base_size}^2"

        return (
            f"join of {gap_level} copies "
            f"of C{base_size}^2"
        )

    if hasattr(
        data,
        "component_cycle_sizes",
    ):
        component_sizes = str(
            data.component_cycle_sizes
        )

        return (
            f"mixed join [{component_sizes}]"
        )

    return "unknown construction"


def evaluate_graph(
    model: GNNNodeScorer,
    data,
) -> dict[str, object]:
    graph = pyg_data_to_networkx_graph(
        data
    )

    with torch.no_grad():
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

    coloring_valid = bool(
        is_valid_coloring(
            graph,
            coloring,
        )
    )

    target_colors = scalar_to_int(
        data.selected_num_colors
    )

    colpack_colors = scalar_to_int(
        data.best_colpack5_colors
    )

    gap_level = scalar_to_int(
        data.gap_level
    )

    gnn_gap_from_target = (
        gnn_colors - target_colors
    )

    colpack_gap_from_target = (
        colpack_colors - target_colors
    )

    colors_saved_vs_colpack = (
        colpack_colors - gnn_colors
    )

    if colpack_gap_from_target > 0:
        percentage_colpack_gap_closed = (
            colors_saved_vs_colpack
            / colpack_gap_from_target
            * 100.0
        )
    else:
        percentage_colpack_gap_closed = 0.0

    return {
        "graph_id": str(
            data.graph_id
        ),
        "construction": describe_graph(
            data
        ),
        "gap_level": gap_level,
        "num_nodes": int(
            data.num_nodes
        ),
        "target_colors": target_colors,
        "colpack5_colors": colpack_colors,
        "colpack_gap_from_target": (
            colpack_gap_from_target
        ),
        "gnn_colors": gnn_colors,
        "gnn_gap_from_target": (
            gnn_gap_from_target
        ),
        "colors_saved_vs_colpack5": (
            colors_saved_vs_colpack
        ),
        "percentage_colpack_gap_closed": (
            percentage_colpack_gap_closed
        ),
        "reached_exact_target": (
            gnn_colors == target_colors
        ),
        "better_than_colpack5": (
            gnn_colors < colpack_colors
        ),
        "coloring_valid": coloring_valid,
    }


def evaluate_representatives(
    representative_df: pd.DataFrame,
    test_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for representative in (
        representative_df.itertuples(
            index=False
        )
    ):
        condition = str(
            representative.condition
        )

        seed = int(
            representative.seed
        )

        checkpoint_path = (
            PROJECT_ROOT
            / str(
                representative.checkpoint_path
            )
        )

        model, checkpoint = load_model(
            checkpoint_path
        )

        if str(
            checkpoint["condition"]
        ) != condition:
            raise ValueError(
                f"{condition}: checkpoint condition mismatch."
            )

        if int(
            checkpoint["seed"]
        ) != seed:
            raise ValueError(
                f"{condition}: checkpoint seed mismatch."
            )

        for test_row in (
            test_manifest_df.itertuples(
                index=False
            )
        ):
            data_path = (
                PROJECT_ROOT
                / str(
                    test_row.source_pt_path
                )
            )

            data = load_torch_object(
                data_path
            )

            graph_result = evaluate_graph(
                model=model,
                data=data,
            )

            graph_result.update(
                {
                    "condition": condition,
                    "condition_label": (
                        CONDITION_LABELS[
                            condition
                        ]
                    ),
                    "representative_seed": (
                        seed
                    ),
                    "representative_best_epoch": int(
                        checkpoint[
                            "best_epoch"
                        ]
                    ),
                    "representative_validation_colors": int(
                        representative.final_validation_total_colors
                    ),
                    "representative_validation_loss": float(
                        representative.final_validation_loss_best_model
                    ),
                    "checkpoint_path": str(
                        checkpoint_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                }
            )

            rows.append(
                graph_result
            )

    per_graph_df = pd.DataFrame(
        rows
    )

    expected_rows = (
        len(CONDITIONS)
        * EXPECTED_TEST_GRAPH_COUNT
    )

    if len(per_graph_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} evaluation rows, "
            f"found {len(per_graph_df)}."
        )

    if not bool(
        per_graph_df[
            "coloring_valid"
        ].all()
    ):
        raise ValueError(
            "At least one representative model produced "
            "an invalid coloring."
        )

    return per_graph_df


def add_baseline_comparisons(
    per_graph_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_df = (
        per_graph_df[
            per_graph_df["condition"]
            == BASELINE_CONDITION
        ][
            [
                "graph_id",
                "gnn_colors",
                "gnn_gap_from_target",
            ]
        ]
        .rename(
            columns={
                "gnn_colors": (
                    "baseline_representative_colors"
                ),
                "gnn_gap_from_target": (
                    "baseline_gap_from_target"
                ),
            }
        )
    )

    result_df = per_graph_df.merge(
        baseline_df,
        on="graph_id",
        how="left",
        validate="many_to_one",
    )

    result_df[
        "colors_improved_vs_baseline"
    ] = (
        result_df[
            "baseline_representative_colors"
        ]
        - result_df[
            "gnn_colors"
        ]
    )

    return result_df


def build_gap_summary(
    per_graph_df: pd.DataFrame,
) -> pd.DataFrame:
    gap_summary_df = (
        per_graph_df
        .groupby(
            [
                "condition",
                "condition_label",
                "representative_seed",
                "gap_level",
            ],
            as_index=False,
        )
        .agg(
            num_graphs=(
                "graph_id",
                "count",
            ),
            total_nodes=(
                "num_nodes",
                "sum",
            ),
            total_target_colors=(
                "target_colors",
                "sum",
            ),
            total_colpack5_colors=(
                "colpack5_colors",
                "sum",
            ),
            total_gnn_colors=(
                "gnn_colors",
                "sum",
            ),
            total_gnn_gap_from_target=(
                "gnn_gap_from_target",
                "sum",
            ),
            total_colors_saved_vs_colpack5=(
                "colors_saved_vs_colpack5",
                "sum",
            ),
            mean_percentage_colpack_gap_closed=(
                "percentage_colpack_gap_closed",
                "mean",
            ),
            exact_target_graphs=(
                "reached_exact_target",
                "sum",
            ),
            all_valid=(
                "coloring_valid",
                "all",
            ),
        )
    )

    gap_summary_df[
        "condition_order"
    ] = gap_summary_df[
        "condition"
    ].map(
        {
            condition: index
            for index, condition
            in enumerate(CONDITIONS)
        }
    )

    gap_summary_df = (
        gap_summary_df
        .sort_values(
            [
                "condition_order",
                "gap_level",
            ]
        )
        .drop(
            columns=[
                "condition_order"
            ]
        )
        .reset_index(drop=True)
    )

    return gap_summary_df


def build_comparison_summary(
    per_graph_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_df = (
        per_graph_df
        .groupby(
            [
                "condition",
                "condition_label",
                "representative_seed",
                "representative_validation_colors",
                "representative_validation_loss",
            ],
            as_index=False,
        )
        .agg(
            num_test_graphs=(
                "graph_id",
                "count",
            ),
            test_target_colors=(
                "target_colors",
                "sum",
            ),
            test_colpack5_colors=(
                "colpack5_colors",
                "sum",
            ),
            test_gnn_colors=(
                "gnn_colors",
                "sum",
            ),
            test_gap_from_target=(
                "gnn_gap_from_target",
                "sum",
            ),
            test_colors_saved_vs_colpack5=(
                "colors_saved_vs_colpack5",
                "sum",
            ),
            exact_target_graphs=(
                "reached_exact_target",
                "sum",
            ),
            better_than_colpack_graphs=(
                "better_than_colpack5",
                "sum",
            ),
            total_colors_improved_vs_baseline=(
                "colors_improved_vs_baseline",
                "sum",
            ),
            all_test_colorings_valid=(
                "coloring_valid",
                "all",
            ),
        )
    )

    colpack_gap = (
        summary_df[
            "test_colpack5_colors"
        ]
        - summary_df[
            "test_target_colors"
        ]
    )

    summary_df[
        "percentage_colpack_gap_closed"
    ] = (
        summary_df[
            "test_colors_saved_vs_colpack5"
        ]
        / colpack_gap
        * 100.0
    )

    summary_df[
        "condition_order"
    ] = summary_df[
        "condition"
    ].map(
        {
            condition: index
            for index, condition
            in enumerate(CONDITIONS)
        }
    )

    summary_df = (
        summary_df
        .sort_values(
            "condition_order"
        )
        .drop(
            columns=[
                "condition_order"
            ]
        )
        .reset_index(drop=True)
    )

    return summary_df


def print_results(
    representative_df: pd.DataFrame,
    per_graph_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    print(
        "Week 18 controlled representative-model analysis"
    )
    print(
        "------------------------------------------------"
    )
    print()

    print(
        "Representative models selected using validation only:"
    )

    representative_display = (
        representative_df[
            [
                "condition",
                "seed",
                "final_validation_total_colors",
                "final_validation_loss_best_model",
                "final_test_total_colors",
            ]
        ]
        .rename(
            columns={
                "seed": "representative_seed",
                "final_validation_total_colors": (
                    "validation_colors"
                ),
                "final_validation_loss_best_model": (
                    "validation_loss"
                ),
                "final_test_total_colors": (
                    "recorded_test_colors"
                ),
            }
        )
    )

    print(
        representative_display
        .round(6)
        .to_string(index=False)
    )
    print()

    print(
        "Representative test colors by gap level:"
    )

    pivot_df = (
        per_graph_df
        .pivot(
            index=[
                "gap_level",
                "construction",
                "target_colors",
                "colpack5_colors",
            ],
            columns="condition_label",
            values="gnn_colors",
        )
        .reset_index()
    )

    ordered_columns = [
        "gap_level",
        "construction",
        "target_colors",
        "colpack5_colors",
        "20 graphs",
        "32 graphs",
        "44 graphs",
        "125 graphs",
    ]

    pivot_df = pivot_df[
        ordered_columns
    ].sort_values(
        "gap_level"
    )

    print(
        pivot_df.to_string(
            index=False
        )
    )
    print()

    print(
        "Overall representative comparison:"
    )

    comparison_display = (
        comparison_df[
            [
                "condition_label",
                "representative_seed",
                "test_target_colors",
                "test_colpack5_colors",
                "test_gnn_colors",
                "test_gap_from_target",
                "test_colors_saved_vs_colpack5",
                "percentage_colpack_gap_closed",
                "exact_target_graphs",
                "total_colors_improved_vs_baseline",
            ]
        ]
    )

    print(
        comparison_display
        .round(3)
        .to_string(index=False)
    )


def main() -> None:
    manifest_df = load_manifest()

    training_df = (
        load_training_summary()
    )

    representative_df = (
        select_representative_models(
            training_df
        )
    )

    test_manifest_df = (
        get_frozen_test_manifest(
            manifest_df
        )
    )

    per_graph_df = (
        evaluate_representatives(
            representative_df=representative_df,
            test_manifest_df=test_manifest_df,
        )
    )

    per_graph_df = (
        add_baseline_comparisons(
            per_graph_df
        )
    )

    condition_order = {
        condition: index
        for index, condition
        in enumerate(CONDITIONS)
    }

    per_graph_df[
        "_condition_order"
    ] = per_graph_df[
        "condition"
    ].map(
        condition_order
    )

    per_graph_df = (
        per_graph_df
        .sort_values(
            [
                "_condition_order",
                "gap_level",
                "graph_id",
            ]
        )
        .drop(
            columns=[
                "_condition_order"
            ]
        )
        .reset_index(drop=True)
    )

    gap_summary_df = (
        build_gap_summary(
            per_graph_df
        )
    )

    comparison_df = (
        build_comparison_summary(
            per_graph_df
        )
    )

    PER_GRAPH_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_graph_df.to_csv(
        PER_GRAPH_OUTPUT_PATH,
        index=False,
    )

    gap_summary_df.to_csv(
        GAP_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    print_results(
        representative_df=representative_df,
        per_graph_df=per_graph_df,
        comparison_df=comparison_df,
    )

    print()
    print(
        f"Saved per-graph results to: "
        f"{PER_GRAPH_OUTPUT_PATH}"
    )
    print(
        f"Saved gap-level summary to: "
        f"{GAP_SUMMARY_OUTPUT_PATH}"
    )
    print(
        f"Saved comparison summary to: "
        f"{COMPARISON_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()