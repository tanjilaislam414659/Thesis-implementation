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
    / "week18_controlled_data_scaling_all_seeds_test_per_graph.csv"
)

GAP_BY_SEED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_gap_by_seed.csv"
)

GAP_AGGREGATE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_gap_aggregate.csv"
)

OVERALL_BY_SEED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_overall_by_seed.csv"
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

CONDITION_ORDER = {
    condition: index
    for index, condition in enumerate(CONDITIONS)
}

BASELINE_CONDITION = "train_20_baseline"
FULL_DATA_CONDITION = "train_125_plus105"

EXPECTED_SEEDS = [0, 1, 2, 3, 4]
EXPECTED_TEST_GRAPHS = 5
EXPECTED_TOTAL_RUNS = 20
EXPECTED_PER_GRAPH_ROWS = 100
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

    missing_columns = (
        required_columns
        - set(manifest_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return manifest_df


def load_training_summary() -> pd.DataFrame:
    if not TRAINING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Training summary not found: "
            f"{TRAINING_SUMMARY_PATH}"
        )

    training_df = pd.read_csv(
        TRAINING_SUMMARY_PATH
    )

    required_columns = {
        "condition",
        "seed",
        "final_test_total_colors",
        "final_test_target_colors",
        "final_test_colpack5_colors",
        "final_test_all_valid",
        "checkpoint_path",
    }

    missing_columns = (
        required_columns
        - set(training_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(training_df) != EXPECTED_TOTAL_RUNS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_RUNS} completed runs, "
            f"found {len(training_df)}."
        )

    duplicate_mask = training_df[
        [
            "condition",
            "seed",
        ]
    ].duplicated()

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate condition-seed rows found."
        )

    for condition in CONDITIONS:
        condition_df = training_df[
            training_df["condition"]
            == condition
        ]

        actual_seeds = sorted(
            condition_df[
                "seed"
            ].astype(int).tolist()
        )

        if actual_seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{condition}: expected seeds "
                f"{EXPECTED_SEEDS}, found {actual_seeds}."
            )

    return training_df


def get_frozen_test_manifest(
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    reference_df = (
        manifest_df[
            (manifest_df["condition"] == BASELINE_CONDITION)
            & (manifest_df["split"] == "test")
        ]
        .sort_values("graph_id")
        .reset_index(drop=True)
    )

    if len(reference_df) != EXPECTED_TEST_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_GRAPHS} test graphs, "
            f"found {len(reference_df)}."
        )

    reference_pairs = set(
        zip(
            reference_df[
                "graph_id"
            ].astype(str),
            reference_df[
                "source_pt_path"
            ].astype(str),
        )
    )

    for condition in CONDITIONS:
        condition_test_df = manifest_df[
            (manifest_df["condition"] == condition)
            & (manifest_df["split"] == "test")
        ]

        condition_pairs = set(
            zip(
                condition_test_df[
                    "graph_id"
                ].astype(str),
                condition_test_df[
                    "source_pt_path"
                ].astype(str),
            )
        )

        if condition_pairs != reference_pairs:
            raise ValueError(
                f"{condition}: test set is not frozen."
            )

    target_total = int(
        reference_df[
            "target_colors"
        ].sum()
    )

    colpack_total = int(
        reference_df[
            "best_colpack5_colors"
        ].sum()
    )

    if target_total != EXPECTED_TEST_TARGET_TOTAL:
        raise ValueError(
            f"Expected target total "
            f"{EXPECTED_TEST_TARGET_TOTAL}, "
            f"found {target_total}."
        )

    if colpack_total != EXPECTED_TEST_COLPACK_TOTAL:
        raise ValueError(
            f"Expected ColPack total "
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
        "condition",
        "seed",
        "input_dim",
        "hidden_channels",
        "out_channels",
        "best_epoch",
    }

    missing_keys = (
        required_keys
        - set(checkpoint.keys())
    )

    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing keys: "
            f"{sorted(missing_keys)}"
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
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model, checkpoint


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

    ordering = scores_to_ordering(
        predicted_scores
    )

    coloring = greedy_color_with_ordering(
        graph=graph,
        ordering=ordering,
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

    target_colors = scalar_to_int(
        data.selected_num_colors
    )

    colpack_colors = scalar_to_int(
        data.best_colpack5_colors
    )

    gap_level = scalar_to_int(
        data.gap_level
    )

    target_gap = (
        gnn_colors
        - target_colors
    )

    colpack_gap = (
        colpack_colors
        - target_colors
    )

    colors_saved = (
        colpack_colors
        - gnn_colors
    )

    if colpack_gap > 0:
        gap_closed_percentage = (
            colors_saved
            / colpack_gap
            * 100.0
        )
    else:
        gap_closed_percentage = 0.0

    return {
        "graph_id": str(
            data.graph_id
        ),
        "gap_level": gap_level,
        "num_nodes": int(
            data.num_nodes
        ),
        "target_colors": target_colors,
        "colpack5_colors": colpack_colors,
        "colpack_gap_from_target": colpack_gap,
        "gnn_colors": gnn_colors,
        "gnn_gap_from_target": target_gap,
        "colors_saved_vs_colpack5": colors_saved,
        "percentage_colpack_gap_closed": (
            gap_closed_percentage
        ),
        "reached_exact_target": (
            gnn_colors == target_colors
        ),
        "better_than_colpack5": (
            gnn_colors < colpack_colors
        ),
        "coloring_valid": valid,
    }


def evaluate_all_runs(
    training_df: pd.DataFrame,
    test_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for run in training_df.itertuples(
        index=False
    ):
        condition = str(
            run.condition
        )

        seed = int(
            run.seed
        )

        checkpoint_path = (
            PROJECT_ROOT
            / str(
                run.checkpoint_path
            )
        )

        model, checkpoint = load_model(
            checkpoint_path
        )

        if str(
            checkpoint["condition"]
        ) != condition:
            raise ValueError(
                f"{condition}, seed {seed}: "
                "checkpoint condition mismatch."
            )

        if int(
            checkpoint["seed"]
        ) != seed:
            raise ValueError(
                f"{condition}, seed {seed}: "
                "checkpoint seed mismatch."
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

            result = evaluate_graph(
                model=model,
                data=data,
            )

            result.update(
                {
                    "condition": condition,
                    "condition_label": (
                        CONDITION_LABELS[
                            condition
                        ]
                    ),
                    "seed": seed,
                    "best_epoch": int(
                        checkpoint[
                            "best_epoch"
                        ]
                    ),
                    "checkpoint_path": str(
                        checkpoint_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                }
            )

            rows.append(result)

    per_graph_df = pd.DataFrame(
        rows
    )

    if len(per_graph_df) != EXPECTED_PER_GRAPH_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_PER_GRAPH_ROWS} rows, "
            f"found {len(per_graph_df)}."
        )

    if not bool(
        per_graph_df[
            "coloring_valid"
        ].all()
    ):
        raise ValueError(
            "At least one invalid coloring was produced."
        )

    return per_graph_df


def validate_against_training_summary(
    per_graph_df: pd.DataFrame,
    training_df: pd.DataFrame,
) -> None:
    evaluated_totals = (
        per_graph_df
        .groupby(
            [
                "condition",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            evaluated_test_colors=(
                "gnn_colors",
                "sum",
            ),
            evaluated_target_colors=(
                "target_colors",
                "sum",
            ),
            evaluated_colpack_colors=(
                "colpack5_colors",
                "sum",
            ),
            evaluated_all_valid=(
                "coloring_valid",
                "all",
            ),
        )
    )

    recorded_totals = training_df[
        [
            "condition",
            "seed",
            "final_test_total_colors",
            "final_test_target_colors",
            "final_test_colpack5_colors",
            "final_test_all_valid",
        ]
    ].copy()

    comparison_df = evaluated_totals.merge(
        recorded_totals,
        on=[
            "condition",
            "seed",
        ],
        how="inner",
        validate="one_to_one",
    )

    if not (
        comparison_df[
            "evaluated_test_colors"
        ]
        == comparison_df[
            "final_test_total_colors"
        ]
    ).all():
        raise ValueError(
            "Re-evaluated test totals do not match "
            "the training summary."
        )

    if not (
        comparison_df[
            "evaluated_target_colors"
        ]
        == comparison_df[
            "final_test_target_colors"
        ]
    ).all():
        raise ValueError(
            "Target totals do not match."
        )

    if not (
        comparison_df[
            "evaluated_colpack_colors"
        ]
        == comparison_df[
            "final_test_colpack5_colors"
        ]
    ).all():
        raise ValueError(
            "ColPack totals do not match."
        )

    if not bool(
        comparison_df[
            "evaluated_all_valid"
        ].all()
    ):
        raise ValueError(
            "At least one re-evaluated coloring is invalid."
        )


def build_gap_by_seed(
    per_graph_df: pd.DataFrame,
) -> pd.DataFrame:
    gap_by_seed_df = (
        per_graph_df
        .groupby(
            [
                "condition",
                "condition_label",
                "seed",
                "gap_level",
            ],
            as_index=False,
        )
        .agg(
            graph_id=(
                "graph_id",
                "first",
            ),
            num_nodes=(
                "num_nodes",
                "first",
            ),
            target_colors=(
                "target_colors",
                "first",
            ),
            colpack5_colors=(
                "colpack5_colors",
                "first",
            ),
            gnn_colors=(
                "gnn_colors",
                "first",
            ),
            gnn_gap_from_target=(
                "gnn_gap_from_target",
                "first",
            ),
            colors_saved_vs_colpack5=(
                "colors_saved_vs_colpack5",
                "first",
            ),
            percentage_colpack_gap_closed=(
                "percentage_colpack_gap_closed",
                "first",
            ),
            reached_exact_target=(
                "reached_exact_target",
                "first",
            ),
            coloring_valid=(
                "coloring_valid",
                "all",
            ),
        )
    )

    gap_by_seed_df[
        "_condition_order"
    ] = gap_by_seed_df[
        "condition"
    ].map(
        CONDITION_ORDER
    )

    gap_by_seed_df = (
        gap_by_seed_df
        .sort_values(
            [
                "_condition_order",
                "seed",
                "gap_level",
            ]
        )
        .drop(
            columns=[
                "_condition_order"
            ]
        )
        .reset_index(drop=True)
    )

    return gap_by_seed_df


def build_gap_aggregate(
    gap_by_seed_df: pd.DataFrame,
) -> pd.DataFrame:
    aggregate_df = (
        gap_by_seed_df
        .groupby(
            [
                "condition",
                "condition_label",
                "gap_level",
            ],
            as_index=False,
        )
        .agg(
            num_seeds=(
                "seed",
                "count",
            ),
            num_nodes=(
                "num_nodes",
                "first",
            ),
            target_colors=(
                "target_colors",
                "first",
            ),
            colpack5_colors=(
                "colpack5_colors",
                "first",
            ),
            mean_gnn_colors=(
                "gnn_colors",
                "mean",
            ),
            std_gnn_colors=(
                "gnn_colors",
                lambda values: float(
                    values.std(ddof=0)
                ),
            ),
            minimum_gnn_colors=(
                "gnn_colors",
                "min",
            ),
            maximum_gnn_colors=(
                "gnn_colors",
                "max",
            ),
            mean_gap_from_target=(
                "gnn_gap_from_target",
                "mean",
            ),
            mean_colors_saved_vs_colpack5=(
                "colors_saved_vs_colpack5",
                "mean",
            ),
            mean_percentage_colpack_gap_closed=(
                "percentage_colpack_gap_closed",
                "mean",
            ),
            exact_target_runs=(
                "reached_exact_target",
                "sum",
            ),
            all_valid=(
                "coloring_valid",
                "all",
            ),
        )
    )

    baseline_means = (
        aggregate_df[
            aggregate_df[
                "condition"
            ]
            == BASELINE_CONDITION
        ][
            [
                "gap_level",
                "mean_gnn_colors",
            ]
        ]
        .rename(
            columns={
                "mean_gnn_colors": (
                    "baseline_mean_gnn_colors"
                )
            }
        )
    )

    aggregate_df = aggregate_df.merge(
        baseline_means,
        on="gap_level",
        how="left",
        validate="many_to_one",
    )

    aggregate_df[
        "mean_improvement_vs_baseline"
    ] = (
        aggregate_df[
            "baseline_mean_gnn_colors"
        ]
        - aggregate_df[
            "mean_gnn_colors"
        ]
    )

    aggregate_df[
        "_condition_order"
    ] = aggregate_df[
        "condition"
    ].map(
        CONDITION_ORDER
    )

    aggregate_df = (
        aggregate_df
        .sort_values(
            [
                "_condition_order",
                "gap_level",
            ]
        )
        .drop(
            columns=[
                "_condition_order"
            ]
        )
        .reset_index(drop=True)
    )

    return aggregate_df


def build_overall_by_seed(
    per_graph_df: pd.DataFrame,
) -> pd.DataFrame:
    overall_df = (
        per_graph_df
        .groupby(
            [
                "condition",
                "condition_label",
                "seed",
            ],
            as_index=False,
        )
        .agg(
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
            all_valid=(
                "coloring_valid",
                "all",
            ),
        )
    )

    overall_df[
        "percentage_colpack_gap_closed"
    ] = (
        overall_df[
            "test_colors_saved_vs_colpack5"
        ]
        / (
            overall_df[
                "test_colpack5_colors"
            ]
            - overall_df[
                "test_target_colors"
            ]
        )
        * 100.0
    )

    overall_df[
        "_condition_order"
    ] = overall_df[
        "condition"
    ].map(
        CONDITION_ORDER
    )

    overall_df = (
        overall_df
        .sort_values(
            [
                "_condition_order",
                "seed",
            ]
        )
        .drop(
            columns=[
                "_condition_order"
            ]
        )
        .reset_index(drop=True)
    )

    return overall_df


def print_results(
    gap_by_seed_df: pd.DataFrame,
    gap_aggregate_df: pd.DataFrame,
    overall_by_seed_df: pd.DataFrame,
) -> None:
    print(
        "Week 18 controlled all-seed gap analysis"
    )
    print(
        "----------------------------------------"
    )
    print()

    print(
        "Five-component test graph across all seeds:"
    )

    gap5_df = gap_by_seed_df[
        gap_by_seed_df[
            "gap_level"
        ]
        == 5
    ]

    gap5_pivot = (
        gap5_df
        .pivot(
            index="seed",
            columns="condition_label",
            values="gnn_colors",
        )
        .reset_index()
    )

    ordered_gap5_columns = [
        "seed",
        "20 graphs",
        "32 graphs",
        "44 graphs",
        "125 graphs",
    ]

    print(
        gap5_pivot[
            ordered_gap5_columns
        ].to_string(
            index=False
        )
    )
    print()

    print(
        "Gap-level aggregate across five seeds:"
    )

    aggregate_display = (
        gap_aggregate_df[
            [
                "condition_label",
                "gap_level",
                "target_colors",
                "colpack5_colors",
                "mean_gnn_colors",
                "std_gnn_colors",
                "minimum_gnn_colors",
                "maximum_gnn_colors",
                "exact_target_runs",
                "mean_improvement_vs_baseline",
            ]
        ]
    )

    print(
        aggregate_display
        .round(3)
        .to_string(
            index=False
        )
    )
    print()

    print(
        "Overall test totals for every seed:"
    )

    overall_pivot = (
        overall_by_seed_df
        .pivot(
            index="seed",
            columns="condition_label",
            values="test_gnn_colors",
        )
        .reset_index()
    )

    print(
        overall_pivot[
            ordered_gap5_columns
        ].to_string(
            index=False
        )
    )
    print()

    baseline_gap5 = gap_aggregate_df[
        (gap_aggregate_df["condition"] == BASELINE_CONDITION)
        & (gap_aggregate_df["gap_level"] == 5)
    ].iloc[0]

    full_gap5 = gap_aggregate_df[
        (gap_aggregate_df["condition"] == FULL_DATA_CONDITION)
        & (gap_aggregate_df["gap_level"] == 5)
    ].iloc[0]

    print(
        "Focused gap-5 comparison:"
    )
    print(
        f"  Baseline mean colors: "
        f"{baseline_gap5['mean_gnn_colors']:.3f}"
    )
    print(
        f"  Full-data mean colors: "
        f"{full_gap5['mean_gnn_colors']:.3f}"
    )
    print(
        f"  Mean improvement: "
        f"{full_gap5['mean_improvement_vs_baseline']:.3f}"
    )
    print(
        f"  Baseline exact runs: "
        f"{int(baseline_gap5['exact_target_runs'])}/5"
    )
    print(
        f"  Full-data exact runs: "
        f"{int(full_gap5['exact_target_runs'])}/5"
    )


def main() -> None:
    manifest_df = load_manifest()

    training_df = (
        load_training_summary()
    )

    test_manifest_df = (
        get_frozen_test_manifest(
            manifest_df
        )
    )

    per_graph_df = evaluate_all_runs(
        training_df=training_df,
        test_manifest_df=test_manifest_df,
    )

    validate_against_training_summary(
        per_graph_df=per_graph_df,
        training_df=training_df,
    )

    per_graph_df[
        "_condition_order"
    ] = per_graph_df[
        "condition"
    ].map(
        CONDITION_ORDER
    )

    per_graph_df = (
        per_graph_df
        .sort_values(
            [
                "_condition_order",
                "seed",
                "gap_level",
            ]
        )
        .drop(
            columns=[
                "_condition_order"
            ]
        )
        .reset_index(drop=True)
    )

    gap_by_seed_df = build_gap_by_seed(
        per_graph_df
    )

    gap_aggregate_df = (
        build_gap_aggregate(
            gap_by_seed_df
        )
    )

    overall_by_seed_df = (
        build_overall_by_seed(
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

    gap_by_seed_df.to_csv(
        GAP_BY_SEED_OUTPUT_PATH,
        index=False,
    )

    gap_aggregate_df.to_csv(
        GAP_AGGREGATE_OUTPUT_PATH,
        index=False,
    )

    overall_by_seed_df.to_csv(
        OVERALL_BY_SEED_OUTPUT_PATH,
        index=False,
    )

    print_results(
        gap_by_seed_df=gap_by_seed_df,
        gap_aggregate_df=gap_aggregate_df,
        overall_by_seed_df=overall_by_seed_df,
    )

    print()
    print(
        f"Saved per-graph results to: "
        f"{PER_GRAPH_OUTPUT_PATH}"
    )
    print(
        f"Saved gap-by-seed results to: "
        f"{GAP_BY_SEED_OUTPUT_PATH}"
    )
    print(
        f"Saved gap aggregate to: "
        f"{GAP_AGGREGATE_OUTPUT_PATH}"
    )
    print(
        f"Saved overall seed summary to: "
        f"{OVERALL_BY_SEED_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()