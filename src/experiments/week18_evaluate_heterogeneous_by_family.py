from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import torch

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

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gnn_node_scorer"
    / "week18_heterogeneous_gnn_runs"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
)

PER_GRAPH_OUTPUT = (
    OUTPUT_DIR
    / "week18_heterogeneous_test_per_graph_all_seeds.csv"
)

FAMILY_BY_SEED_OUTPUT = (
    OUTPUT_DIR
    / "week18_heterogeneous_test_family_by_seed.csv"
)

FAMILY_AGGREGATE_OUTPUT = (
    OUTPUT_DIR
    / "week18_heterogeneous_test_family_aggregate.csv"
)

GRAPH_STABILITY_OUTPUT = (
    OUTPUT_DIR
    / "week18_heterogeneous_test_graph_stability.csv"
)

SEED_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "week18_heterogeneous_test_seed_summary.csv"
)


SEEDS = [0, 1, 2, 3, 4]

EXPECTED_TEST_GRAPHS = 12

EXPECTED_TEST_FAMILY_COUNTS = {
    "crown": 4,
    "erdos_renyi": 1,
    "stochastic_block_model": 6,
    "watts_strogatz": 1,
}


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


def load_test_graphs() -> list:
    """
    Load and validate the Week 18 test graphs.
    """
    if not PYG_DATA_DIR.exists():
        raise FileNotFoundError(
            f"PyG data directory not found: {PYG_DATA_DIR}"
        )

    dataset = load_all_pyg_graphs(
        PYG_DATA_DIR
    )

    grouped = group_dataset_by_split(
        dataset
    )

    test_graphs = grouped.get(
        "test",
        [],
    )

    if len(test_graphs) != EXPECTED_TEST_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_GRAPHS} test graphs, "
            f"found {len(test_graphs)}."
        )

    graph_ids = [
        str(data.graph_id)
        for data in test_graphs
    ]

    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError(
            "Duplicate graph IDs found in the test set."
        )

    actual_family_counts = (
        pd.Series(
            [
                str(data.graph_family)
                for data in test_graphs
            ]
        )
        .value_counts()
        .to_dict()
    )

    if actual_family_counts != EXPECTED_TEST_FAMILY_COUNTS:
        raise ValueError(
            "Unexpected test family distribution.\n"
            f"Expected: {EXPECTED_TEST_FAMILY_COUNTS}\n"
            f"Found: {actual_family_counts}"
        )

    return sorted(
        test_graphs,
        key=lambda data: str(data.graph_id),
    )


def load_model_for_seed(
    seed: int,
) -> tuple[GNNNodeScorer, dict[str, object]]:
    """
    Load the validation-selected checkpoint for one seed.
    """
    checkpoint_path = (
        CHECKPOINT_DIR
        / (
            "week18_best_gnn_node_scorer_"
            f"seed_{seed}.pt"
        )
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for seed {seed}: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    required_keys = {
        "model_state_dict",
        "input_dim",
        "hidden_channels",
        "out_channels",
        "seed",
        "best_epoch",
        "best_validation_total_colors",
        "best_validation_loss",
    }

    missing_keys = (
        required_keys
        - set(checkpoint.keys())
    )

    if missing_keys:
        raise ValueError(
            f"Seed {seed} checkpoint is missing keys: "
            f"{sorted(missing_keys)}"
        )

    checkpoint_seed = int(
        checkpoint["seed"]
    )

    if checkpoint_seed != seed:
        raise ValueError(
            f"Expected checkpoint seed {seed}, "
            f"found {checkpoint_seed}."
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


def evaluate_graph(
    model: GNNNodeScorer,
    data,
    seed: int,
    checkpoint: dict[str, object],
) -> dict[str, object]:
    """
    Evaluate one model checkpoint on one test graph.
    """
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

    if len(learned_ordering) != int(
        data.num_nodes
    ):
        raise ValueError(
            f"{data.graph_id}: learned ordering has "
            f"{len(learned_ordering)} vertices, expected "
            f"{data.num_nodes}."
        )

    if set(learned_ordering) != set(
        range(int(data.num_nodes))
    ):
        raise ValueError(
            f"{data.graph_id}: learned ordering is not "
            "a complete vertex permutation."
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
        data.selected_num_colors
    )

    best_colpack5_colors = int(
        data.best_colpack5_colors
    )

    if target_colors != best_colpack5_colors:
        raise ValueError(
            f"{data.graph_id}: selected target uses "
            f"{target_colors} colors, but best ColPack-5 "
            f"uses {best_colpack5_colors}."
        )

    gap_from_target = (
        gnn_colors - target_colors
    )

    return {
        "seed": seed,
        "best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "best_validation_total_colors": int(
            checkpoint[
                "best_validation_total_colors"
            ]
        ),
        "best_validation_loss": float(
            checkpoint[
                "best_validation_loss"
            ]
        ),
        "graph_id": str(
            data.graph_id
        ),
        "family": str(
            data.graph_family
        ),
        "num_vertices": int(
            data.num_nodes
        ),
        "num_edges": int(
            data.edge_index.shape[1] // 2
        ),
        "selected_teacher_ordering": str(
            data.selected_teacher_ordering
        ),
        "ordering_gap": int(
            data.ordering_gap
        ),
        "gnn_colors": gnn_colors,
        "target_colors": target_colors,
        "best_colpack5_colors": (
            best_colpack5_colors
        ),
        "gap_from_target": gap_from_target,
        "exact_match": (
            gap_from_target == 0
        ),
        "better_than_target": (
            gap_from_target < 0
        ),
        "worse_than_target": (
            gap_from_target > 0
        ),
        "valid": valid,
    }


def create_family_by_seed_table(
    per_graph_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize each graph family separately for every seed.
    """
    summary = (
        per_graph_df
        .groupby(
            [
                "seed",
                "family",
            ],
            as_index=False,
        )
        .agg(
            num_graphs=(
                "graph_id",
                "nunique",
            ),
            total_gnn_colors=(
                "gnn_colors",
                "sum",
            ),
            total_target_colors=(
                "target_colors",
                "sum",
            ),
            total_gap_from_target=(
                "gap_from_target",
                "sum",
            ),
            average_gap_per_graph=(
                "gap_from_target",
                "mean",
            ),
            maximum_gap_on_one_graph=(
                "gap_from_target",
                "max",
            ),
            exact_matches=(
                "exact_match",
                "sum",
            ),
            better_than_target_count=(
                "better_than_target",
                "sum",
            ),
            worse_than_target_count=(
                "worse_than_target",
                "sum",
            ),
            all_valid=(
                "valid",
                "all",
            ),
        )
    )

    summary["exact_match_rate"] = (
        summary["exact_matches"]
        / summary["num_graphs"]
    )

    return summary.sort_values(
        [
            "family",
            "seed",
        ]
    ).reset_index(drop=True)


def create_family_aggregate_table(
    per_graph_df: pd.DataFrame,
    family_by_seed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate family performance across all five seeds.
    """
    seed_level = (
        family_by_seed_df
        .groupby(
            "family",
            as_index=False,
        )
        .agg(
            num_seeds=(
                "seed",
                "nunique",
            ),
            num_graphs=(
                "num_graphs",
                "first",
            ),
            target_colors_per_seed=(
                "total_target_colors",
                "first",
            ),
            mean_gnn_colors_per_seed=(
                "total_gnn_colors",
                "mean",
            ),
            std_gnn_colors_per_seed=(
                "total_gnn_colors",
                "std",
            ),
            minimum_gnn_colors_per_seed=(
                "total_gnn_colors",
                "min",
            ),
            maximum_gnn_colors_per_seed=(
                "total_gnn_colors",
                "max",
            ),
            mean_total_gap_per_seed=(
                "total_gap_from_target",
                "mean",
            ),
            std_total_gap_per_seed=(
                "total_gap_from_target",
                "std",
            ),
            minimum_total_gap_per_seed=(
                "total_gap_from_target",
                "min",
            ),
            maximum_total_gap_per_seed=(
                "total_gap_from_target",
                "max",
            ),
            mean_exact_matches_per_seed=(
                "exact_matches",
                "mean",
            ),
            all_valid=(
                "all_valid",
                "all",
            ),
        )
    )

    observation_level = (
        per_graph_df
        .groupby(
            "family",
            as_index=False,
        )
        .agg(
            total_seed_graph_evaluations=(
                "graph_id",
                "count",
            ),
            exact_match_evaluations=(
                "exact_match",
                "sum",
            ),
            better_than_target_evaluations=(
                "better_than_target",
                "sum",
            ),
            worse_than_target_evaluations=(
                "worse_than_target",
                "sum",
            ),
            average_gap_per_seed_graph=(
                "gap_from_target",
                "mean",
            ),
            maximum_observed_gap=(
                "gap_from_target",
                "max",
            ),
        )
    )

    aggregate = seed_level.merge(
        observation_level,
        on="family",
        how="inner",
        validate="one_to_one",
    )

    aggregate["exact_match_rate"] = (
        aggregate["exact_match_evaluations"]
        / aggregate[
            "total_seed_graph_evaluations"
        ]
    )

    return aggregate.sort_values(
        "family"
    ).reset_index(drop=True)


def create_graph_stability_table(
    per_graph_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Show how stable each test graph is across the five seeds.
    """
    stability = (
        per_graph_df
        .groupby(
            [
                "graph_id",
                "family",
            ],
            as_index=False,
        )
        .agg(
            num_vertices=(
                "num_vertices",
                "first",
            ),
            selected_teacher_ordering=(
                "selected_teacher_ordering",
                "first",
            ),
            ordering_gap=(
                "ordering_gap",
                "first",
            ),
            target_colors=(
                "target_colors",
                "first",
            ),
            mean_gnn_colors=(
                "gnn_colors",
                "mean",
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
                "gap_from_target",
                "mean",
            ),
            exact_match_seeds=(
                "exact_match",
                "sum",
            ),
            worse_than_target_seeds=(
                "worse_than_target",
                "sum",
            ),
            all_valid=(
                "valid",
                "all",
            ),
        )
    )

    stability["exact_match_rate"] = (
        stability["exact_match_seeds"]
        / len(SEEDS)
    )

    return stability.sort_values(
        [
            "family",
            "graph_id",
        ]
    ).reset_index(drop=True)


def create_seed_summary(
    per_graph_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one overall test row for each seed.
    """
    summary = (
        per_graph_df
        .groupby(
            "seed",
            as_index=False,
        )
        .agg(
            num_test_graphs=(
                "graph_id",
                "nunique",
            ),
            total_gnn_colors=(
                "gnn_colors",
                "sum",
            ),
            total_target_colors=(
                "target_colors",
                "sum",
            ),
            total_gap_from_target=(
                "gap_from_target",
                "sum",
            ),
            exact_matches=(
                "exact_match",
                "sum",
            ),
            better_than_target_count=(
                "better_than_target",
                "sum",
            ),
            worse_than_target_count=(
                "worse_than_target",
                "sum",
            ),
            all_valid=(
                "valid",
                "all",
            ),
        )
    )

    summary["exact_match_rate"] = (
        summary["exact_matches"]
        / summary["num_test_graphs"]
    )

    return summary.sort_values(
        "seed"
    ).reset_index(drop=True)


def validate_results(
    per_graph_df: pd.DataFrame,
) -> None:
    """
    Validate completeness of the full evaluation.
    """
    expected_rows = (
        len(SEEDS)
        * EXPECTED_TEST_GRAPHS
    )

    if len(per_graph_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} seed-graph rows, "
            f"found {len(per_graph_df)}."
        )

    duplicate_rows = per_graph_df.duplicated(
        subset=[
            "seed",
            "graph_id",
        ]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Duplicate seed-graph evaluations found."
        )

    evaluations_per_seed = (
        per_graph_df
        .groupby("seed")["graph_id"]
        .nunique()
    )

    for seed in SEEDS:
        actual_count = int(
            evaluations_per_seed.get(seed, 0)
        )

        if actual_count != EXPECTED_TEST_GRAPHS:
            raise ValueError(
                f"Seed {seed}: expected "
                f"{EXPECTED_TEST_GRAPHS} test graphs, "
                f"found {actual_count}."
            )

    if not bool(
        per_graph_df["valid"].all()
    ):
        invalid_rows = per_graph_df.loc[
            ~per_graph_df["valid"],
            [
                "seed",
                "graph_id",
            ],
        ]

        raise ValueError(
            "Invalid colorings were produced:\n"
            f"{invalid_rows.to_string(index=False)}"
        )


def main() -> None:
    print(
        "Week 18 heterogeneous family evaluation"
    )
    print(
        "---------------------------------------"
    )
    print(f"Seeds: {SEEDS}")
    print()

    test_graphs = load_test_graphs()

    result_rows: list[
        dict[str, object]
    ] = []

    for seed in SEEDS:
        print(
            f"Evaluating seed {seed}"
        )

        model, checkpoint = (
            load_model_for_seed(seed)
        )

        for data in test_graphs:
            result_rows.append(
                evaluate_graph(
                    model=model,
                    data=data,
                    seed=seed,
                    checkpoint=checkpoint,
                )
            )

    per_graph_df = pd.DataFrame(
        result_rows
    )

    validate_results(
        per_graph_df
    )

    per_graph_df = per_graph_df.sort_values(
        [
            "seed",
            "family",
            "graph_id",
        ]
    ).reset_index(drop=True)

    family_by_seed_df = (
        create_family_by_seed_table(
            per_graph_df
        )
    )

    family_aggregate_df = (
        create_family_aggregate_table(
            per_graph_df=per_graph_df,
            family_by_seed_df=(
                family_by_seed_df
            ),
        )
    )

    graph_stability_df = (
        create_graph_stability_table(
            per_graph_df
        )
    )

    seed_summary_df = (
        create_seed_summary(
            per_graph_df
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_graph_df.to_csv(
        PER_GRAPH_OUTPUT,
        index=False,
    )

    family_by_seed_df.to_csv(
        FAMILY_BY_SEED_OUTPUT,
        index=False,
    )

    family_aggregate_df.to_csv(
        FAMILY_AGGREGATE_OUTPUT,
        index=False,
    )

    graph_stability_df.to_csv(
        GRAPH_STABILITY_OUTPUT,
        index=False,
    )

    seed_summary_df.to_csv(
        SEED_SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print(
        "Family performance across all five seeds"
    )
    print(
        "----------------------------------------"
    )

    display_columns = [
        "family",
        "num_graphs",
        "target_colors_per_seed",
        "mean_gnn_colors_per_seed",
        "minimum_gnn_colors_per_seed",
        "maximum_gnn_colors_per_seed",
        "mean_total_gap_per_seed",
        "exact_match_evaluations",
        "total_seed_graph_evaluations",
        "exact_match_rate",
        "all_valid",
    ]

    print(
        family_aggregate_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print("Overall performance by seed")
    print("---------------------------")
    print(
        seed_summary_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print(
        "Week 18 family evaluation completed."
    )
    print(
        "------------------------------------"
    )
    print(
        f"Saved per-graph results to: "
        f"{PER_GRAPH_OUTPUT}"
    )
    print(
        f"Saved family-by-seed table to: "
        f"{FAMILY_BY_SEED_OUTPUT}"
    )
    print(
        f"Saved family aggregate table to: "
        f"{FAMILY_AGGREGATE_OUTPUT}"
    )
    print(
        f"Saved graph stability table to: "
        f"{GRAPH_STABILITY_OUTPUT}"
    )
    print(
        f"Saved seed summary to: "
        f"{SEED_SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()