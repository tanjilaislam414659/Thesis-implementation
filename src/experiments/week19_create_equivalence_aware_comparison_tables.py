from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEEK19_RUN_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week19_equivalence_aware_objective_run_summary.csv"
)

WEEK18_RUN_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_gnn_training_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
)

OVERALL_OUTPUT_CSV = (
    OUTPUT_DIR
    / "week19_equivalence_aware_final_comparison.csv"
)

PER_GRAPH_OUTPUT_CSV = (
    OUTPUT_DIR
    / "week19_equivalence_aware_per_graph_comparison.csv"
)

REPRESENTATIVE_OUTPUT_CSV = (
    OUTPUT_DIR
    / "week19_equivalence_aware_representative_comparison.csv"
)

FULL_CONDITION = "train_125_plus105"
OBJECTIVES = ["mse", "ranking"]
EXPECTED_SEEDS = {0, 1, 2, 3, 4}
EXPECTED_TEST_GRAPHS = 5
EXPECTED_TARGET_TOTAL = 60
EXPECTED_COLPACK_TOTAL = 75

GRAPH_ORDER = [
    "week17_gap_cycle_square_c41",
    "week17_gap_join_c41_join_2",
    "week17_gap_join_c41_join_3",
    "week17_gap_join_c41_join_4",
    "week17_gap_join_c41_join_5",
]

GRAPH_SHORT_NAMES = {
    "week17_gap_cycle_square_c41": "C41_squared",
    "week17_gap_join_c41_join_2": "join_2",
    "week17_gap_join_c41_join_3": "join_3",
    "week17_gap_join_c41_join_4": "join_4",
    "week17_gap_join_c41_join_5": "join_5",
}

PER_GRAPH_PATTERN = re.compile(
    r"^(?P<graph_id>[^:]+):"
    r"(?P<description>.+?)/"
    r"gnn(?P<gnn_colors>\d+)/"
    r"target(?P<target_colors>\d+)/"
    r"colpack5(?P<colpack5_colors>\d+)$"
)


def load_week19_runs() -> pd.DataFrame:
    if not WEEK19_RUN_CSV.exists():
        raise FileNotFoundError(
            f"Week 19 run summary not found: {WEEK19_RUN_CSV}"
        )

    runs = pd.read_csv(WEEK19_RUN_CSV)
    required_columns = {
        "condition",
        "objective",
        "seed",
        "best_epoch",
        "final_validation_total_colors",
        "final_validation_loss_best_model",
        "final_test_total_colors",
        "final_test_target_colors",
        "final_test_colpack5_colors",
        "final_test_gap_from_target",
        "final_test_colors_saved_vs_colpack5",
        "final_test_exact_target_graphs",
        "final_test_all_valid",
        "final_test_per_graph_colors",
    }
    missing = required_columns - set(runs.columns)

    if missing:
        raise ValueError(
            "Week 19 run summary is missing columns: "
            f"{sorted(missing)}"
        )

    runs = runs[
        runs["condition"] == FULL_CONDITION
    ].copy()

    if len(runs) != len(OBJECTIVES) * len(EXPECTED_SEEDS):
        raise ValueError(
            "Expected 10 completed Week 19 runs, found "
            f"{len(runs)}."
        )

    if runs[["objective", "seed"]].duplicated().any():
        raise ValueError(
            "Duplicate Week 19 objective/seed rows found."
        )

    if set(runs["objective"].astype(str)) != set(OBJECTIVES):
        raise ValueError(
            "Week 19 objective set is incomplete."
        )

    for objective in OBJECTIVES:
        objective_seeds = set(
            runs[
                runs["objective"] == objective
            ]["seed"].astype(int)
        )

        if objective_seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{objective}: expected seeds "
                f"{sorted(EXPECTED_SEEDS)}, found "
                f"{sorted(objective_seeds)}."
            )

    if not bool(runs["final_test_all_valid"].all()):
        raise ValueError(
            "At least one Week 19 test coloring is invalid."
        )

    if not bool(
        (
            runs["final_test_target_colors"]
            == EXPECTED_TARGET_TOTAL
        ).all()
    ):
        raise ValueError(
            "Week 19 target totals are not frozen at 60."
        )

    if not bool(
        (
            runs["final_test_colpack5_colors"]
            == EXPECTED_COLPACK_TOTAL
        ).all()
    ):
        raise ValueError(
            "Week 19 ColPack totals are not frozen at 75."
        )

    return runs


def load_week18_full_condition_runs() -> pd.DataFrame:
    if not WEEK18_RUN_CSV.exists():
        raise FileNotFoundError(
            f"Week 18 run summary not found: {WEEK18_RUN_CSV}"
        )

    runs = pd.read_csv(WEEK18_RUN_CSV)
    required_columns = {
        "condition",
        "seed",
        "best_epoch",
        "final_validation_total_colors",
        "final_validation_loss_best_model",
        "final_test_total_colors",
        "final_test_target_colors",
        "final_test_colpack5_colors",
        "final_test_gap_from_target",
        "final_test_colors_saved_vs_colpack5",
        "final_test_exact_target_graphs",
        "final_test_all_valid",
        "final_test_per_graph_colors",
    }
    missing = required_columns - set(runs.columns)

    if missing:
        raise ValueError(
            "Week 18 run summary is missing columns: "
            f"{sorted(missing)}"
        )

    runs = runs[
        runs["condition"] == FULL_CONDITION
    ].copy()

    if len(runs) != len(EXPECTED_SEEDS):
        raise ValueError(
            "Expected five Week 18 full-condition runs, "
            f"found {len(runs)}."
        )

    if set(runs["seed"].astype(int)) != EXPECTED_SEEDS:
        raise ValueError(
            "Week 18 full-condition seed set is incomplete."
        )

    if not bool(runs["final_test_all_valid"].all()):
        raise ValueError(
            "At least one Week 18 test coloring is invalid."
        )

    return runs


def parse_per_graph_results(
    value: str,
) -> list[dict[str, object]]:
    parts = [
        part.strip()
        for part in str(value).split(";")
        if part.strip()
    ]

    if len(parts) != EXPECTED_TEST_GRAPHS:
        raise ValueError(
            "Expected five per-graph result parts, found "
            f"{len(parts)} in: {value}"
        )

    parsed_rows: list[dict[str, object]] = []

    for part in parts:
        match = PER_GRAPH_PATTERN.fullmatch(part)

        if match is None:
            raise ValueError(
                f"Could not parse per-graph result: {part}"
            )

        description = match.group("description")
        gap_match = re.search(
            r"(?:^|/)gap(?P<gap>\d+)$",
            description,
        )

        if gap_match is None:
            raise ValueError(
                f"Could not parse gap level from: {part}"
            )

        parsed_rows.append(
            {
                "graph_id": match.group("graph_id"),
                "gap_level": int(
                    gap_match.group("gap")
                ),
                "gnn_colors": int(
                    match.group("gnn_colors")
                ),
                "target_colors": int(
                    match.group("target_colors")
                ),
                "colpack5_colors": int(
                    match.group("colpack5_colors")
                ),
            }
        )

    parsed_ids = {
        row["graph_id"]
        for row in parsed_rows
    }

    if parsed_ids != set(GRAPH_ORDER):
        raise ValueError(
            "Unexpected per-graph test IDs: "
            f"{sorted(parsed_ids)}"
        )

    return parsed_rows


def expand_week19_per_graph(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for run in runs.itertuples(index=False):
        parsed = parse_per_graph_results(
            run.final_test_per_graph_colors
        )

        if sum(
            int(item["gnn_colors"])
            for item in parsed
        ) != int(run.final_test_total_colors):
            raise ValueError(
                f"{run.objective}, seed {run.seed}: "
                "per-graph colors do not sum to test total."
            )

        for item in parsed:
            rows.append(
                {
                    "objective": str(run.objective),
                    "seed": int(run.seed),
                    **item,
                    "exact_target": bool(
                        item["gnn_colors"]
                        == item["target_colors"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def select_representative(
    runs: pd.DataFrame,
) -> pd.Series:
    return (
        runs
        .sort_values(
            [
                "final_validation_total_colors",
                "final_validation_loss_best_model",
                "seed",
            ]
        )
        .iloc[0]
    )


def build_overall_comparison(
    week19_runs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for objective in OBJECTIVES:
        objective_df = week19_runs[
            week19_runs["objective"] == objective
        ].copy()
        representative = select_representative(
            objective_df
        )
        total_exact_outcomes = int(
            objective_df[
                "final_test_exact_target_graphs"
            ].sum()
        )

        rows.append(
            {
                "objective": objective,
                "num_seeds": len(objective_df),
                "mean_test_colors": float(
                    objective_df[
                        "final_test_total_colors"
                    ].mean()
                ),
                "std_test_colors_population": float(
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
                "mean_gap_from_target": float(
                    objective_df[
                        "final_test_gap_from_target"
                    ].mean()
                ),
                "mean_colors_saved_vs_colpack5": float(
                    objective_df[
                        "final_test_colors_saved_vs_colpack5"
                    ].mean()
                ),
                "mean_exact_target_graphs_per_seed": float(
                    objective_df[
                        "final_test_exact_target_graphs"
                    ].mean()
                ),
                "total_exact_target_graph_outcomes": (
                    total_exact_outcomes
                ),
                "total_test_graph_outcomes": (
                    len(objective_df)
                    * EXPECTED_TEST_GRAPHS
                ),
                "representative_seed": int(
                    representative["seed"]
                ),
                "representative_best_epoch": int(
                    representative["best_epoch"]
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

    comparison = pd.DataFrame(rows)
    mse_row = comparison[
        comparison["objective"] == "mse"
    ].iloc[0]
    ranking_row = comparison[
        comparison["objective"] == "ranking"
    ].iloc[0]

    comparison["change_vs_mse_mean_test_colors"] = (
        comparison["mean_test_colors"]
        - float(mse_row["mean_test_colors"])
    )
    comparison["change_vs_mse_std_test_colors"] = (
        comparison["std_test_colors_population"]
        - float(mse_row["std_test_colors_population"])
    )
    comparison[
        "change_vs_mse_exact_target_outcomes"
    ] = (
        comparison["total_exact_target_graph_outcomes"]
        - int(mse_row["total_exact_target_graph_outcomes"])
    )

    mse_std = float(
        mse_row["std_test_colors_population"]
    )
    ranking_std = float(
        ranking_row["std_test_colors_population"]
    )
    comparison["seed_std_reduction_percent_vs_mse"] = 0.0
    comparison.loc[
        comparison["objective"] == "ranking",
        "seed_std_reduction_percent_vs_mse",
    ] = 100.0 * (1.0 - ranking_std / mse_std)

    return comparison


def build_per_graph_comparison(
    expanded: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for graph_id in GRAPH_ORDER:
        graph_df = expanded[
            expanded["graph_id"] == graph_id
        ]

        if len(graph_df) != 10:
            raise ValueError(
                f"{graph_id}: expected 10 objective/seed rows, "
                f"found {len(graph_df)}."
            )

        fixed_targets = set(
            graph_df["target_colors"].astype(int)
        )
        fixed_colpack = set(
            graph_df["colpack5_colors"].astype(int)
        )
        fixed_gaps = set(
            graph_df["gap_level"].astype(int)
        )

        if not (
            len(fixed_targets)
            == len(fixed_colpack)
            == len(fixed_gaps)
            == 1
        ):
            raise ValueError(
                f"{graph_id}: frozen metadata differ across runs."
            )

        objective_values = {}

        for objective in OBJECTIVES:
            objective_df = graph_df[
                graph_df["objective"] == objective
            ]

            if len(objective_df) != 5:
                raise ValueError(
                    f"{graph_id}, {objective}: expected five seeds."
                )

            objective_values[objective] = {
                "mean": float(
                    objective_df["gnn_colors"].mean()
                ),
                "std": float(
                    objective_df["gnn_colors"].std(ddof=0)
                ),
                "min": int(
                    objective_df["gnn_colors"].min()
                ),
                "max": int(
                    objective_df["gnn_colors"].max()
                ),
                "exact": int(
                    objective_df["exact_target"].sum()
                ),
            }

        rows.append(
            {
                "graph_id": graph_id,
                "graph_short_name": GRAPH_SHORT_NAMES[graph_id],
                "gap_level": next(iter(fixed_gaps)),
                "target_colors": next(iter(fixed_targets)),
                "colpack5_colors": next(iter(fixed_colpack)),
                "mse_mean_colors": objective_values["mse"]["mean"],
                "mse_std_colors_population": (
                    objective_values["mse"]["std"]
                ),
                "mse_min_colors": objective_values["mse"]["min"],
                "mse_max_colors": objective_values["mse"]["max"],
                "mse_exact_target_runs": objective_values["mse"]["exact"],
                "ranking_mean_colors": (
                    objective_values["ranking"]["mean"]
                ),
                "ranking_std_colors_population": (
                    objective_values["ranking"]["std"]
                ),
                "ranking_min_colors": (
                    objective_values["ranking"]["min"]
                ),
                "ranking_max_colors": (
                    objective_values["ranking"]["max"]
                ),
                "ranking_exact_target_runs": (
                    objective_values["ranking"]["exact"]
                ),
                "ranking_minus_mse_mean_colors": (
                    objective_values["ranking"]["mean"]
                    - objective_values["mse"]["mean"]
                ),
                "ranking_minus_mse_exact_target_runs": (
                    objective_values["ranking"]["exact"]
                    - objective_values["mse"]["exact"]
                ),
            }
        )

    return pd.DataFrame(rows)


def representative_to_row(
    experiment_label: str,
    objective_label: str,
    representative: pd.Series,
) -> dict[str, object]:
    parsed = parse_per_graph_results(
        representative["final_test_per_graph_colors"]
    )
    by_graph = {
        str(item["graph_id"]): int(item["gnn_colors"])
        for item in parsed
    }

    row: dict[str, object] = {
        "experiment": experiment_label,
        "objective": objective_label,
        "seed": int(representative["seed"]),
        "best_epoch": int(representative["best_epoch"]),
        "validation_total_colors": int(
            representative["final_validation_total_colors"]
        ),
        "test_total_colors": int(
            representative["final_test_total_colors"]
        ),
        "test_gap_from_target": int(
            representative["final_test_gap_from_target"]
        ),
        "colors_saved_vs_colpack5": int(
            representative[
                "final_test_colors_saved_vs_colpack5"
            ]
        ),
        "exact_target_graphs": int(
            representative["final_test_exact_target_graphs"]
        ),
        "all_test_colorings_valid": bool(
            representative["final_test_all_valid"]
        ),
    }

    for graph_id in GRAPH_ORDER:
        short_name = GRAPH_SHORT_NAMES[graph_id]
        row[f"{short_name}_colors"] = by_graph[graph_id]

    return row


def build_representative_comparison(
    week18_runs: pd.DataFrame,
    week19_runs: pd.DataFrame,
) -> pd.DataFrame:
    week18_representative = select_representative(
        week18_runs
    )
    week19_mse_representative = select_representative(
        week19_runs[
            week19_runs["objective"] == "mse"
        ]
    )
    week19_ranking_representative = select_representative(
        week19_runs[
            week19_runs["objective"] == "ranking"
        ]
    )

    return pd.DataFrame(
        [
            representative_to_row(
                experiment_label=(
                    "week18_historical_full_data"
                ),
                objective_label=(
                    "mse_fixed_graph_order"
                ),
                representative=week18_representative,
            ),
            representative_to_row(
                experiment_label=(
                    "week19_fair_comparison"
                ),
                objective_label=(
                    "mse_shuffled_graph_order"
                ),
                representative=week19_mse_representative,
            ),
            representative_to_row(
                experiment_label=(
                    "week19_fair_comparison"
                ),
                objective_label=(
                    "equivalence_aware_pairwise_ranking"
                ),
                representative=week19_ranking_representative,
            ),
        ]
    )


def main() -> None:
    week19_runs = load_week19_runs()
    week18_runs = load_week18_full_condition_runs()
    expanded = expand_week19_per_graph(week19_runs)

    overall_df = build_overall_comparison(
        week19_runs
    )
    per_graph_df = build_per_graph_comparison(
        expanded
    )
    representative_df = build_representative_comparison(
        week18_runs=week18_runs,
        week19_runs=week19_runs,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(
        OVERALL_OUTPUT_CSV,
        index=False,
    )
    per_graph_df.to_csv(
        PER_GRAPH_OUTPUT_CSV,
        index=False,
    )
    representative_df.to_csv(
        REPRESENTATIVE_OUTPUT_CSV,
        index=False,
    )

    print(
        "Week 19 equivalence-aware comparison tables "
        "created successfully."
    )
    print("------------------------------------------")
    print()
    print("Overall five-seed comparison:")
    print(
        overall_df[
            [
                "objective",
                "mean_test_colors",
                "std_test_colors_population",
                "minimum_test_colors",
                "maximum_test_colors",
                "total_exact_target_graph_outcomes",
                "representative_seed",
                "representative_test_colors",
            ]
        ].round(3).to_string(index=False)
    )
    print()
    print("Per-test-graph comparison:")
    print(
        per_graph_df[
            [
                "graph_short_name",
                "target_colors",
                "mse_mean_colors",
                "ranking_mean_colors",
                "ranking_minus_mse_mean_colors",
                "mse_exact_target_runs",
                "ranking_exact_target_runs",
            ]
        ].round(3).to_string(index=False)
    )
    print()
    print("Representative comparison:")
    print(
        representative_df[
            [
                "experiment",
                "objective",
                "seed",
                "validation_total_colors",
                "test_total_colors",
                "exact_target_graphs",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Saved overall table to: {OVERALL_OUTPUT_CSV}")
    print(f"Saved per-graph table to: {PER_GRAPH_OUTPUT_CSV}")
    print(
        "Saved representative table to: "
        f"{REPRESENTATIVE_OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()