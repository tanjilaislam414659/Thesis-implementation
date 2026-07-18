from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POOL_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_combined_safeguarded_heterogeneous_pool.csv"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week18_heterogeneous_balanced_split.csv"
)

GNN_PER_GRAPH_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_heterogeneous_test_per_graph_all_seeds.csv"
)

TRAINING_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_heterogeneous_gnn_training_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
)

PER_GRAPH_OUTPUT = (
    OUTPUT_DIR
    / "week18_final_test_baseline_comparison_per_graph.csv"
)

FAMILY_OUTPUT = (
    OUTPUT_DIR
    / "week18_final_test_baseline_comparison_by_family.csv"
)

OVERALL_OUTPUT = (
    OUTPUT_DIR
    / "week18_final_test_baseline_comparison_overall.csv"
)


HEURISTICS = [
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]

SEEDS = [0, 1, 2, 3, 4]

EXPECTED_TEST_GRAPHS = 12
EXPECTED_GNN_ROWS = 60
EXPECTED_REPRESENTATIVE_SEED = 1


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    """
    Ensure that a table contains all required columns.
    """
    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def load_test_split() -> pd.DataFrame:
    """
    Load the final Week 18 test split.
    """
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {SPLIT_CSV}"
        )

    split_df = pd.read_csv(SPLIT_CSV)

    require_columns(
        dataframe=split_df,
        required_columns={
            "graph_id",
            "split",
            "family",
            "num_vertices",
            "num_edges",
            "selected_teacher_ordering",
            "ordering_gap",
        },
        table_name="Week 18 split table",
    )

    test_df = (
        split_df[
            split_df["split"] == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(test_df) != EXPECTED_TEST_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_GRAPHS} test graphs, "
            f"found {len(test_df)}."
        )

    if test_df["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in the test split."
        )

    return test_df


def load_colpack_pool() -> pd.DataFrame:
    """
    Load the five ColPack heuristic results.
    """
    if not POOL_CSV.exists():
        raise FileNotFoundError(
            f"Combined ColPack pool not found: {POOL_CSV}"
        )

    pool_df = pd.read_csv(POOL_CSV)

    require_columns(
        dataframe=pool_df,
        required_columns={
            "graph_id",
            "family",
            "num_vertices",
            "num_edges",
            *HEURISTICS,
            "best_colpack5_colors",
            "worst_colpack5_colors",
            "ordering_gap",
            "best_colpack5_orderings",
            "num_best_orderings",
        },
        table_name="Week 18 combined ColPack pool",
    )

    if pool_df["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in the ColPack pool."
        )

    return pool_df


def load_gnn_results() -> pd.DataFrame:
    """
    Load all five-seed test evaluations.
    """
    if not GNN_PER_GRAPH_CSV.exists():
        raise FileNotFoundError(
            f"GNN per-graph results not found: "
            f"{GNN_PER_GRAPH_CSV}"
        )

    gnn_df = pd.read_csv(
        GNN_PER_GRAPH_CSV
    )

    require_columns(
        dataframe=gnn_df,
        required_columns={
            "seed",
            "graph_id",
            "family",
            "gnn_colors",
            "target_colors",
            "best_colpack5_colors",
            "gap_from_target",
            "exact_match",
            "better_than_target",
            "worse_than_target",
            "valid",
        },
        table_name="Week 18 GNN per-graph results",
    )

    if len(gnn_df) != EXPECTED_GNN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_GNN_ROWS} GNN evaluation rows, "
            f"found {len(gnn_df)}."
        )

    duplicate_rows = gnn_df.duplicated(
        subset=[
            "seed",
            "graph_id",
        ]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Duplicate seed-graph rows found in GNN results."
        )

    actual_seeds = sorted(
        gnn_df["seed"]
        .astype(int)
        .unique()
        .tolist()
    )

    if actual_seeds != SEEDS:
        raise ValueError(
            f"Expected seeds {SEEDS}, found {actual_seeds}."
        )

    if not bool(
        gnn_df["valid"].all()
    ):
        raise ValueError(
            "At least one GNN coloring is invalid."
        )

    return gnn_df


def select_representative_seed() -> int:
    """
    Select the representative model using validation only:

    1. Fewest validation colors.
    2. Lowest validation loss as the tie-breaker.
    3. Lowest seed only as a deterministic final tie-breaker.
    """
    if not TRAINING_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Training summary not found: "
            f"{TRAINING_SUMMARY_CSV}"
        )

    training_df = pd.read_csv(
        TRAINING_SUMMARY_CSV
    )

    require_columns(
        dataframe=training_df,
        required_columns={
            "seed",
            "best_validation_total_colors",
            "best_validation_loss",
        },
        table_name="Week 18 training summary",
    )

    ranked = training_df.sort_values(
        by=[
            "best_validation_total_colors",
            "best_validation_loss",
            "seed",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    representative_seed = int(
        ranked.iloc[0]["seed"]
    )

    if (
        representative_seed
        != EXPECTED_REPRESENTATIVE_SEED
    ):
        raise ValueError(
            "Validation-based representative seed changed. "
            f"Expected seed "
            f"{EXPECTED_REPRESENTATIVE_SEED}, "
            f"found seed {representative_seed}."
        )

    return representative_seed


def build_baseline_test_table(
    test_split: pd.DataFrame,
    pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join the final test split with the five ColPack results.
    """
    baseline_columns = [
        "graph_id",
        *HEURISTICS,
        "best_colpack5_colors",
        "worst_colpack5_colors",
        "ordering_gap",
        "best_colpack5_orderings",
        "num_best_orderings",
    ]

    baseline_df = test_split[
        [
            "graph_id",
            "family",
            "num_vertices",
            "num_edges",
            "selected_teacher_ordering",
        ]
    ].merge(
        pool_df[baseline_columns],
        on="graph_id",
        how="left",
        validate="one_to_one",
    )

    if baseline_df[
        "best_colpack5_colors"
    ].isna().any():
        missing_graphs = baseline_df.loc[
            baseline_df[
                "best_colpack5_colors"
            ].isna(),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Missing ColPack results for: "
            f"{missing_graphs}"
        )

    computed_best = baseline_df[
        HEURISTICS
    ].min(axis=1)

    computed_worst = baseline_df[
        HEURISTICS
    ].max(axis=1)

    if not (
        computed_best.astype(int)
        == baseline_df[
            "best_colpack5_colors"
        ].astype(int)
    ).all():
        raise ValueError(
            "Stored best-of-five values do not match "
            "the five heuristic columns."
        )

    if not (
        computed_worst.astype(int)
        == baseline_df[
            "worst_colpack5_colors"
        ].astype(int)
    ).all():
        raise ValueError(
            "Stored worst-of-five values do not match "
            "the five heuristic columns."
        )

    return baseline_df


def build_per_graph_table(
    baseline_df: pd.DataFrame,
    gnn_df: pd.DataFrame,
    representative_seed: int,
) -> pd.DataFrame:
    """
    Build the central per-graph comparison table.
    """
    representative_df = (
        gnn_df[
            gnn_df["seed"]
            == representative_seed
        ][
            [
                "graph_id",
                "gnn_colors",
                "valid",
            ]
        ]
        .rename(
            columns={
                "gnn_colors": (
                    "gnn_representative_colors"
                ),
                "valid": (
                    "gnn_representative_valid"
                ),
            }
        )
    )

    gnn_statistics = (
        gnn_df
        .groupby(
            "graph_id",
            as_index=False,
        )
        .agg(
            gnn_mean_colors=(
                "gnn_colors",
                "mean",
            ),
            gnn_std_colors=(
                "gnn_colors",
                "std",
            ),
            gnn_min_colors=(
                "gnn_colors",
                "min",
            ),
            gnn_max_colors=(
                "gnn_colors",
                "max",
            ),
            exact_match_seed_count=(
                "exact_match",
                "sum",
            ),
            better_than_best_seed_count=(
                "better_than_target",
                "sum",
            ),
            worse_than_best_seed_count=(
                "worse_than_target",
                "sum",
            ),
            all_gnn_colorings_valid=(
                "valid",
                "all",
            ),
        )
    )

    result = (
        baseline_df
        .merge(
            representative_df,
            on="graph_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            gnn_statistics,
            on="graph_id",
            how="left",
            validate="one_to_one",
        )
    )

    if result[
        "gnn_representative_colors"
    ].isna().any():
        raise ValueError(
            "Representative GNN results are missing "
            "for at least one test graph."
        )

    result[
        "gnn_representative_gap_vs_best5"
    ] = (
        result["gnn_representative_colors"]
        - result["best_colpack5_colors"]
    )

    result[
        "gnn_mean_gap_vs_best5"
    ] = (
        result["gnn_mean_colors"]
        - result["best_colpack5_colors"]
    )

    result[
        "gnn_min_gap_vs_best5"
    ] = (
        result["gnn_min_colors"]
        - result["best_colpack5_colors"]
    )

    result[
        "gnn_max_gap_vs_best5"
    ] = (
        result["gnn_max_colors"]
        - result["best_colpack5_colors"]
    )

    result[
        "representative_exact_match"
    ] = (
        result[
            "gnn_representative_gap_vs_best5"
        ]
        == 0
    )

    result[
        "all_seeds_exact_match"
    ] = (
        result["exact_match_seed_count"]
        == len(SEEDS)
    )

    result[
        "any_seed_beats_best5"
    ] = (
        result[
            "better_than_best_seed_count"
        ]
        > 0
    )

    output_columns = [
        "graph_id",
        "family",
        "num_vertices",
        "num_edges",
        "selected_teacher_ordering",
        "NATURAL",
        "LARGEST_FIRST",
        "DYNAMIC_LARGEST_FIRST",
        "INCIDENCE_DEGREE",
        "SMALLEST_LAST",
        "best_colpack5_colors",
        "worst_colpack5_colors",
        "ordering_gap",
        "best_colpack5_orderings",
        "num_best_orderings",
        "gnn_representative_colors",
        "gnn_mean_colors",
        "gnn_std_colors",
        "gnn_min_colors",
        "gnn_max_colors",
        "gnn_representative_gap_vs_best5",
        "gnn_mean_gap_vs_best5",
        "gnn_min_gap_vs_best5",
        "gnn_max_gap_vs_best5",
        "exact_match_seed_count",
        "better_than_best_seed_count",
        "worse_than_best_seed_count",
        "representative_exact_match",
        "all_seeds_exact_match",
        "any_seed_beats_best5",
        "gnn_representative_valid",
        "all_gnn_colorings_valid",
    ]

    return (
        result[output_columns]
        .sort_values(
            [
                "family",
                "graph_id",
            ]
        )
        .reset_index(drop=True)
    )


def identify_best_fixed_heuristics(
    row: pd.Series,
) -> tuple[str, int]:
    """
    Identify the fixed heuristic or tied heuristics with the
    smallest total color count for one table row.
    """
    heuristic_totals = {
        heuristic: int(
            row[heuristic]
        )
        for heuristic in HEURISTICS
    }

    best_total = min(
        heuristic_totals.values()
    )

    best_names = [
        heuristic
        for heuristic, total
        in heuristic_totals.items()
        if total == best_total
    ]

    return (
        "; ".join(best_names),
        best_total,
    )


def build_family_table(
    per_graph_df: pd.DataFrame,
    gnn_df: pd.DataFrame,
    representative_seed: int,
) -> pd.DataFrame:
    """
    Aggregate the final comparison by graph family.
    """
    baseline_aggregation = {
        "graph_id": "count",
        **{
            heuristic: "sum"
            for heuristic in HEURISTICS
        },
        "best_colpack5_colors": "sum",
        "worst_colpack5_colors": "sum",
    }

    family_baselines = (
        per_graph_df
        .groupby(
            "family",
            as_index=False,
        )
        .agg(
            baseline_aggregation
        )
        .rename(
            columns={
                "graph_id": "num_graphs",
                "best_colpack5_colors": (
                    "best_colpack5_oracle_total"
                ),
                "worst_colpack5_colors": (
                    "worst_colpack5_total"
                ),
            }
        )
    )

    best_fixed_names: list[str] = []
    best_fixed_totals: list[int] = []

    for _, row in family_baselines.iterrows():
        names, total = (
            identify_best_fixed_heuristics(
                row
            )
        )

        best_fixed_names.append(
            names
        )

        best_fixed_totals.append(
            total
        )

    family_baselines[
        "best_fixed_heuristics"
    ] = best_fixed_names

    family_baselines[
        "best_fixed_heuristic_total"
    ] = best_fixed_totals

    family_seed_totals = (
        gnn_df
        .groupby(
            [
                "seed",
                "family",
            ],
            as_index=False,
        )
        .agg(
            total_gnn_colors=(
                "gnn_colors",
                "sum",
            ),
            total_target_colors=(
                "target_colors",
                "sum",
            ),
            exact_matches=(
                "exact_match",
                "sum",
            ),
            better_than_best_count=(
                "better_than_target",
                "sum",
            ),
            worse_than_best_count=(
                "worse_than_target",
                "sum",
            ),
            all_valid=(
                "valid",
                "all",
            ),
        )
    )

    representative_family = (
        family_seed_totals[
            family_seed_totals["seed"]
            == representative_seed
        ][
            [
                "family",
                "total_gnn_colors",
                "exact_matches",
                "better_than_best_count",
                "worse_than_best_count",
                "all_valid",
            ]
        ]
        .rename(
            columns={
                "total_gnn_colors": (
                    "gnn_representative_total"
                ),
                "exact_matches": (
                    "gnn_representative_exact_matches"
                ),
                "better_than_best_count": (
                    "gnn_representative_better_count"
                ),
                "worse_than_best_count": (
                    "gnn_representative_worse_count"
                ),
                "all_valid": (
                    "gnn_representative_all_valid"
                ),
            }
        )
    )

    family_seed_statistics = (
        family_seed_totals
        .groupby(
            "family",
            as_index=False,
        )
        .agg(
            gnn_mean_total_across_seeds=(
                "total_gnn_colors",
                "mean",
            ),
            gnn_std_total_across_seeds=(
                "total_gnn_colors",
                "std",
            ),
            gnn_min_total_across_seeds=(
                "total_gnn_colors",
                "min",
            ),
            gnn_max_total_across_seeds=(
                "total_gnn_colors",
                "max",
            ),
            mean_exact_matches_per_seed=(
                "exact_matches",
                "mean",
            ),
            all_seed_family_colorings_valid=(
                "all_valid",
                "all",
            ),
        )
    )

    result = (
        family_baselines
        .merge(
            representative_family,
            on="family",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            family_seed_statistics,
            on="family",
            how="inner",
            validate="one_to_one",
        )
    )

    result[
        "gnn_representative_gap_vs_oracle"
    ] = (
        result["gnn_representative_total"]
        - result[
            "best_colpack5_oracle_total"
        ]
    )

    result[
        "gnn_representative_gap_vs_best_fixed"
    ] = (
        result["gnn_representative_total"]
        - result[
            "best_fixed_heuristic_total"
        ]
    )

    result[
        "gnn_mean_gap_vs_oracle"
    ] = (
        result[
            "gnn_mean_total_across_seeds"
        ]
        - result[
            "best_colpack5_oracle_total"
        ]
    )

    result[
        "gnn_representative_exact_match_rate"
    ] = (
        result[
            "gnn_representative_exact_matches"
        ]
        / result["num_graphs"]
    )

    output_columns = [
        "family",
        "num_graphs",
        "NATURAL",
        "LARGEST_FIRST",
        "DYNAMIC_LARGEST_FIRST",
        "INCIDENCE_DEGREE",
        "SMALLEST_LAST",
        "best_fixed_heuristics",
        "best_fixed_heuristic_total",
        "best_colpack5_oracle_total",
        "worst_colpack5_total",
        "gnn_representative_total",
        "gnn_mean_total_across_seeds",
        "gnn_std_total_across_seeds",
        "gnn_min_total_across_seeds",
        "gnn_max_total_across_seeds",
        "gnn_representative_gap_vs_oracle",
        "gnn_representative_gap_vs_best_fixed",
        "gnn_mean_gap_vs_oracle",
        "gnn_representative_exact_matches",
        "gnn_representative_exact_match_rate",
        "mean_exact_matches_per_seed",
        "gnn_representative_better_count",
        "gnn_representative_worse_count",
        "gnn_representative_all_valid",
        "all_seed_family_colorings_valid",
    ]

    return (
        result[output_columns]
        .sort_values("family")
        .reset_index(drop=True)
    )


def make_overall_row(
    method: str,
    category: str,
    total_colors: float,
    oracle_total: int,
    best_fixed_total: int,
    exact_matches: float | None,
    all_valid: bool | None,
    notes: str,
) -> dict[str, object]:
    """
    Build one row of the overall comparison table.
    """
    gap_vs_oracle = (
        float(total_colors)
        - float(oracle_total)
    )

    gap_vs_best_fixed = (
        float(total_colors)
        - float(best_fixed_total)
    )

    relative_gap_percent = (
        100.0
        * gap_vs_oracle
        / float(oracle_total)
    )

    return {
        "method": method,
        "category": category,
        "total_colors": total_colors,
        "average_colors_per_graph": (
            float(total_colors)
            / EXPECTED_TEST_GRAPHS
        ),
        "gap_vs_best_of_5_oracle": (
            gap_vs_oracle
        ),
        "average_gap_per_graph": (
            gap_vs_oracle
            / EXPECTED_TEST_GRAPHS
        ),
        "relative_gap_to_oracle_percent": (
            relative_gap_percent
        ),
        "gap_vs_best_fixed_heuristic": (
            gap_vs_best_fixed
        ),
        "exact_matches_with_oracle": (
            exact_matches
        ),
        "all_gnn_colorings_valid": (
            all_valid
        ),
        "notes": notes,
    }


def build_overall_table(
    per_graph_df: pd.DataFrame,
    gnn_df: pd.DataFrame,
    representative_seed: int,
) -> pd.DataFrame:
    """
    Build the central overall test-set comparison.
    """
    oracle_total = int(
        per_graph_df[
            "best_colpack5_colors"
        ].sum()
    )

    heuristic_totals = {
        heuristic: int(
            per_graph_df[heuristic].sum()
        )
        for heuristic in HEURISTICS
    }

    best_fixed_total = min(
        heuristic_totals.values()
    )

    best_fixed_heuristics = [
        heuristic
        for heuristic, total
        in heuristic_totals.items()
        if total == best_fixed_total
    ]

    seed_totals = (
        gnn_df
        .groupby(
            "seed",
            as_index=False,
        )
        .agg(
            total_gnn_colors=(
                "gnn_colors",
                "sum",
            ),
            exact_matches=(
                "exact_match",
                "sum",
            ),
            all_valid=(
                "valid",
                "all",
            ),
        )
        .sort_values("seed")
        .reset_index(drop=True)
    )

    representative_row = seed_totals[
        seed_totals["seed"]
        == representative_seed
    ]

    if len(representative_row) != 1:
        raise ValueError(
            "Representative seed result is missing."
        )

    representative_row = (
        representative_row.iloc[0]
    )

    minimum_seed_total = int(
        seed_totals[
            "total_gnn_colors"
        ].min()
    )

    maximum_seed_total = int(
        seed_totals[
            "total_gnn_colors"
        ].max()
    )

    best_seed_rows = seed_totals[
        seed_totals["total_gnn_colors"]
        == minimum_seed_total
    ]

    worst_seed_rows = seed_totals[
        seed_totals["total_gnn_colors"]
        == maximum_seed_total
    ]

    best_seed_ids = (
        best_seed_rows["seed"]
        .astype(int)
        .tolist()
    )

    worst_seed_ids = (
        worst_seed_rows["seed"]
        .astype(int)
        .tolist()
    )

    mean_seed_total = float(
        seed_totals[
            "total_gnn_colors"
        ].mean()
    )

    mean_exact_matches = float(
        seed_totals[
            "exact_matches"
        ].mean()
    )

    rows: list[
        dict[str, object]
    ] = []

    for heuristic in HEURISTICS:
        total = heuristic_totals[
            heuristic
        ]

        exact_matches = int(
            (
                per_graph_df[heuristic]
                == per_graph_df[
                    "best_colpack5_colors"
                ]
            ).sum()
        )

        notes = (
            "Fixed ColPack heuristic."
        )

        if total == best_fixed_total:
            notes = (
                "Best fixed ColPack heuristic "
                "on the complete test set."
            )

        rows.append(
            make_overall_row(
                method=heuristic,
                category=(
                    "fixed_colpack_heuristic"
                ),
                total_colors=total,
                oracle_total=oracle_total,
                best_fixed_total=(
                    best_fixed_total
                ),
                exact_matches=exact_matches,
                all_valid=None,
                notes=notes,
            )
        )

    rows.append(
        make_overall_row(
            method="BEST_OF_5_ORACLE",
            category=(
                "per_graph_colpack_oracle"
            ),
            total_colors=oracle_total,
            oracle_total=oracle_total,
            best_fixed_total=best_fixed_total,
            exact_matches=(
                EXPECTED_TEST_GRAPHS
            ),
            all_valid=None,
            notes=(
                "Chooses the best result among the five "
                "ColPack heuristics separately for each graph."
            ),
        )
    )

    rows.append(
        make_overall_row(
            method=(
                f"GNN_REPRESENTATIVE_SEED_"
                f"{representative_seed}"
            ),
            category=(
                "single_validation_selected_gnn"
            ),
            total_colors=int(
                representative_row[
                    "total_gnn_colors"
                ]
            ),
            oracle_total=oracle_total,
            best_fixed_total=best_fixed_total,
            exact_matches=int(
                representative_row[
                    "exact_matches"
                ]
            ),
            all_valid=bool(
                representative_row[
                    "all_valid"
                ]
            ),
            notes=(
                "Representative checkpoint selected using "
                "validation colors and validation loss only."
            ),
        )
    )

    rows.append(
        make_overall_row(
            method="GNN_MEAN_ACROSS_5_SEEDS",
            category=(
                "cross_seed_summary"
            ),
            total_colors=mean_seed_total,
            oracle_total=oracle_total,
            best_fixed_total=best_fixed_total,
            exact_matches=mean_exact_matches,
            all_valid=bool(
                seed_totals[
                    "all_valid"
                ].all()
            ),
            notes=(
                "Mean test total across five independently "
                "trained GNN seeds."
            ),
        )
    )

    rows.append(
        make_overall_row(
            method="GNN_MIN_SINGLE_SEED_TOTAL",
            category=(
                "cross_seed_summary"
            ),
            total_colors=minimum_seed_total,
            oracle_total=oracle_total,
            best_fixed_total=best_fixed_total,
            exact_matches=float(
                best_seed_rows[
                    "exact_matches"
                ].mean()
            ),
            all_valid=bool(
                best_seed_rows[
                    "all_valid"
                ].all()
            ),
            notes=(
                "Minimum single-seed test total. "
                f"Achieved by seeds {best_seed_ids}."
            ),
        )
    )

    rows.append(
        make_overall_row(
            method="GNN_MAX_SINGLE_SEED_TOTAL",
            category=(
                "cross_seed_summary"
            ),
            total_colors=maximum_seed_total,
            oracle_total=oracle_total,
            best_fixed_total=best_fixed_total,
            exact_matches=float(
                worst_seed_rows[
                    "exact_matches"
                ].mean()
            ),
            all_valid=bool(
                worst_seed_rows[
                    "all_valid"
                ].all()
            ),
            notes=(
                "Maximum single-seed test total. "
                f"Achieved by seeds {worst_seed_ids}."
            ),
        )
    )

    overall_df = pd.DataFrame(
        rows
    )

    overall_df[
        "best_fixed_heuristics_on_test"
    ] = "; ".join(
        best_fixed_heuristics
    )

    overall_df[
        "best_fixed_total_on_test"
    ] = best_fixed_total

    overall_df[
        "best_of_5_oracle_total"
    ] = oracle_total

    return overall_df


def validate_final_tables(
    per_graph_df: pd.DataFrame,
    family_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> None:
    """
    Perform final consistency checks.
    """
    if len(per_graph_df) != EXPECTED_TEST_GRAPHS:
        raise ValueError(
            "Unexpected number of per-graph rows."
        )

    if not bool(
        per_graph_df[
            "all_gnn_colorings_valid"
        ].all()
    ):
        raise ValueError(
            "The final table contains invalid "
            "GNN colorings."
        )

    per_graph_oracle_total = int(
        per_graph_df[
            "best_colpack5_colors"
        ].sum()
    )

    family_oracle_total = int(
        family_df[
            "best_colpack5_oracle_total"
        ].sum()
    )

    if (
        per_graph_oracle_total
        != family_oracle_total
    ):
        raise ValueError(
            "Oracle totals differ between the "
            "per-graph and family tables."
        )

    oracle_rows = overall_df[
        overall_df["method"]
        == "BEST_OF_5_ORACLE"
    ]

    if len(oracle_rows) != 1:
        raise ValueError(
            "Overall table must contain exactly "
            "one best-of-five oracle row."
        )

    overall_oracle_total = int(
        oracle_rows.iloc[0][
            "total_colors"
        ]
    )

    if (
        overall_oracle_total
        != per_graph_oracle_total
    ):
        raise ValueError(
            "Overall oracle total does not match "
            "the per-graph table."
        )


def main() -> None:
    print(
        "Week 18 final baseline comparison"
    )
    print(
        "---------------------------------"
    )

    test_split = load_test_split()
    pool_df = load_colpack_pool()
    gnn_df = load_gnn_results()

    representative_seed = (
        select_representative_seed()
    )

    test_graph_ids = set(
        test_split["graph_id"].astype(str)
    )

    pool_graph_ids = set(
        pool_df["graph_id"].astype(str)
    )

    gnn_graph_ids = set(
        gnn_df["graph_id"].astype(str)
    )

    missing_from_pool = (
        test_graph_ids
        - pool_graph_ids
    )

    if missing_from_pool:
        raise ValueError(
            f"Test graphs missing from ColPack pool: "
            f"{sorted(missing_from_pool)}"
        )

    if gnn_graph_ids != test_graph_ids:
        raise ValueError(
            "GNN graph IDs do not exactly match "
            "the final test split.\n"
            f"Only in split: "
            f"{sorted(test_graph_ids - gnn_graph_ids)}\n"
            f"Only in GNN results: "
            f"{sorted(gnn_graph_ids - test_graph_ids)}"
        )

    baseline_df = (
        build_baseline_test_table(
            test_split=test_split,
            pool_df=pool_df,
        )
    )

    per_graph_df = (
        build_per_graph_table(
            baseline_df=baseline_df,
            gnn_df=gnn_df,
            representative_seed=(
                representative_seed
            ),
        )
    )

    family_df = build_family_table(
        per_graph_df=per_graph_df,
        gnn_df=gnn_df,
        representative_seed=(
            representative_seed
        ),
    )

    overall_df = build_overall_table(
        per_graph_df=per_graph_df,
        gnn_df=gnn_df,
        representative_seed=(
            representative_seed
        ),
    )

    validate_final_tables(
        per_graph_df=per_graph_df,
        family_df=family_df,
        overall_df=overall_df,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_graph_df.to_csv(
        PER_GRAPH_OUTPUT,
        index=False,
    )

    family_df.to_csv(
        FAMILY_OUTPUT,
        index=False,
    )

    overall_df.to_csv(
        OVERALL_OUTPUT,
        index=False,
    )

    print(
        f"Representative seed: "
        f"{representative_seed}"
    )
    print()

    print("Overall test-set comparison")
    print("---------------------------")

    overall_display_columns = [
        "method",
        "total_colors",
        "average_colors_per_graph",
        "gap_vs_best_of_5_oracle",
        "gap_vs_best_fixed_heuristic",
        "exact_matches_with_oracle",
        "all_gnn_colorings_valid",
    ]

    print(
        overall_df[
            overall_display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print("Comparison by graph family")
    print("--------------------------")

    family_display_columns = [
        "family",
        "num_graphs",
        "best_fixed_heuristics",
        "best_fixed_heuristic_total",
        "best_colpack5_oracle_total",
        "gnn_representative_total",
        "gnn_mean_total_across_seeds",
        "gnn_min_total_across_seeds",
        "gnn_max_total_across_seeds",
        "gnn_representative_gap_vs_oracle",
        "gnn_representative_gap_vs_best_fixed",
        "gnn_representative_exact_matches",
    ]

    print(
        family_df[
            family_display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print(
        "Week 18 final baseline comparison completed."
    )
    print(
        "--------------------------------------------"
    )
    print(
        f"Saved per-graph table to: "
        f"{PER_GRAPH_OUTPUT}"
    )
    print(
        f"Saved family table to: "
        f"{FAMILY_OUTPUT}"
    )
    print(
        f"Saved overall table to: "
        f"{OVERALL_OUTPUT}"
    )


if __name__ == "__main__":
    main()