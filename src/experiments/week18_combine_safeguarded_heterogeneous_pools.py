from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INITIAL_SELECTED_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_safeguarded_heterogeneous_pool.csv"
)

INITIAL_METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_heterogeneous_candidate_graph_summary.csv"
)

ADDITIONAL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_additional_heterogeneous_colpack_summary.csv"
)

COMBINED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_combined_safeguarded_heterogeneous_pool.csv"
)

FAMILY_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_combined_safeguarded_pool_summary.csv"
)


ORDERINGS = [
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]


FINAL_COLUMNS = [
    "graph_id",
    "family",
    "construction",
    "source_pool",
    "selection_reason",
    "generation_seed",
    "labeling",
    "parameters",
    "num_vertices",
    "num_edges",
    "density",
    "num_components",
    "matrix_path",
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
    "unique_best_ordering",
]


def first_pool_parameters(row: pd.Series) -> str:
    parts: list[str] = []

    parameter_1_name = row.get("parameter_1_name")
    parameter_1_value = row.get("parameter_1_value")

    if pd.notna(parameter_1_name) and str(parameter_1_name).strip():
        parts.append(
            f"{str(parameter_1_name).strip()}={parameter_1_value}"
        )

    parameter_2_name = row.get("parameter_2_name")
    parameter_2_value = row.get("parameter_2_value")

    if pd.notna(parameter_2_name) and str(parameter_2_name).strip():
        parts.append(
            f"{str(parameter_2_name).strip()}={parameter_2_value}"
        )

    return "; ".join(parts)


def check_matrix_paths(df: pd.DataFrame) -> None:
    missing_paths: list[str] = []

    for row in df.itertuples(index=False):
        matrix_path = PROJECT_ROOT / str(row.matrix_path)

        if not matrix_path.exists():
            missing_paths.append(
                f"{row.graph_id}: {matrix_path}"
            )

    if missing_paths:
        raise FileNotFoundError(
            "Missing matrix files:\n"
            + "\n".join(missing_paths[:20])
        )


def main() -> None:
    required_files = [
        INITIAL_SELECTED_PATH,
        INITIAL_METADATA_PATH,
        ADDITIONAL_SUMMARY_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Week 18 file not found: {path}"
            )

    initial_selected = pd.read_csv(INITIAL_SELECTED_PATH)
    initial_metadata = pd.read_csv(INITIAL_METADATA_PATH)
    additional_summary = pd.read_csv(ADDITIONAL_SUMMARY_PATH)

    # Add matrix paths from the original candidate metadata.
    matrix_metadata = initial_metadata[
        ["graph_id", "matrix_path"]
    ].copy()

    initial_selected = initial_selected.merge(
        matrix_metadata,
        on="graph_id",
        how="left",
        validate="one_to_one",
    )

    if initial_selected["matrix_path"].isna().any():
        missing_graphs = initial_selected.loc[
            initial_selected["matrix_path"].isna(),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Missing matrix paths for initial graphs: {missing_graphs}"
        )

    if not (initial_selected["ordering_gap"] >= 2).all():
        raise ValueError(
            "Initial safeguarded pool contains a graph "
            "with ordering gap below 2."
        )

    initial_selected["source_pool"] = (
        "initial_erdos_renyi_and_crown"
    )

    initial_selected["parameters"] = initial_selected.apply(
        first_pool_parameters,
        axis=1,
    )

    # The first pool records its own detailed safeguard reason.
    if "selection_reason" not in initial_selected.columns:
        initial_selected["selection_reason"] = (
            "selected_from_initial_safeguarded_pool"
        )

    # Keep every structurally distinct additional graph with gap >= 2.
    additional_selected = additional_summary[
        additional_summary["ordering_gap"] >= 2
    ].copy()

    additional_selected["source_pool"] = (
        "additional_graph_families"
    )

    additional_selected["selection_reason"] = (
        "selected_additional_family_ordering_gap_at_least_2"
    )

    additional_selected["labeling"] = "generated"

    expected_additional_count = 17

    if len(additional_selected) != expected_additional_count:
        raise ValueError(
            f"Expected {expected_additional_count} useful additional "
            f"graphs, but found {len(additional_selected)}."
        )

    initial_standardized = initial_selected.reindex(
        columns=FINAL_COLUMNS
    )

    additional_standardized = additional_selected.reindex(
        columns=FINAL_COLUMNS
    )

    combined = pd.concat(
        [
            initial_standardized,
            additional_standardized,
        ],
        ignore_index=True,
    )

    if combined["graph_id"].duplicated().any():
        duplicates = combined.loc[
            combined["graph_id"].duplicated(keep=False),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Duplicate graph IDs in combined pool: {duplicates}"
        )

    required_nonempty_columns = [
        "graph_id",
        "family",
        "num_vertices",
        "num_edges",
        "matrix_path",
        "best_colpack5_colors",
        "ordering_gap",
        "best_colpack5_orderings",
        "num_best_orderings",
    ]

    for column in required_nonempty_columns:
        if combined[column].isna().any():
            missing_graphs = combined.loc[
                combined[column].isna(),
                "graph_id",
            ].tolist()

            raise ValueError(
                f"Missing values in {column}: {missing_graphs}"
            )

    combined["ordering_gap"] = pd.to_numeric(
        combined["ordering_gap"],
        errors="raise",
    ).astype(int)

    combined["num_best_orderings"] = pd.to_numeric(
        combined["num_best_orderings"],
        errors="raise",
    ).astype(int)

    for ordering in ORDERINGS:
        combined[ordering] = pd.to_numeric(
            combined[ordering],
            errors="raise",
        ).astype(int)

    if not (combined["ordering_gap"] >= 2).all():
        raise ValueError(
            "Combined pool contains a graph with ordering gap below 2."
        )

    check_matrix_paths(combined)

    combined = combined.sort_values(
        by=[
            "family",
            "num_vertices",
            "graph_id",
        ],
    ).reset_index(drop=True)

    COMBINED_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        COMBINED_OUTPUT_PATH,
        index=False,
    )

    family_summary = (
        combined
        .groupby("family", as_index=False)
        .agg(
            num_graphs=("graph_id", "count"),
            minimum_vertices=("num_vertices", "min"),
            maximum_vertices=("num_vertices", "max"),
            minimum_ordering_gap=("ordering_gap", "min"),
            maximum_ordering_gap=("ordering_gap", "max"),
            unique_winner_graphs=(
                "num_best_orderings",
                lambda values: int((values == 1).sum()),
            ),
            tied_winner_graphs=(
                "num_best_orderings",
                lambda values: int((values > 1).sum()),
            ),
        )
        .sort_values("family")
    )

    family_summary.to_csv(
        FAMILY_SUMMARY_PATH,
        index=False,
    )

    print("Week 18 combined safeguarded heterogeneous pool")
    print("------------------------------------------------")
    print(f"Initial safeguarded graphs: {len(initial_selected)}")
    print(
        f"Additional useful graphs: "
        f"{len(additional_selected)}"
    )
    print(f"Combined graph count: {len(combined)}")
    print()

    print("Combined pool by family:")

    for row in family_summary.itertuples(index=False):
        print(
            f"  {row.family}: "
            f"graphs={row.num_graphs}, "
            f"vertices={row.minimum_vertices}–"
            f"{row.maximum_vertices}, "
            f"gap={row.minimum_ordering_gap}–"
            f"{row.maximum_ordering_gap}, "
            f"unique={row.unique_winner_graphs}, "
            f"tied={row.tied_winner_graphs}"
        )

    print()
    print(
        f"Unique-winner graphs: "
        f"{int((combined['num_best_orderings'] == 1).sum())}"
    )

    print(
        f"Tied-winner graphs: "
        f"{int((combined['num_best_orderings'] > 1).sum())}"
    )

    print()
    print("All 68 matrix paths were verified.")
    print()
    print(f"Saved combined pool to: {COMBINED_OUTPUT_PATH}")
    print(f"Saved family summary to: {FAMILY_SUMMARY_PATH}")


if __name__ == "__main__":
    main()