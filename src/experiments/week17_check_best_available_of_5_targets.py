from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

TARGET_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week17_best_available_of_5_ordering_targets.csv"
)

SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_best_available_of_5_target_summary.csv"
)


REQUIRED_TARGET_COLUMNS = {
    "graph_id",
    "selected_ordering",
    "selected_num_colors",
    "node_id",
    "order_position",
    "target_score",
    "source_file",
    "benchmark_source",
}

REQUIRED_SUMMARY_COLUMNS = {
    "graph_id",
    "selected_ordering",
    "num_vertices",
    "num_edges",
    "selected_num_colors",
}


def main() -> None:
    targets = pd.read_csv(TARGET_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    missing_target_cols = REQUIRED_TARGET_COLUMNS - set(targets.columns)
    missing_summary_cols = REQUIRED_SUMMARY_COLUMNS - set(summary.columns)

    if missing_target_cols:
        raise ValueError(f"Target CSV missing columns: {missing_target_cols}")

    if missing_summary_cols:
        raise ValueError(f"Summary CSV missing columns: {missing_summary_cols}")

    print("Loaded Week 17 best-available targets.")
    print(f"Target rows: {len(targets)}")
    print(f"Target graphs: {targets['graph_id'].nunique()}")
    print(f"Summary graphs: {summary['graph_id'].nunique()}")
    print()

    errors = []

    for row in summary.itertuples(index=False):
        graph_id = row.graph_id
        expected_vertices = int(row.num_vertices)

        graph_targets = targets[targets["graph_id"] == graph_id].copy()

        if len(graph_targets) != expected_vertices:
            errors.append(
                f"{graph_id}: expected {expected_vertices} rows, "
                f"found {len(graph_targets)}"
            )
            continue

        if graph_targets["node_id"].nunique() != expected_vertices:
            errors.append(f"{graph_id}: duplicate or missing node_id values")

        if graph_targets["order_position"].nunique() != expected_vertices:
            errors.append(f"{graph_id}: duplicate or missing order_position values")

        expected_positions = set(range(expected_vertices))
        actual_positions = set(graph_targets["order_position"].astype(int))

        if actual_positions != expected_positions:
            errors.append(f"{graph_id}: order_position does not span 0..n-1")

        min_score = graph_targets["target_score"].min()
        max_score = graph_targets["target_score"].max()

        if min_score < 0 or max_score > 1:
            errors.append(
                f"{graph_id}: target_score outside [0, 1] "
                f"(min={min_score}, max={max_score})"
            )

        selected_orderings = graph_targets["selected_ordering"].unique()
        if len(selected_orderings) != 1:
            errors.append(f"{graph_id}: multiple selected_ordering values")

        selected_colors = graph_targets["selected_num_colors"].unique()
        if len(selected_colors) != 1:
            errors.append(f"{graph_id}: multiple selected_num_colors values")

        source_files = graph_targets["source_file"].unique()
        if len(source_files) != 1:
            errors.append(f"{graph_id}: multiple source_file values")

        for source_file in source_files:
            if not Path(source_file).exists():
                errors.append(f"{graph_id}: missing source file {source_file}")

    if errors:
        print("Validation failed.")
        print()
        for error in errors:
            print("ERROR:", error)
        raise SystemExit(1)

    print("Validation passed.")
    print()

    print("Selected ordering counts:")
    print(targets.groupby("graph_id")["selected_ordering"].first().value_counts())
    print()

    print("Selected color summary:")
    selected_summary = (
        targets.groupby("graph_id")
        .agg(
            selected_ordering=("selected_ordering", "first"),
            selected_num_colors=("selected_num_colors", "first"),
            num_vertices=("node_id", "count"),
        )
        .reset_index()
        .sort_values("graph_id")
    )

    print(selected_summary.to_string(index=False))


if __name__ == "__main__":
    main()