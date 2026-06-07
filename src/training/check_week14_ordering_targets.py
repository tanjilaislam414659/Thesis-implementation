from pathlib import Path

import pandas as pd


BENCHMARK_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week14_expanded_benchmark.csv"
)

TARGET_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "ordering_targets/smallest_last_ordering_targets_week14_expanded.csv"
)


def main() -> None:
    benchmark = pd.read_csv(BENCHMARK_CSV)
    targets = pd.read_csv(TARGET_CSV)

    smallest_last = benchmark[benchmark["ordering_name"] == "SMALLEST_LAST"][
        ["graph_id", "num_vertices"]
    ]

    target_counts = (
        targets.groupby("graph_id")
        .size()
        .reset_index(name="target_rows")
    )

    check = smallest_last.merge(target_counts, on="graph_id", how="left")
    check["matches"] = check["num_vertices"] == check["target_rows"]

    print(check.to_string(index=False))
    print()
    print(f"All matched: {check['matches'].all()}")

    if not check["matches"].all():
        raise ValueError("Some graphs have mismatched target row counts.")


    quality_summary = targets.groupby("graph_id").agg(
        num_rows=("node_id", "size"),
        unique_nodes=("node_id", "nunique"),
        unique_positions=("order_position", "nunique"),
        min_score=("target_score", "min"),
        max_score=("target_score", "max"),
    )

    quality_summary["nodes_ok"] = (
        quality_summary["num_rows"] == quality_summary["unique_nodes"]
    )
    quality_summary["positions_ok"] = (
        quality_summary["num_rows"] == quality_summary["unique_positions"]
    )
    quality_summary["score_range_ok"] = (
        (quality_summary["min_score"] >= 0.0)
        & (quality_summary["max_score"] <= 1.0)
    )

    print()
    print("Target quality checks:")
    print(quality_summary.to_string())
    print()

    all_quality_checks_passed = (
        quality_summary["nodes_ok"]
        & quality_summary["positions_ok"]
        & quality_summary["score_range_ok"]
    ).all()

    print(f"All target quality checks passed: {all_quality_checks_passed}")

    if not all_quality_checks_passed:
        raise ValueError(
            "Some graphs have invalid ordering targets. "
            "Check duplicate nodes, duplicate order positions, or target score range."
        )

if __name__ == "__main__":
    main()