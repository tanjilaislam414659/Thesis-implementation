from pathlib import Path

import pandas as pd


TARGET_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/ordering_targets/"
    "best_available_ordering_targets_week15.csv"
)

GRAPH_SUMMARY_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/graph_metadata/"
    "week13_expanded_graph_summary.csv"
)


def main() -> None:
    targets = pd.read_csv(TARGET_CSV)
    graph_summary = pd.read_csv(GRAPH_SUMMARY_CSV)

    target_summary = targets.groupby("graph_id").agg(
        target_rows=("node_id", "size"),
        unique_nodes=("node_id", "nunique"),
        unique_positions=("order_position", "nunique"),
        min_score=("target_score", "min"),
        max_score=("target_score", "max"),
    ).reset_index()

    expected = graph_summary[["graph_id", "graph_vertices"]]

    check = target_summary.merge(expected, on="graph_id", how="left")

    check["rows_match_vertices"] = (
        check["target_rows"] == check["graph_vertices"]
    )
    check["nodes_unique"] = (
        check["target_rows"] == check["unique_nodes"]
    )
    check["positions_unique"] = (
        check["target_rows"] == check["unique_positions"]
    )
    check["score_range_ok"] = (
        (check["min_score"] >= 0.0)
        & (check["max_score"] <= 1.0)
    )

    print(check.to_string(index=False))
    print()

    print(f"All row counts match vertices: {check['rows_match_vertices'].all()}")
    print(f"All node IDs unique: {check['nodes_unique'].all()}")
    print(f"All order positions unique: {check['positions_unique'].all()}")
    print(f"All score ranges valid: {check['score_range_ok'].all()}")

    all_ok = (
        check["rows_match_vertices"].all()
        and check["nodes_unique"].all()
        and check["positions_unique"].all()
        and check["score_range_ok"].all()
    )

    print()
    print(f"All best-available target checks passed: {all_ok}")

    if not all_ok:
        raise ValueError("Best-available ordering target validation failed.")


if __name__ == "__main__":
    main()