from pathlib import Path

import pandas as pd


BENCHMARK_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week14_expanded_benchmark.csv"
)

GRAPH_SUMMARY_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/graph_metadata/"
    "week13_expanded_graph_summary.csv"
)


def main() -> None:
    benchmark = pd.read_csv(BENCHMARK_CSV)
    graph_summary = pd.read_csv(GRAPH_SUMMARY_CSV)

    colpack_counts = (
        benchmark.groupby("graph_id")[["num_vertices", "num_edges"]]
        .first()
        .reset_index()
    )

    expected_counts = graph_summary[
        ["graph_id", "graph_vertices", "graph_edges"]
    ]

    check = colpack_counts.merge(expected_counts, on="graph_id", how="left")

    check["vertices_match"] = (
        check["num_vertices"] == check["graph_vertices"]
    )
    check["edges_match"] = (
        check["num_edges"] == check["graph_edges"]
    )

    print(check.to_string(index=False))
    print()
    print(f"All vertices match: {check['vertices_match'].all()}")
    print(f"All edges match: {check['edges_match'].all()}")

    if not check["vertices_match"].all():
        raise ValueError("Some ColPack vertex counts do not match the graph summary.")

    if not check["edges_match"].all():
        raise ValueError("Some ColPack edge counts do not match the graph summary.")


if __name__ == "__main__":
    main()