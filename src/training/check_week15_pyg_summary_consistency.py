from pathlib import Path

import pandas as pd


PYG_SUMMARY_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "pyg_data_week15_expanded/pyg_week15_expanded_dataset_summary.csv"
)

GRAPH_SUMMARY_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/graph_metadata/"
    "week13_expanded_graph_summary.csv"
)


def main() -> None:
    pyg_summary = pd.read_csv(PYG_SUMMARY_CSV)
    graph_summary = pd.read_csv(GRAPH_SUMMARY_CSV)

    merged = pyg_summary.merge(
        graph_summary[["graph_id", "graph_vertices", "graph_edges"]],
        on="graph_id",
        how="left",
    )

    merged["nodes_match"] = merged["num_nodes"] == merged["graph_vertices"]
    merged["edges_match"] = (
        merged["num_undirected_edges"] == merged["graph_edges"]
    )
    merged["targets_match_nodes"] = (
        merged["num_targets"] == merged["num_nodes"]
    )

    check_columns = [
        "graph_id",
        "split",
        "num_nodes",
        "graph_vertices",
        "num_undirected_edges",
        "graph_edges",
        "num_targets",
        "nodes_match",
        "edges_match",
        "targets_match_nodes",
    ]

    print(merged[check_columns].to_string(index=False))
    print()

    print(f"All node counts match: {merged['nodes_match'].all()}")
    print(f"All edge counts match: {merged['edges_match'].all()}")
    print(f"All target counts match nodes: {merged['targets_match_nodes'].all()}")

    all_ok = (
        merged["nodes_match"].all()
        and merged["edges_match"].all()
        and merged["targets_match_nodes"].all()
    )

    print()
    print(f"All PyG consistency checks passed: {all_ok}")

    if not all_ok:
        raise ValueError("Week 15 PyG summary consistency check failed.")


if __name__ == "__main__":
    main()