from pathlib import Path

import pandas as pd


SPLIT_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/splits/"
    "expanded_graph_split_week15.csv"
)

GRAPH_SUMMARY_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/graph_metadata/"
    "week13_expanded_graph_summary.csv"
)


def main() -> None:
    split_df = pd.read_csv(SPLIT_CSV)
    graph_summary = pd.read_csv(GRAPH_SUMMARY_CSV)

    expected_graphs = set(graph_summary["graph_id"])
    split_graphs = set(split_df["graph_id"])

    missing_graphs = sorted(expected_graphs - split_graphs)
    extra_graphs = sorted(split_graphs - expected_graphs)

    duplicate_rows = split_df[split_df.duplicated("graph_id", keep=False)]

    split_counts = split_df["split"].value_counts().sort_index()

    print("Split file:")
    print(SPLIT_CSV)
    print()

    print("Split counts:")
    print(split_counts.to_string())
    print()

    print(f"Expected number of graphs: {len(expected_graphs)}")
    print(f"Number of graphs in split file: {len(split_graphs)}")
    print()

    print(f"Missing graphs: {missing_graphs}")
    print(f"Extra graphs: {extra_graphs}")
    print()

    if duplicate_rows.empty:
        print("Duplicate graph rows: none")
    else:
        print("Duplicate graph rows:")
        print(duplicate_rows.to_string(index=False))

    all_ok = (
        len(missing_graphs) == 0
        and len(extra_graphs) == 0
        and duplicate_rows.empty
        and len(split_graphs) == len(expected_graphs)
    )

    print()
    print(f"All split checks passed: {all_ok}")

    if not all_ok:
        raise ValueError("Expanded split validation failed.")


if __name__ == "__main__":
    main()