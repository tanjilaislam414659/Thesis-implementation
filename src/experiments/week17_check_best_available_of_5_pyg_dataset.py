from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

PYG_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "pyg_data_week17_best_available_of_5_improved_features"
)

SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_best_available_of_5_pyg_summary.csv"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week17_best_available_of_5_split.csv"
)


def load_torch_data(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    split = pd.read_csv(SPLIT_CSV)

    print("Loaded Week 17 PyG summary.")
    print(f"Summary rows: {len(summary)}")
    print(f"Split rows: {len(split)}")
    print()

    errors = []

    expected_graphs = set(split["graph_id"])
    summary_graphs = set(summary["graph_id"])

    if expected_graphs != summary_graphs:
        missing_from_summary = expected_graphs - summary_graphs
        extra_in_summary = summary_graphs - expected_graphs

        if missing_from_summary:
            errors.append(f"Missing graphs in summary: {sorted(missing_from_summary)}")

        if extra_in_summary:
            errors.append(f"Extra graphs in summary: {sorted(extra_in_summary)}")

    for row in summary.itertuples(index=False):
        path = Path(row.path)

        if not path.exists():
            errors.append(f"{row.graph_id}: missing PyG file {path}")
            continue

        data = load_torch_data(path)

        if data.graph_id != row.graph_id:
            errors.append(
                f"{row.graph_id}: stored graph_id mismatch: {data.graph_id}"
            )

        if data.split != row.split:
            errors.append(
                f"{row.graph_id}: stored split mismatch: {data.split} vs {row.split}"
            )

        if data.num_nodes != row.num_nodes:
            errors.append(
                f"{row.graph_id}: num_nodes mismatch: {data.num_nodes} vs {row.num_nodes}"
            )

        if data.x.shape[0] != data.num_nodes:
            errors.append(
                f"{row.graph_id}: x rows {data.x.shape[0]} != num_nodes {data.num_nodes}"
            )

        if data.y.shape[0] != data.num_nodes:
            errors.append(
                f"{row.graph_id}: y rows {data.y.shape[0]} != num_nodes {data.num_nodes}"
            )

        if data.x.shape[1] != row.num_features:
            errors.append(
                f"{row.graph_id}: feature count mismatch: {data.x.shape[1]} vs {row.num_features}"
            )

        if data.x.shape[1] != 18:
            errors.append(
                f"{row.graph_id}: expected 18 improved features, got {data.x.shape[1]}"
            )

        if data.edge_index.shape[0] != 2:
            errors.append(
                f"{row.graph_id}: edge_index first dimension should be 2, got {data.edge_index.shape[0]}"
            )

        if data.edge_index.shape[1] != row.num_directed_edges:
            errors.append(
                f"{row.graph_id}: directed edge count mismatch: "
                f"{data.edge_index.shape[1]} vs {row.num_directed_edges}"
            )

        if float(data.y.min()) < 0 or float(data.y.max()) > 1:
            errors.append(
                f"{row.graph_id}: y target scores outside [0, 1]"
            )

        if not hasattr(data, "selected_ordering"):
            errors.append(f"{row.graph_id}: missing selected_ordering attribute")

        if not hasattr(data, "selected_num_colors"):
            errors.append(f"{row.graph_id}: missing selected_num_colors attribute")

    if errors:
        print("Validation failed.")
        print()
        for error in errors:
            print("ERROR:", error)
        raise SystemExit(1)

    print("Validation passed.")
    print()

    print("Split counts:")
    print(summary["split"].value_counts())
    print()

    print("Group counts:")
    print(summary["group"].value_counts())
    print()

    print("Feature count check:")
    print(summary["num_features"].value_counts())
    print()

    print("Selected ordering counts:")
    print(summary["selected_ordering"].value_counts())
    print()

    print("Dataset summary:")
    print(
        summary[
            [
                "graph_id",
                "split",
                "group",
                "num_nodes",
                "num_features",
                "selected_ordering",
                "selected_num_colors",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()