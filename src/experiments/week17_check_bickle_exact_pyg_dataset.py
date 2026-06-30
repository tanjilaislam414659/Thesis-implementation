from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

PYG_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "pyg_data_week17_bickle_exact_symmetry_breaking"
)

PYG_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week17_bickle_exact_symmetry_breaking_pyg_summary.csv"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week17_bickle_exact_split.csv"
)


def load_torch_data(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def main() -> None:
    summary = pd.read_csv(PYG_SUMMARY_CSV)
    split = pd.read_csv(SPLIT_CSV)

    print("Loaded Week 17 exact-optimal Bickle PyG summary.")
    print(f"Summary rows: {len(summary)}")
    print(f"Split rows: {len(split)}")
    print()

    errors = []

    expected_graphs = set(split["graph_id"])
    summary_graphs = set(summary["graph_id"])

    if expected_graphs != summary_graphs:
        errors.append(
            f"Graph mismatch. Missing from summary: {sorted(expected_graphs - summary_graphs)}, "
            f"extra in summary: {sorted(summary_graphs - expected_graphs)}"
        )

    for row in summary.itertuples(index=False):
        path = Path(row.path)

        if not path.exists():
            errors.append(f"{row.graph_id}: missing PyG file {path}")
            continue

        data = load_torch_data(path)

        if data.graph_id != row.graph_id:
            errors.append(f"{row.graph_id}: stored graph_id mismatch.")

        if data.split != row.split:
            errors.append(f"{row.graph_id}: split mismatch.")

        if data.num_nodes != row.num_nodes:
            errors.append(f"{row.graph_id}: num_nodes mismatch.")

        if data.x.shape[0] != data.num_nodes:
            errors.append(f"{row.graph_id}: x rows do not match num_nodes.")

        if data.y.shape[0] != data.num_nodes:
            errors.append(f"{row.graph_id}: y rows do not match num_nodes.")

        if data.x.shape[1] != 25:
            errors.append(
                f"{row.graph_id}: expected 25 features, got {data.x.shape[1]}."
            )

        if int(data.selected_num_colors) != 4:
            errors.append(f"{row.graph_id}: selected_num_colors is not 4.")

        if int(data.known_chromatic_number) != 4:
            errors.append(f"{row.graph_id}: known_chromatic_number is not 4.")

        if int(data.exact_greedy_colors) != 4:
            errors.append(f"{row.graph_id}: exact_greedy_colors is not 4.")

        if data.edge_index.shape[0] != 2:
            errors.append(f"{row.graph_id}: edge_index first dimension is not 2.")

        if data.edge_index.shape[1] != row.num_directed_edges:
            errors.append(f"{row.graph_id}: directed edge count mismatch.")

        if float(data.y.min()) < 0 or float(data.y.max()) > 1:
            errors.append(f"{row.graph_id}: target scores outside [0, 1].")

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

    print("Feature count check:")
    print(summary["num_features"].value_counts())
    print()

    print("ColPack-5 stuck above optimum counts:")
    print(summary["colpack5_stuck_above_optimum"].value_counts())
    print()

    print("Dataset summary:")
    print(
        summary[
            [
                "graph_id",
                "split",
                "cycle_size",
                "num_nodes",
                "num_features",
                "target_colors",
                "best_colpack5_colors",
                "colpack5_stuck_above_optimum",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()