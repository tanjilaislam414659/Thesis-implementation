from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results/tables/gnn_node_scorer")

COLOR_SEED_SUMMARY_CSV = RESULTS_DIR / (
    "week17_validation_color_selected_checkpoint_seed_summary.csv"
)

LOSS_SEED_SUMMARY_CSV = RESULTS_DIR / (
    "week17_validation_loss_selected_checkpoint_seed_summary.csv"
)

COLOR_PER_GRAPH_CSV = RESULTS_DIR / (
    "week17_validation_color_selected_checkpoint_per_graph_evaluation.csv"
)

LOSS_PER_GRAPH_CSV = RESULTS_DIR / (
    "week17_validation_loss_selected_checkpoint_per_graph_evaluation.csv"
)

OUTPUT_METHOD_SUMMARY_CSV = RESULTS_DIR / (
    "week17_checkpoint_selection_method_summary.csv"
)

OUTPUT_BEST_SEED_PER_GRAPH_CSV = RESULTS_DIR / (
    "week17_checkpoint_selection_best_seed_per_graph_comparison.csv"
)


def load_seed_summary(path: Path, method_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing seed summary CSV: {path}")

    df = pd.read_csv(path)
    df["checkpoint_selection"] = method_name
    return df


def load_per_graph(path: Path, method_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing per-graph CSV: {path}")

    df = pd.read_csv(path)
    df["checkpoint_selection"] = method_name
    return df


def select_best_test_seed(seed_summary: pd.DataFrame) -> int:
    test_rows = seed_summary[seed_summary["split"] == "test"].copy()

    if test_rows.empty:
        raise ValueError("No test rows found in seed summary.")

    best_row = (
        test_rows.sort_values(
            ["total_gap_from_target", "total_colors", "seed"],
            ascending=[True, True, True],
        )
        .head(1)
        .iloc[0]
    )

    return int(best_row["seed"])


def main() -> None:
    color_seed_summary = load_seed_summary(
        COLOR_SEED_SUMMARY_CSV,
        "validation_total_colors_then_validation_loss",
    )

    loss_seed_summary = load_seed_summary(
        LOSS_SEED_SUMMARY_CSV,
        "validation_loss",
    )

    combined_seed_summary = pd.concat(
        [color_seed_summary, loss_seed_summary],
        ignore_index=True,
    )

    method_summary_rows = []

    for method_name, method_df in combined_seed_summary.groupby("checkpoint_selection"):
        test_rows = method_df[method_df["split"] == "test"].copy()

        best_seed = select_best_test_seed(method_df)
        best_seed_row = test_rows[test_rows["seed"] == best_seed].iloc[0]

        method_summary_rows.append(
            {
                "checkpoint_selection": method_name,
                "best_test_seed": int(best_seed),
                "best_test_total_colors": int(best_seed_row["total_colors"]),
                "best_test_target_colors": int(best_seed_row["total_target_colors"]),
                "best_test_gap_from_target": int(
                    best_seed_row["total_gap_from_target"]
                ),
                "best_test_average_gap_from_target": float(
                    best_seed_row["average_gap_from_target"]
                ),
                "best_test_all_valid": bool(best_seed_row["all_valid"]),
                "mean_test_total_colors_over_seeds": float(
                    test_rows["total_colors"].mean()
                ),
                "mean_test_gap_over_seeds": float(
                    test_rows["total_gap_from_target"].mean()
                ),
                "std_test_gap_over_seeds": float(
                    test_rows["total_gap_from_target"].std(ddof=0)
                ),
                "min_test_gap_over_seeds": int(
                    test_rows["total_gap_from_target"].min()
                ),
                "max_test_gap_over_seeds": int(
                    test_rows["total_gap_from_target"].max()
                ),
            }
        )

    method_summary = pd.DataFrame(method_summary_rows).sort_values(
        ["best_test_gap_from_target", "best_test_total_colors"]
    )

    color_per_graph = load_per_graph(
        COLOR_PER_GRAPH_CSV,
        "validation_total_colors_then_validation_loss",
    )

    loss_per_graph = load_per_graph(
        LOSS_PER_GRAPH_CSV,
        "validation_loss",
    )

    color_best_seed = select_best_test_seed(color_seed_summary)
    loss_best_seed = select_best_test_seed(loss_seed_summary)

    color_best_graphs = color_per_graph[
        (color_per_graph["split"] == "test")
        & (color_per_graph["seed"] == color_best_seed)
    ].copy()

    loss_best_graphs = loss_per_graph[
        (loss_per_graph["split"] == "test")
        & (loss_per_graph["seed"] == loss_best_seed)
    ].copy()

    merged = color_best_graphs.merge(
        loss_best_graphs,
        on=["graph_id", "split", "group", "target_colors"],
        suffixes=("_color_selection", "_loss_selection"),
    )

    best_seed_per_graph = merged[
        [
            "graph_id",
            "group",
            "target_colors",
            "seed_color_selection",
            "num_colors_color_selection",
            "gap_from_target_color_selection",
            "valid_color_selection",
            "seed_loss_selection",
            "num_colors_loss_selection",
            "gap_from_target_loss_selection",
            "valid_loss_selection",
        ]
    ].copy()

    best_seed_per_graph["color_selection_minus_loss_selection"] = (
        best_seed_per_graph["num_colors_color_selection"]
        - best_seed_per_graph["num_colors_loss_selection"]
    )

    best_seed_per_graph = best_seed_per_graph.sort_values("graph_id")

    OUTPUT_METHOD_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    method_summary.to_csv(OUTPUT_METHOD_SUMMARY_CSV, index=False)
    best_seed_per_graph.to_csv(OUTPUT_BEST_SEED_PER_GRAPH_CSV, index=False)

    print("Week 17 checkpoint-selection comparison")
    print("--------------------------------------")
    print()
    print("Method summary:")
    print(method_summary.to_string(index=False))
    print()
    print(f"Saved method summary to: {OUTPUT_METHOD_SUMMARY_CSV}")
    print()

    print("Best-seed per-graph comparison:")
    print(best_seed_per_graph.to_string(index=False))
    print()
    print(f"Saved per-graph comparison to: {OUTPUT_BEST_SEED_PER_GRAPH_CSV}")


if __name__ == "__main__":
    main()