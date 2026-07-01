from pathlib import Path
import re

import pandas as pd


INPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_gnn_training_summary.csv"
)

PER_GRAPH_OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_gnn_per_graph_results.csv"
)

PER_GAP_OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_gnn_per_gap_summary.csv"
)

OVERALL_OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_gnn_overall_summary.csv"
)


PER_GRAPH_PATTERN = re.compile(
    r"^(?P<graph_id>[^:]+):"
    r"base(?P<base_cycle_size>\d+)/"
    r"gap(?P<gap_level>\d+)/"
    r"gnn(?P<gnn_colors>\d+)/"
    r"target(?P<target_colors>\d+)/"
    r"colpack5(?P<colpack5_colors>\d+)$"
)


def parse_per_graph_string(seed: int, per_graph_string: str) -> list[dict]:
    rows = []

    parts = [part.strip() for part in per_graph_string.split(";")]

    for part in parts:
        match = PER_GRAPH_PATTERN.match(part)

        if match is None:
            raise ValueError(f"Could not parse per-graph result: {part}")

        row = match.groupdict()

        gnn_colors = int(row["gnn_colors"])
        target_colors = int(row["target_colors"])
        colpack5_colors = int(row["colpack5_colors"])

        rows.append(
            {
                "seed": seed,
                "split": "test",
                "graph_id": row["graph_id"],
                "base_cycle_size": int(row["base_cycle_size"]),
                "gap_level": int(row["gap_level"]),
                "gnn_colors": gnn_colors,
                "target_colors": target_colors,
                "colpack5_colors": colpack5_colors,
                "gap_from_target": gnn_colors - target_colors,
                "gap_from_colpack5": gnn_colors - colpack5_colors,
                "colors_saved_vs_colpack5": colpack5_colors - gnn_colors,
                "reached_target": gnn_colors == target_colors,
                "improved_over_colpack5": gnn_colors < colpack5_colors,
            }
        )

    return rows


def main() -> None:
    training_df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "seed",
        "final_test_per_graph_colors",
        "final_test_total_colors",
        "final_test_target_colors",
        "final_test_colpack5_colors",
        "final_test_colors_saved_vs_colpack5",
    }

    missing = required_columns - set(training_df.columns)

    if missing:
        raise ValueError(f"Training summary missing columns: {missing}")

    per_graph_rows = []

    for row in training_df.itertuples(index=False):
        per_graph_rows.extend(
            parse_per_graph_string(
                seed=int(row.seed),
                per_graph_string=row.final_test_per_graph_colors,
            )
        )

    per_graph_df = pd.DataFrame(per_graph_rows)

    per_gap_df = (
        per_graph_df
        .groupby("gap_level")
        .agg(
            num_seed_runs=("seed", "count"),
            target_colors=("target_colors", "first"),
            colpack5_colors=("colpack5_colors", "first"),
            mean_gnn_colors=("gnn_colors", "mean"),
            std_gnn_colors=("gnn_colors", "std"),
            min_gnn_colors=("gnn_colors", "min"),
            max_gnn_colors=("gnn_colors", "max"),
            mean_gap_from_target=("gap_from_target", "mean"),
            mean_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "mean"),
            min_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "min"),
            max_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "max"),
            target_reached_count=("reached_target", "sum"),
            improved_over_colpack5_count=("improved_over_colpack5", "sum"),
        )
        .reset_index()
    )

    overall_df = pd.DataFrame(
        [
            {
                "num_seeds": training_df["seed"].nunique(),
                "num_test_graphs_per_seed": per_graph_df["graph_id"].nunique(),
                "total_test_runs": len(per_graph_df),
                "mean_total_gnn_colors": training_df[
                    "final_test_total_colors"
                ].mean(),
                "std_total_gnn_colors": training_df[
                    "final_test_total_colors"
                ].std(),
                "min_total_gnn_colors": training_df[
                    "final_test_total_colors"
                ].min(),
                "max_total_gnn_colors": training_df[
                    "final_test_total_colors"
                ].max(),
                "target_total_colors": training_df[
                    "final_test_target_colors"
                ].iloc[0],
                "colpack5_total_colors": training_df[
                    "final_test_colpack5_colors"
                ].iloc[0],
                "mean_total_colors_saved_vs_colpack5": training_df[
                    "final_test_colors_saved_vs_colpack5"
                ].mean(),
                "min_total_colors_saved_vs_colpack5": training_df[
                    "final_test_colors_saved_vs_colpack5"
                ].min(),
                "max_total_colors_saved_vs_colpack5": training_df[
                    "final_test_colors_saved_vs_colpack5"
                ].max(),
            }
        ]
    )

    PER_GRAPH_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    per_graph_df.to_csv(PER_GRAPH_OUTPUT_CSV, index=False)
    per_gap_df.to_csv(PER_GAP_OUTPUT_CSV, index=False)
    overall_df.to_csv(OVERALL_OUTPUT_CSV, index=False)

    print("Week 17 heuristic-gap GNN result summary")
    print("----------------------------------------")
    print()
    print("Per-gap summary:")
    print(per_gap_df.to_string(index=False))
    print()
    print("Overall summary:")
    print(overall_df.to_string(index=False))
    print()
    print(f"Saved per-graph results to: {PER_GRAPH_OUTPUT_CSV}")
    print(f"Saved per-gap summary to: {PER_GAP_OUTPUT_CSV}")
    print(f"Saved overall summary to: {OVERALL_OUTPUT_CSV}")


if __name__ == "__main__":
    main()