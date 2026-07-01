from pathlib import Path
import pandas as pd


C41_PER_GRAPH_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_gnn_per_graph_results.csv"
)

C44_PER_GRAPH_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_extra_c44_gnn_per_graph_results.csv"
)

COMBINED_PER_GRAPH_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_combined_c41_c44_per_graph_results.csv"
)

COMBINED_PER_GAP_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_combined_c41_c44_per_gap_summary.csv"
)

COMBINED_PER_BASE_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_combined_c41_c44_per_base_summary.csv"
)

COMBINED_OVERALL_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_combined_c41_c44_overall_summary.csv"
)


def main() -> None:
    c41_df = pd.read_csv(C41_PER_GRAPH_CSV)
    c44_df = pd.read_csv(C44_PER_GRAPH_CSV)

    c41_df["evaluation_set"] = "test_c41"
    c44_df["evaluation_set"] = "extra_test_c44"

    combined_df = pd.concat([c41_df, c44_df], ignore_index=True)

    required_columns = {
        "seed",
        "graph_id",
        "base_cycle_size",
        "gap_level",
        "gnn_colors",
        "target_colors",
        "colpack5_colors",
        "gap_from_target",
        "colors_saved_vs_colpack5",
        "reached_target",
        "improved_over_colpack5",
    }

    missing = required_columns - set(combined_df.columns)

    if missing:
        raise ValueError(f"Combined results missing columns: {missing}")

    combined_df["colpack_error_above_target"] = (
        combined_df["colpack5_colors"] - combined_df["target_colors"]
    )

    per_gap_df = (
        combined_df
        .groupby("gap_level")
        .agg(
            num_runs=("seed", "count"),
            num_base_sizes=("base_cycle_size", "nunique"),
            target_colors=("target_colors", "first"),
            colpack5_colors=("colpack5_colors", "first"),
            colpack_error_above_target=("colpack_error_above_target", "first"),
            mean_gnn_colors=("gnn_colors", "mean"),
            std_gnn_colors=("gnn_colors", "std"),
            min_gnn_colors=("gnn_colors", "min"),
            max_gnn_colors=("gnn_colors", "max"),
            mean_gnn_error_above_target=("gap_from_target", "mean"),
            std_gnn_error_above_target=("gap_from_target", "std"),
            mean_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "mean"),
            min_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "min"),
            max_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "max"),
            target_reached_count=("reached_target", "sum"),
            improved_over_colpack5_count=("improved_over_colpack5", "sum"),
        )
        .reset_index()
    )

    per_base_seed_df = (
        combined_df
        .groupby(["base_cycle_size", "seed"])
        .agg(
            total_gnn_colors=("gnn_colors", "sum"),
            total_target_colors=("target_colors", "sum"),
            total_colpack5_colors=("colpack5_colors", "sum"),
            total_gap_from_target=("gap_from_target", "sum"),
            total_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "sum"),
        )
        .reset_index()
    )

    per_base_df = (
        per_base_seed_df
        .groupby("base_cycle_size")
        .agg(
            num_seeds=("seed", "count"),
            target_total_colors=("total_target_colors", "first"),
            colpack5_total_colors=("total_colpack5_colors", "first"),
            mean_gnn_total_colors=("total_gnn_colors", "mean"),
            std_gnn_total_colors=("total_gnn_colors", "std"),
            min_gnn_total_colors=("total_gnn_colors", "min"),
            max_gnn_total_colors=("total_gnn_colors", "max"),
            mean_total_gap_from_target=("total_gap_from_target", "mean"),
            mean_total_colors_saved_vs_colpack5=(
                "total_colors_saved_vs_colpack5",
                "mean",
            ),
            min_total_colors_saved_vs_colpack5=(
                "total_colors_saved_vs_colpack5",
                "min",
            ),
            max_total_colors_saved_vs_colpack5=(
                "total_colors_saved_vs_colpack5",
                "max",
            ),
        )
        .reset_index()
    )

    per_seed_combined_df = (
        combined_df
        .groupby("seed")
        .agg(
            total_gnn_colors=("gnn_colors", "sum"),
            total_target_colors=("target_colors", "sum"),
            total_colpack5_colors=("colpack5_colors", "sum"),
            total_gap_from_target=("gap_from_target", "sum"),
            total_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "sum"),
        )
        .reset_index()
    )

    overall_df = pd.DataFrame(
        [
            {
                "num_seeds": per_seed_combined_df["seed"].nunique(),
                "num_base_sizes": combined_df["base_cycle_size"].nunique(),
                "num_gap_levels": combined_df["gap_level"].nunique(),
                "num_graphs_per_seed": combined_df["graph_id"].nunique(),
                "total_runs": len(combined_df),
                "target_total_colors_per_seed": per_seed_combined_df[
                    "total_target_colors"
                ].iloc[0],
                "colpack5_total_colors_per_seed": per_seed_combined_df[
                    "total_colpack5_colors"
                ].iloc[0],
                "mean_gnn_total_colors_per_seed": per_seed_combined_df[
                    "total_gnn_colors"
                ].mean(),
                "std_gnn_total_colors_per_seed": per_seed_combined_df[
                    "total_gnn_colors"
                ].std(),
                "min_gnn_total_colors_per_seed": per_seed_combined_df[
                    "total_gnn_colors"
                ].min(),
                "max_gnn_total_colors_per_seed": per_seed_combined_df[
                    "total_gnn_colors"
                ].max(),
                "mean_total_gap_from_target_per_seed": per_seed_combined_df[
                    "total_gap_from_target"
                ].mean(),
                "mean_total_colors_saved_vs_colpack5_per_seed": (
                    per_seed_combined_df["total_colors_saved_vs_colpack5"].mean()
                ),
                "min_total_colors_saved_vs_colpack5_per_seed": (
                    per_seed_combined_df["total_colors_saved_vs_colpack5"].min()
                ),
                "max_total_colors_saved_vs_colpack5_per_seed": (
                    per_seed_combined_df["total_colors_saved_vs_colpack5"].max()
                ),
            }
        ]
    )

    COMBINED_PER_GRAPH_CSV.parent.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(COMBINED_PER_GRAPH_CSV, index=False)
    per_gap_df.to_csv(COMBINED_PER_GAP_CSV, index=False)
    per_base_df.to_csv(COMBINED_PER_BASE_CSV, index=False)
    overall_df.to_csv(COMBINED_OVERALL_CSV, index=False)

    print("Combined C41 + C44 heuristic-gap result summary")
    print("------------------------------------------------")
    print()

    print("Per-gap summary:")
    print(per_gap_df.to_string(index=False))
    print()

    print("Per-base-size summary:")
    print(per_base_df.to_string(index=False))
    print()

    print("Overall summary:")
    print(overall_df.to_string(index=False))
    print()

    print(f"Saved combined per-graph results to: {COMBINED_PER_GRAPH_CSV}")
    print(f"Saved combined per-gap summary to: {COMBINED_PER_GAP_CSV}")
    print(f"Saved combined per-base summary to: {COMBINED_PER_BASE_CSV}")
    print(f"Saved combined overall summary to: {COMBINED_OVERALL_CSV}")


if __name__ == "__main__":
    main()