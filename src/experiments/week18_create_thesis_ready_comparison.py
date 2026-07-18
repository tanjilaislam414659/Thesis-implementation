from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_OVERALL_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_final_test_baseline_comparison_overall.csv"
)

INPUT_SEED_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_heterogeneous_test_seed_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
)

THESIS_COMPARISON_OUTPUT = (
    OUTPUT_DIR
    / "week18_thesis_ready_test_baseline_comparison.csv"
)

GNN_STABILITY_OUTPUT = (
    OUTPUT_DIR
    / "week18_thesis_ready_gnn_seed_stability.csv"
)


METHOD_LABELS = {
    "NATURAL": "Natural",
    "LARGEST_FIRST": "Largest First",
    "DYNAMIC_LARGEST_FIRST": "Dynamic Largest First",
    "INCIDENCE_DEGREE": "Incidence Degree",
    "SMALLEST_LAST": "Smallest Last",
    "BEST_OF_5_ORACLE": "Best-of-five oracle",
    "GNN_REPRESENTATIVE_SEED_1": "GNN representative model",
    "GNN_MEAN_ACROSS_5_SEEDS": "GNN mean over five seeds",
}


MAIN_METHODS = list(METHOD_LABELS.keys())


def format_exact_matches(
    method: str,
    value: float,
) -> str:
    if method == "GNN_MEAN_ACROSS_5_SEEDS":
        return f"{value:.1f} mean"

    if float(value).is_integer():
        return str(int(value))

    return f"{value:.2f}"


def format_validity(value: object) -> str:
    if pd.isna(value):
        return "Not applicable"

    if bool(value):
        return "Yes"

    return "No"


def main() -> None:
    if not INPUT_OVERALL_CSV.exists():
        raise FileNotFoundError(
            f"Overall comparison file not found: "
            f"{INPUT_OVERALL_CSV}"
        )

    if not INPUT_SEED_CSV.exists():
        raise FileNotFoundError(
            f"Seed summary file not found: "
            f"{INPUT_SEED_CSV}"
        )

    overall_df = pd.read_csv(
        INPUT_OVERALL_CSV
    )

    seed_df = pd.read_csv(
        INPUT_SEED_CSV
    )

    required_overall_columns = {
        "method",
        "total_colors",
        "average_colors_per_graph",
        "gap_vs_best_of_5_oracle",
        "gap_vs_best_fixed_heuristic",
        "exact_matches_with_oracle",
        "all_gnn_colorings_valid",
    }

    missing_columns = (
        required_overall_columns
        - set(overall_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Overall table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    thesis_df = (
        overall_df[
            overall_df["method"].isin(
                MAIN_METHODS
            )
        ]
        .copy()
    )

    thesis_df["method_order"] = (
        thesis_df["method"].map(
            {
                method: index
                for index, method in enumerate(
                    MAIN_METHODS
                )
            }
        )
    )

    thesis_df = thesis_df.sort_values(
        "method_order"
    ).reset_index(drop=True)

    thesis_df["Method"] = (
        thesis_df["method"].map(
            METHOD_LABELS
        )
    )

    thesis_df["Total colors"] = (
        thesis_df["total_colors"]
    )

    thesis_df["Average colors per graph"] = (
        thesis_df[
            "average_colors_per_graph"
        ].round(3)
    )

    thesis_df["Gap to best-of-five oracle"] = (
        thesis_df[
            "gap_vs_best_of_5_oracle"
        ].round(3)
    )

    thesis_df["Gap to best fixed heuristic"] = (
        thesis_df[
            "gap_vs_best_fixed_heuristic"
        ].round(3)
    )

    thesis_df["Exact oracle matches"] = (
        thesis_df.apply(
            lambda row: format_exact_matches(
                method=str(row["method"]),
                value=float(
                    row[
                        "exact_matches_with_oracle"
                    ]
                ),
            ),
            axis=1,
        )
    )

    thesis_df["All GNN colorings valid"] = (
        thesis_df[
            "all_gnn_colorings_valid"
        ].apply(format_validity)
    )

    thesis_output = thesis_df[
        [
            "Method",
            "Total colors",
            "Average colors per graph",
            "Gap to best-of-five oracle",
            "Gap to best fixed heuristic",
            "Exact oracle matches",
            "All GNN colorings valid",
        ]
    ]

    required_seed_columns = {
        "seed",
        "num_test_graphs",
        "total_gnn_colors",
        "total_target_colors",
        "total_gap_from_target",
        "exact_matches",
        "exact_match_rate",
        "all_valid",
    }

    missing_seed_columns = (
        required_seed_columns
        - set(seed_df.columns)
    )

    if missing_seed_columns:
        raise ValueError(
            f"Seed table is missing columns: "
            f"{sorted(missing_seed_columns)}"
        )

    stability_output = seed_df[
        [
            "seed",
            "num_test_graphs",
            "total_gnn_colors",
            "total_target_colors",
            "total_gap_from_target",
            "exact_matches",
            "exact_match_rate",
            "all_valid",
        ]
    ].copy()

    stability_output = stability_output.rename(
        columns={
            "seed": "Seed",
            "num_test_graphs": "Test graphs",
            "total_gnn_colors": "Total GNN colors",
            "total_target_colors": "Oracle colors",
            "total_gap_from_target": "Gap to oracle",
            "exact_matches": "Exact oracle matches",
            "exact_match_rate": "Exact match rate",
            "all_valid": "All colorings valid",
        }
    )

    stability_output[
        "Exact match rate"
    ] = stability_output[
        "Exact match rate"
    ].round(3)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    thesis_output.to_csv(
        THESIS_COMPARISON_OUTPUT,
        index=False,
    )

    stability_output.to_csv(
        GNN_STABILITY_OUTPUT,
        index=False,
    )

    print("Week 18 thesis-ready tables created")
    print("-----------------------------------")
    print()
    print("Main comparison:")
    print(
        thesis_output.to_string(
            index=False
        )
    )
    print()
    print("GNN seed stability:")
    print(
        stability_output.to_string(
            index=False
        )
    )
    print()
    print(
        f"Saved comparison table to: "
        f"{THESIS_COMPARISON_OUTPUT}"
    )
    print(
        f"Saved stability table to: "
        f"{GNN_STABILITY_OUTPUT}"
    )


if __name__ == "__main__":
    main()