from pathlib import Path
import pandas as pd


INPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_clean_heuristic_diversity_cases.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_clean_heuristic_diversity_oracle_summary.csv"
)

ORDERINGS = [
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "graph_id",
        "best_colors",
        "worst_colors",
        "spread_best_to_worst",
    } | set(ORDERINGS)

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    for ordering in ORDERINGS:
        if df[ordering].isna().any():
            missing_graphs = df[df[ordering].isna()]["graph_id"].tolist()
            raise ValueError(
                f"Ordering {ordering} has missing values for graphs: {missing_graphs}"
            )

    oracle_total = int(df["best_colors"].sum())
    num_graphs = len(df)

    rows = []

    for ordering in ORDERINGS:
        total_colors = int(df[ordering].sum())
        gap_to_oracle = total_colors - oracle_total
        matched_oracle = int((df[ordering] == df["best_colors"]).sum())

        rows.append(
            {
                "method": ordering,
                "num_graphs": num_graphs,
                "total_colors": total_colors,
                "oracle_total_colors": oracle_total,
                "gap_to_best_of_5_oracle": gap_to_oracle,
                "average_gap_per_graph": gap_to_oracle / num_graphs,
                "graphs_matching_best_of_5": matched_oracle,
                "graphs_not_matching_best_of_5": num_graphs - matched_oracle,
            }
        )

    rows.append(
        {
            "method": "BEST_OF_5_ORACLE",
            "num_graphs": num_graphs,
            "total_colors": oracle_total,
            "oracle_total_colors": oracle_total,
            "gap_to_best_of_5_oracle": 0,
            "average_gap_per_graph": 0.0,
            "graphs_matching_best_of_5": num_graphs,
            "graphs_not_matching_best_of_5": 0,
        }
    )

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df.sort_values(
        by=[
            "gap_to_best_of_5_oracle",
            "graphs_not_matching_best_of_5",
            "method",
        ],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    print("Clean heuristic-diversity oracle summary")
    print("----------------------------------------")
    print()
    print(f"Graphs in clean diversity set: {num_graphs}")
    print(f"Best-of-5 oracle total colors: {oracle_total}")
    print()
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()