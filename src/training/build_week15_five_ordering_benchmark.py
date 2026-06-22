from pathlib import Path
import pandas as pd


WEEK14_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week14_expanded_benchmark.csv"
)

EXTRA_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week15_extra_orderings_benchmark.csv"
)

OUTPUT_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week15_five_ordering_benchmark.csv"
)


def main():
    week14_df = pd.read_csv(WEEK14_CSV)
    extra_df = pd.read_csv(EXTRA_CSV)

    combined_df = pd.concat([week14_df, extra_df], ignore_index=True)

    combined_df = combined_df.sort_values(
        by=["graph_id", "ordering_name"]
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Week 14 rows: {len(week14_df)}")
    print(f"Extra Week 15 rows: {len(extra_df)}")
    print(f"Combined rows: {len(combined_df)}")
    print()
    print("Orderings:")
    print(sorted(combined_df["ordering_name"].unique()))
    print()
    print("Rows per ordering:")
    print(combined_df["ordering_name"].value_counts().sort_index().to_string())
    print()
    print(f"Saved combined benchmark table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()