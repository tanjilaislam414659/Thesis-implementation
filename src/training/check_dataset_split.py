from pathlib import Path

import pandas as pd


BENCHMARK_CSV = Path("results/tables/initial_graph_coloring_benchmarks/colpack_initial_benchmark.csv")
SPLIT_CSV = Path("data/processed/initial_graph_coloring_dataset/splits/initial_graph_split.csv")


def main() -> None:
    benchmark_df = pd.read_csv(BENCHMARK_CSV)
    split_df = pd.read_csv(SPLIT_CSV)

    benchmark_graph_ids = set(benchmark_df["graph_id"].unique())
    split_graph_ids = set(split_df["graph_id"].unique())

    missing_in_split = sorted(benchmark_graph_ids - split_graph_ids)
    missing_in_benchmark = sorted(split_graph_ids - benchmark_graph_ids)

    print("Dataset split check")
    print("-------------------")
    print(f"Benchmark graph IDs: {sorted(benchmark_graph_ids)}")
    print(f"Split graph IDs: {sorted(split_graph_ids)}")
    print(f"Missing in split: {missing_in_split}")
    print(f"Missing in benchmark: {missing_in_benchmark}")
    print()
    print("Split counts:")
    print(split_df["split"].value_counts())

    if missing_in_split or missing_in_benchmark:
        raise ValueError("Split file and benchmark table do not match.")

    print()
    print("Split file is consistent with benchmark table.")


if __name__ == "__main__":
    main()