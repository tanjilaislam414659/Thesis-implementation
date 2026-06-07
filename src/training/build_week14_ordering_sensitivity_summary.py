from pathlib import Path

import pandas as pd


INPUT_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week14_expanded_benchmark.csv"
)

OUTPUT_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week14_ordering_sensitivity_summary.csv"
)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    summary = df.groupby("graph_id")["num_colors"].agg(["min", "max", "mean"])
    summary["color_range"] = summary["max"] - summary["min"]

    summary = summary.sort_values(
        ["color_range", "max"],
        ascending=[False, False],
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV)

    print(summary.to_string())
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()