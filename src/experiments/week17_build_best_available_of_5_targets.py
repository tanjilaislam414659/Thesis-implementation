from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

BENCHMARK_CSVS = [
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "colpack_week15_five_ordering_benchmark.csv",
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_bickle_hard_cases_colpack_benchmark.csv",
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_structured_colpack_benchmark.csv",
]

COLPACK_OUTPUT_DIRS = [
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week13_expanded",
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week15_extra_orderings",
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week17_bickle_hard_cases",
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week17_structured_matrices",
]

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week17_best_available_of_5_ordering_targets.csv"
)

SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_best_available_of_5_target_summary.csv"
)

AVAILABLE_ORDERINGS = {
    "SMALLEST_LAST": "smallest_last",
    "LARGEST_FIRST": "largest_first",
    "NATURAL": "natural",
    "DYNAMIC_LARGEST_FIRST": "dynamic_largest_first",
    "INCIDENCE_DEGREE": "incidence_degree",
}

TIE_PRIORITY = {
    "DYNAMIC_LARGEST_FIRST": 0,
    "INCIDENCE_DEGREE": 1,
    "SMALLEST_LAST": 2,
    "LARGEST_FIRST": 3,
    "NATURAL": 4,
}

ORDER_LINE_PATTERN = re.compile(
    r"Order Position\s+(\d+)\s+->\s+Vertex\s+(\d+)"
)


def load_combined_benchmark() -> pd.DataFrame:
    frames = []

    for path in BENCHMARK_CSVS:
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark CSV: {path}")

        df = pd.read_csv(path)
        df["benchmark_source"] = str(path)
        frames.append(df)

    benchmark = pd.concat(frames, ignore_index=True)

    required_columns = {
        "graph_id",
        "ordering_name",
        "num_vertices",
        "num_edges",
        "num_colors",
    }

    missing = required_columns - set(benchmark.columns)
    if missing:
        raise ValueError(f"Benchmark is missing columns: {missing}")

    return benchmark


def choose_best_orderings(benchmark: pd.DataFrame) -> pd.DataFrame:
    available = benchmark[
        benchmark["ordering_name"].isin(AVAILABLE_ORDERINGS.keys())
    ].copy()

    if available.empty:
        raise ValueError("No benchmark rows found for available orderings.")

    available["tie_priority"] = available["ordering_name"].map(TIE_PRIORITY)

    best = (
        available.sort_values(
            ["graph_id", "num_colors", "tie_priority"]
        )
        .groupby("graph_id", as_index=False)
        .first()
    )

    return best[
        [
            "graph_id",
            "ordering_name",
            "num_vertices",
            "num_edges",
            "num_colors",
            "benchmark_source",
        ]
    ]


def parse_ordering_file(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"ColPack output file not found: {path}")

    ordering = []

    with path.open(encoding="utf-8", errors="ignore") as file:
        for line in file:
            match = ORDER_LINE_PATTERN.search(line)

            if match:
                order_position = int(match.group(1))
                vertex = int(match.group(2))
                ordering.append((order_position, vertex))

    if not ordering:
        raise ValueError(f"No ordering lines found in: {path}")

    ordering.sort(key=lambda item: item[0])
    return ordering


def find_colpack_output_file(graph_id: str, ordering_name: str) -> Path:
    file_suffix = AVAILABLE_ORDERINGS[ordering_name]
    output_filename = f"{graph_id}_{file_suffix}.txt"

    for output_dir in COLPACK_OUTPUT_DIRS:
        output_file = output_dir / output_filename

        if output_file.exists():
            return output_file

    raise FileNotFoundError(
        f"Could not find ColPack output file for graph={graph_id}, "
        f"ordering={ordering_name}. Expected filename: {output_filename}"
    )


def build_targets_for_graph(
    graph_id: str,
    ordering_name: str,
    num_vertices: int,
    num_colors: int,
    benchmark_source: str,
) -> list[dict[str, object]]:
    output_file = find_colpack_output_file(graph_id, ordering_name)
    ordering = parse_ordering_file(output_file)

    if len(ordering) != num_vertices:
        raise ValueError(
            f"{graph_id}: parsed ordering length {len(ordering)} does not "
            f"match num_vertices {num_vertices}."
        )

    rows = []

    for order_position, node_id in ordering:
        if num_vertices > 1:
            target_score = (
                num_vertices - 1 - order_position
            ) / (num_vertices - 1)
        else:
            target_score = 1.0

        rows.append(
            {
                "graph_id": graph_id,
                "selected_ordering": ordering_name,
                "selected_num_colors": num_colors,
                "node_id": node_id,
                "order_position": order_position,
                "target_score": target_score,
                "source_file": str(output_file),
                "benchmark_source": benchmark_source,
            }
        )

    return rows


def main() -> None:
    benchmark = load_combined_benchmark()
    best_orderings = choose_best_orderings(benchmark)

    print("Week 17 best available ordering per graph")
    print("----------------------------------------")
    print(best_orderings.to_string(index=False))
    print()

    all_rows = []

    for row in best_orderings.itertuples(index=False):
        graph_rows = build_targets_for_graph(
            graph_id=row.graph_id,
            ordering_name=row.ordering_name,
            num_vertices=int(row.num_vertices),
            num_colors=int(row.num_colors),
            benchmark_source=row.benchmark_source,
        )

        all_rows.extend(graph_rows)

        print(
            f"{row.graph_id}: selected {row.ordering_name} "
            f"with {row.num_colors} colors "
            f"({len(graph_rows)} target rows)"
        )

    target_df = pd.DataFrame(all_rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_csv(OUTPUT_CSV, index=False)

    summary = best_orderings.rename(
        columns={
            "ordering_name": "selected_ordering",
            "num_colors": "selected_num_colors",
        }
    )

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    print()
    print(f"Saved Week 17 best-available targets to: {OUTPUT_CSV}")
    print(f"Saved Week 17 target summary to: {SUMMARY_CSV}")
    print(f"Total target rows: {len(target_df)}")
    print(f"Total graphs: {target_df['graph_id'].nunique()}")


if __name__ == "__main__":
    main()