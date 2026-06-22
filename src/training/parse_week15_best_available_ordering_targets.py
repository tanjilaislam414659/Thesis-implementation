"""
Build Week 15 best-available-ordering targets.

For each graph, this script selects the ColPack ordering with the lowest
number of colors among the available Week 14 outputs:

- SMALLEST_LAST
- LARGEST_FIRST
- NATURAL

Then it extracts the node ordering from the selected ColPack output and
creates node-level normalized target scores.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


COLPACK_OUTPUT_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "colpack_outputs_week13_expanded"
)

BENCHMARK_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week14_expanded_benchmark.csv"
)

OUTPUT_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/ordering_targets/"
    "best_available_ordering_targets_week15.csv"
)

AVAILABLE_ORDERINGS = {
    "SMALLEST_LAST": "smallest_last",
    "LARGEST_FIRST": "largest_first",
    "NATURAL": "natural",
}


ORDER_LINE_PATTERN = re.compile(
    r"Order Position\s+(\d+)\s+->\s+Vertex\s+(\d+)"
)


def choose_best_orderings(benchmark: pd.DataFrame) -> pd.DataFrame:
    available = benchmark[
        benchmark["ordering_name"].isin(AVAILABLE_ORDERINGS.keys())
    ].copy()

    if available.empty:
        raise ValueError("No benchmark rows found for available orderings.")

    # Stable tie-breaking: if two orderings use the same number of colors,
    # prefer SMALLEST_LAST, then LARGEST_FIRST, then NATURAL.
    tie_priority = {
        "SMALLEST_LAST": 0,
        "LARGEST_FIRST": 1,
        "NATURAL": 2,
    }

    available["tie_priority"] = available["ordering_name"].map(tie_priority)

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
        ]
    ]


def parse_ordering_file(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"ColPack output file not found: {path}")

    ordering = []

    with path.open(encoding="utf-8") as file:
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


def build_targets_for_graph(
    graph_id: str,
    ordering_name: str,
    num_vertices: int,
    num_colors: int,
) -> list[dict[str, object]]:
    file_suffix = AVAILABLE_ORDERINGS[ordering_name]
    output_file = COLPACK_OUTPUT_DIR / f"{graph_id}_{file_suffix}.txt"

    ordering = parse_ordering_file(output_file)

    if len(ordering) != num_vertices:
        raise ValueError(
            f"{graph_id}: parsed ordering length {len(ordering)} does not "
            f"match num_vertices {num_vertices}."
        )

    rows = []

    for order_position, node_id in ordering:
        if num_vertices > 1:
            target_score = (num_vertices - 1 - order_position) / (num_vertices - 1)
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
            }
        )

    return rows


def main() -> None:
    benchmark = pd.read_csv(BENCHMARK_CSV)
    best_orderings = choose_best_orderings(benchmark)

    print("Best available ordering per graph")
    print("---------------------------------")
    print(best_orderings.to_string(index=False))
    print()

    all_rows = []

    for row in best_orderings.itertuples(index=False):
        graph_rows = build_targets_for_graph(
            graph_id=row.graph_id,
            ordering_name=row.ordering_name,
            num_vertices=int(row.num_vertices),
            num_colors=int(row.num_colors),
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

    print()
    print(f"Saved best-available-ordering targets to: {OUTPUT_CSV}")
    print(f"Total target rows: {len(target_df)}")


if __name__ == "__main__":
    main()