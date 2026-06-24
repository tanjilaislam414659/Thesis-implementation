"""
Build BEST_AVAILABLE_OF_5 ordering targets for the Week 16 larger-graph extension.

The target ordering for each graph is selected from the five ColPack orderings
according to minimum color count. Ties are broken using the same preference order
as the main BEST_AVAILABLE_OF_5 experiment.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


BENCHMARK_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week16_larger_graph_benchmark.csv"
)

COLPACK_OUTPUT_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "colpack_outputs_week16_larger_graphs"
)

OUTPUT_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/ordering_targets/"
    "best_available_of_5_ordering_targets_week16_larger_graphs.csv"
)

TIE_PRIORITY = {
    "DYNAMIC_LARGEST_FIRST": 0,
    "INCIDENCE_DEGREE": 1,
    "SMALLEST_LAST": 2,
    "LARGEST_FIRST": 3,
    "NATURAL": 4,
}


def output_file_name(graph_id: str, ordering: str) -> str:
    return f"{graph_id}_{ordering.lower()}.txt"


def read_benchmark_rows() -> list[dict[str, str]]:
    with BENCHMARK_CSV.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def choose_best_orderings(rows: list[dict[str, str]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        grouped.setdefault(row["graph_id"], []).append(row)

    best_ordering_by_graph = {}

    for graph_id, graph_rows in grouped.items():
        sorted_rows = sorted(
            graph_rows,
            key=lambda row: (
                int(row["num_colors"]),
                TIE_PRIORITY[row["ordering_name"]],
            ),
        )

        best_ordering_by_graph[graph_id] = sorted_rows[0]["ordering_name"]

    return best_ordering_by_graph


def parse_ordering(path: Path) -> list[tuple[int, int]]:
    text = path.read_text(encoding="utf-8")

    pattern = re.compile(r"Order Position\s+(\d+)\s+->\s+Vertex\s+(\d+)")

    ordering = []

    for match in pattern.finditer(text):
        order_position = int(match.group(1))
        node_id = int(match.group(2))
        ordering.append((order_position, node_id))

    if not ordering:
        raise ValueError(f"No ordering positions found in {path}")

    ordering.sort(key=lambda item: item[0])

    return ordering


def main() -> None:
    benchmark_rows = read_benchmark_rows()
    best_orderings = choose_best_orderings(benchmark_rows)

    output_rows = []

    for graph_id, ordering_name in sorted(best_orderings.items()):
        output_path = COLPACK_OUTPUT_DIR / output_file_name(graph_id, ordering_name)

        if not output_path.exists():
            raise FileNotFoundError(f"Missing ColPack output: {output_path}")

        ordering = parse_ordering(output_path)
        num_vertices = len(ordering)

        for order_position, node_id in ordering:
            if num_vertices <= 1:
                target_score = 0.0
            else:
                target_score = (num_vertices - 1 - order_position) / (num_vertices - 1)

            output_rows.append(
                {
                    "graph_id": graph_id,
                    "target_strategy": "BEST_AVAILABLE_OF_5_LARGER_GRAPHS",
                    "ordering_name": ordering_name,
                    "node_id": node_id,
                    "order_position": order_position,
                    "target_score": target_score,
                }
            )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "graph_id",
            "target_strategy",
            "ordering_name",
            "node_id",
            "order_position",
            "target_score",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("Built larger-graph BEST_AVAILABLE_OF_5 ordering targets")
    print(f"Graphs: {len(best_orderings)}")
    print(f"Rows: {len(output_rows)}")
    print("Selected best orderings:")
    for graph_id, ordering_name in sorted(best_orderings.items()):
        print(f"  {graph_id}: {ordering_name}")
    print(f"Saved CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()