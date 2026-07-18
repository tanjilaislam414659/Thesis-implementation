from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_additional_heterogeneous_candidate_graph_summary.csv"
)

COLPACK_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week18_additional_heterogeneous_generalization"
)

BENCHMARK_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_additional_heterogeneous_colpack_benchmark.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_additional_heterogeneous_colpack_summary.csv"
)

ORDERINGS = [
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]


def output_name(graph_id: str, ordering: str) -> str:
    return f"{graph_id}_{ordering.lower()}.txt"


def parse_colpack_output(path: Path) -> dict[str, object]:
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    def find_int(patterns: list[str]) -> int | None:
        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return int(match.group(1))

        return None

    def find_float(patterns: list[str]) -> float | None:
        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return float(match.group(1))

        return None

    num_colors = find_int(
        [
            r"Total\s+Colors\s*[:=]\s*(\d+)",
            r"Number\s+of\s+Colors\s*[:=]\s*(\d+)",
            r"Colors\s*[:=]\s*(\d+)",
        ]
    )

    num_vertices = find_int(
        [
            r"Vertex\s+Count\s*[:=]\s*(\d+)",
            r"Vertices\s*[:=]\s*(\d+)",
        ]
    )

    num_edges = find_int(
        [
            r"Edge\s+Count\s*[:=]\s*(\d+)",
            r"Edges\s*[:=]\s*(\d+)",
        ]
    )

    runtime = find_float(
        [
            r"Runtime\s*[:=]\s*([0-9.eE+-]+)",
            r"Time\s*[:=]\s*([0-9.eE+-]+)",
        ]
    )

    validity: bool | None = None

    if re.search(
        r"valid\s*[:=]\s*true",
        text,
        flags=re.IGNORECASE,
    ):
        validity = True

    elif re.search(
        r"valid\s*[:=]\s*false",
        text,
        flags=re.IGNORECASE,
    ):
        validity = False

    parse_success = (
        num_colors is not None
        and num_vertices is not None
        and num_edges is not None
    )

    return {
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "num_colors": num_colors,
        "runtime": runtime,
        "valid": validity,
        "parse_success": parse_success,
    }


def load_metadata() -> list[dict[str, str]]:
    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for {output_path.name}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Additional Week 18 metadata not found: "
            f"{METADATA_PATH}"
        )

    if not COLPACK_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"Additional ColPack output directory not found: "
            f"{COLPACK_OUTPUT_DIR}"
        )

    metadata_rows = load_metadata()

    metadata_by_graph = {
        row["graph_id"]: row
        for row in metadata_rows
    }

    benchmark_rows: list[dict[str, object]] = []
    missing_outputs: list[str] = []
    parse_failures: list[str] = []

    for metadata_row in metadata_rows:
        graph_id = metadata_row["graph_id"]

        for ordering in ORDERINGS:
            output_path = (
                COLPACK_OUTPUT_DIR
                / output_name(graph_id, ordering)
            )

            if not output_path.exists():
                missing_outputs.append(output_path.name)
                continue

            parsed = parse_colpack_output(output_path)

            if not parsed["parse_success"]:
                parse_failures.append(output_path.name)

            benchmark_rows.append(
                {
                    "graph_id": graph_id,
                    "family": metadata_row["family"],
                    "construction": metadata_row["construction"],
                    "generation_seed": metadata_row[
                        "generation_seed"
                    ],
                    "parameters": metadata_row["parameters"],
                    "source_type": (
                        "week18_additional_handcrafted_candidate"
                    ),
                    "num_vertices": parsed["num_vertices"],
                    "num_edges": parsed["num_edges"],
                    "density": metadata_row["density"],
                    "num_components": metadata_row[
                        "num_components"
                    ],
                    "minimum_degree": metadata_row[
                        "minimum_degree"
                    ],
                    "maximum_degree": metadata_row[
                        "maximum_degree"
                    ],
                    "average_degree": metadata_row[
                        "average_degree"
                    ],
                    "coloring_distance": 1,
                    "method_family": "colpack",
                    "method_name": "greedy_coloring",
                    "ordering_name": ordering,
                    "num_colors": parsed["num_colors"],
                    "runtime": parsed["runtime"],
                    "valid": parsed["valid"],
                    "parse_success": parsed["parse_success"],
                    "matrix_path": metadata_row["matrix_path"],
                    "output_file": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                }
            )

    expected_rows = len(metadata_rows) * len(ORDERINGS)

    if missing_outputs:
        raise FileNotFoundError(
            f"Missing {len(missing_outputs)} ColPack outputs. "
            f"First files: {missing_outputs[:10]}"
        )

    if parse_failures:
        raise ValueError(
            f"Failed to parse {len(parse_failures)} outputs. "
            f"First files: {parse_failures[:10]}"
        )

    if len(benchmark_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} benchmark rows, "
            f"but found {len(benchmark_rows)}."
        )

    write_csv(
        BENCHMARK_OUTPUT_PATH,
        benchmark_rows,
    )

    rows_by_graph: dict[
        str,
        list[dict[str, object]],
    ] = {}

    for row in benchmark_rows:
        graph_id = str(row["graph_id"])
        rows_by_graph.setdefault(graph_id, []).append(row)

    summary_rows: list[dict[str, object]] = []

    for graph_id in sorted(rows_by_graph):
        graph_rows = rows_by_graph[graph_id]
        metadata_row = metadata_by_graph[graph_id]

        colors_by_ordering = {
            str(row["ordering_name"]): int(row["num_colors"])
            for row in graph_rows
        }

        missing_orderings = [
            ordering
            for ordering in ORDERINGS
            if ordering not in colors_by_ordering
        ]

        if missing_orderings:
            raise ValueError(
                f"{graph_id} is missing orderings: "
                f"{missing_orderings}"
            )

        best_colors = min(colors_by_ordering.values())
        worst_colors = max(colors_by_ordering.values())
        ordering_gap = worst_colors - best_colors

        best_orderings = [
            ordering
            for ordering in ORDERINGS
            if colors_by_ordering[ordering] == best_colors
        ]

        summary_rows.append(
            {
                "graph_id": graph_id,
                "family": metadata_row["family"],
                "construction": metadata_row["construction"],
                "generation_seed": metadata_row[
                    "generation_seed"
                ],
                "parameters": metadata_row["parameters"],
                "num_vertices": metadata_row["num_vertices"],
                "num_edges": metadata_row["num_edges"],
                "density": metadata_row["density"],
                "num_components": metadata_row[
                    "num_components"
                ],
                "minimum_degree": metadata_row[
                    "minimum_degree"
                ],
                "maximum_degree": metadata_row[
                    "maximum_degree"
                ],
                "average_degree": metadata_row[
                    "average_degree"
                ],
                "matrix_path": metadata_row["matrix_path"],
                "NATURAL": colors_by_ordering["NATURAL"],
                "LARGEST_FIRST": colors_by_ordering[
                    "LARGEST_FIRST"
                ],
                "DYNAMIC_LARGEST_FIRST": colors_by_ordering[
                    "DYNAMIC_LARGEST_FIRST"
                ],
                "INCIDENCE_DEGREE": colors_by_ordering[
                    "INCIDENCE_DEGREE"
                ],
                "SMALLEST_LAST": colors_by_ordering[
                    "SMALLEST_LAST"
                ],
                "best_colpack5_colors": best_colors,
                "worst_colpack5_colors": worst_colors,
                "ordering_gap": ordering_gap,
                "ordering_gap_at_least_2": (
                    ordering_gap >= 2
                ),
                "best_colpack5_orderings": "; ".join(
                    best_orderings
                ),
                "num_best_orderings": len(best_orderings),
                "unique_best_ordering": (
                    best_orderings[0]
                    if len(best_orderings) == 1
                    else ""
                ),
            }
        )

    write_csv(
        SUMMARY_OUTPUT_PATH,
        summary_rows,
    )

    useful_rows = [
        row
        for row in summary_rows
        if int(row["ordering_gap"]) >= 2
    ]

    unique_useful_rows = [
        row
        for row in useful_rows
        if int(row["num_best_orderings"]) == 1
    ]

    tied_useful_rows = [
        row
        for row in useful_rows
        if int(row["num_best_orderings"]) > 1
    ]

    useful_family_counts = Counter(
        str(row["family"])
        for row in useful_rows
    )

    unique_winner_counts = Counter(
        str(row["unique_best_ordering"])
        for row in unique_useful_rows
    )

    winner_appearance_counts: Counter[str] = Counter()

    for row in useful_rows:
        winners = str(
            row["best_colpack5_orderings"]
        ).split("; ")

        for winner in winners:
            winner_appearance_counts[winner] += 1

    print(
        "Parsed additional Week 18 heterogeneous "
        "ColPack outputs."
    )
    print("------------------------------------------------")
    print(f"Candidate graphs: {len(summary_rows)}")
    print(f"Parsed benchmark rows: {len(benchmark_rows)}")
    print(f"Expected benchmark rows: {expected_rows}")
    print(f"Missing outputs: {len(missing_outputs)}")
    print(f"Parse failures: {len(parse_failures)}")
    print()

    print(
        f"Candidates with ordering gap >= 2: "
        f"{len(useful_rows)}"
    )
    print(
        f"Gap >= 2 with a unique winner: "
        f"{len(unique_useful_rows)}"
    )
    print(
        f"Gap >= 2 with tied winners: "
        f"{len(tied_useful_rows)}"
    )
    print()

    print("Gap >= 2 candidates by family:")

    for family in [
        "barabasi_albert",
        "watts_strogatz",
        "stochastic_block_model",
    ]:
        print(
            f"  {family}: "
            f"{useful_family_counts.get(family, 0)}"
        )

    print()
    print("Unique winners among gap >= 2 candidates:")

    for ordering in ORDERINGS:
        print(
            f"  {ordering}: "
            f"{unique_winner_counts.get(ordering, 0)}"
        )

    print()
    print("Winner appearances among gap >= 2 candidates:")

    for ordering in ORDERINGS:
        print(
            f"  {ordering}: "
            f"{winner_appearance_counts.get(ordering, 0)}"
        )

    print()
    print(f"Saved full benchmark to: {BENCHMARK_OUTPUT_PATH}")
    print(f"Saved graph summary to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()