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
    / "week18_heterogeneous_candidate_graph_summary.csv"
)

COLPACK_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week18_heterogeneous_generalization"
)

BENCHMARK_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_heterogeneous_colpack_benchmark.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_heterogeneous_colpack_summary.csv"
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


def parse_optional_int(value: str | None) -> int | None:
    """
    Parse integers stored as strings, including values such as '4.0'.

    Empty values are returned as None.
    """
    if value is None:
        return None

    cleaned = value.strip()

    if cleaned == "":
        return None

    return int(float(cleaned))


def parse_colpack_output(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def find_int(patterns: list[str]) -> int | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)

            if match:
                return int(match.group(1))

        return None

    def find_float(patterns: list[str]) -> float | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)

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

    if re.search(r"valid\s*[:=]\s*true", text, flags=re.IGNORECASE):
        validity = True
    elif re.search(r"valid\s*[:=]\s*false", text, flags=re.IGNORECASE):
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
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(f"No rows available for {output_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            f"Week 18 metadata not found: {METADATA_PATH}"
        )

    if not COLPACK_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"Week 18 ColPack output directory not found: "
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
                    "labeling": metadata_row["labeling"],
                    "generation_seed": metadata_row["generation_seed"],
                    "parameter_1_name": metadata_row[
                        "parameter_1_name"
                    ],
                    "parameter_1_value": metadata_row[
                        "parameter_1_value"
                    ],
                    "parameter_2_name": metadata_row[
                        "parameter_2_name"
                    ],
                    "parameter_2_value": metadata_row[
                        "parameter_2_value"
                    ],
                    "source_type": "week18_handcrafted_candidate",
                    "num_vertices": parsed["num_vertices"],
                    "num_edges": parsed["num_edges"],
                    "density": metadata_row["density"],
                    "num_components": metadata_row["num_components"],
                    "coloring_distance": 1,
                    "method_family": "colpack",
                    "method_name": "greedy_coloring",
                    "ordering_name": ordering,
                    "num_colors": parsed["num_colors"],
                    "runtime": parsed["runtime"],
                    "valid": parsed["valid"],
                    "parse_success": parsed["parse_success"],
                    "known_chromatic_number": (
                        metadata_row["known_chromatic_number"]
                    ),
                    "output_file": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                }
            )

    expected_output_count = len(metadata_rows) * len(ORDERINGS)

    if missing_outputs:
        preview = ", ".join(missing_outputs[:10])

        raise FileNotFoundError(
            f"Missing {len(missing_outputs)} expected ColPack outputs. "
            f"First files: {preview}"
        )

    if len(benchmark_rows) != expected_output_count:
        raise ValueError(
            f"Expected {expected_output_count} benchmark rows, "
            f"but found {len(benchmark_rows)}."
        )

    if parse_failures:
        preview = ", ".join(parse_failures[:10])

        raise ValueError(
            f"Failed to parse {len(parse_failures)} ColPack outputs. "
            f"First files: {preview}"
        )

    write_csv(
        output_path=BENCHMARK_OUTPUT_PATH,
        rows=benchmark_rows,
    )

    benchmark_by_graph: dict[str, list[dict[str, object]]] = {}

    for row in benchmark_rows:
        graph_id = str(row["graph_id"])
        benchmark_by_graph.setdefault(graph_id, []).append(row)

    summary_rows: list[dict[str, object]] = []

    for graph_id in sorted(benchmark_by_graph):
        graph_rows = benchmark_by_graph[graph_id]
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

        known_chromatic_number = parse_optional_int(
            metadata_row["known_chromatic_number"]
        )

        best_gap_from_known = (
            best_colors - known_chromatic_number
            if known_chromatic_number is not None
            else None
        )

        summary_rows.append(
            {
                "graph_id": graph_id,
                "family": metadata_row["family"],
                "construction": metadata_row["construction"],
                "labeling": metadata_row["labeling"],
                "generation_seed": metadata_row["generation_seed"],
                "parameter_1_name": metadata_row[
                    "parameter_1_name"
                ],
                "parameter_1_value": metadata_row[
                    "parameter_1_value"
                ],
                "parameter_2_name": metadata_row[
                    "parameter_2_name"
                ],
                "parameter_2_value": metadata_row[
                    "parameter_2_value"
                ],
                "num_vertices": metadata_row["num_vertices"],
                "num_edges": metadata_row["num_edges"],
                "density": metadata_row["density"],
                "num_components": metadata_row["num_components"],
                "known_chromatic_number": known_chromatic_number,
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
                "ordering_gap_at_least_2": ordering_gap >= 2,
                "best_colpack5_orderings": "; ".join(
                    best_orderings
                ),
                "num_best_orderings": len(best_orderings),
                "unique_best_ordering": (
                    best_orderings[0]
                    if len(best_orderings) == 1
                    else ""
                ),
                "best_colpack5_gap_from_known": (
                    best_gap_from_known
                ),
                "best_colpack5_reaches_known": (
                    best_colors == known_chromatic_number
                    if known_chromatic_number is not None
                    else ""
                ),
            }
        )

    write_csv(
        output_path=SUMMARY_OUTPUT_PATH,
        rows=summary_rows,
    )

    family_counts = Counter(
        str(row["family"])
        for row in summary_rows
    )

    gap_2_rows = [
        row
        for row in summary_rows
        if bool(row["ordering_gap_at_least_2"])
    ]

    unique_winner_rows = [
        row
        for row in summary_rows
        if int(row["num_best_orderings"]) == 1
    ]

    tied_winner_rows = [
        row
        for row in summary_rows
        if int(row["num_best_orderings"]) > 1
    ]

    unique_winner_counts = Counter(
        str(row["unique_best_ordering"])
        for row in unique_winner_rows
    )

    winner_appearance_counts: Counter[str] = Counter()

    for row in summary_rows:
        winners = str(row["best_colpack5_orderings"]).split("; ")

        for winner in winners:
            winner_appearance_counts[winner] += 1

    print("Parsed Week 18 heterogeneous ColPack outputs.")
    print("---------------------------------------------")
    print(f"Candidate graphs: {len(summary_rows)}")
    print(f"Parsed benchmark rows: {len(benchmark_rows)}")
    print(f"Expected benchmark rows: {expected_output_count}")
    print(f"Missing outputs: {len(missing_outputs)}")
    print(f"Parse failures: {len(parse_failures)}")
    print()

    print("Candidate count by family:")

    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count}")

    print()
    print(
        f"Candidates with ordering gap >= 2: "
        f"{len(gap_2_rows)}"
    )
    print(f"Candidates with a unique winner: {len(unique_winner_rows)}")
    print(f"Candidates with tied winners: {len(tied_winner_rows)}")
    print()

    print("Unique winner counts:")

    if unique_winner_counts:
        for ordering in ORDERINGS:
            print(
                f"  {ordering}: "
                f"{unique_winner_counts.get(ordering, 0)}"
            )
    else:
        print("  No unique winners found.")

    print()
    print("Winner appearances, including ties:")

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