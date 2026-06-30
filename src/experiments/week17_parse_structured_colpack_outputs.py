from pathlib import Path
import csv
import re


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week17_structured_matrices"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_structured_matrix_metadata.csv"
)

RESULT_CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_structured_colpack_benchmark.csv"
)

SUMMARY_CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_structured_colpack_summary.csv"
)

ORDERINGS = [
    "SMALLEST_LAST",
    "LARGEST_FIRST",
    "NATURAL",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
]


def parse_colpack_output(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    def find_int(patterns):
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def find_float(patterns):
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    num_colors = find_int(
        [
            r"Total\s+Colors\s*[:=]\s*(\d+)",
            r"Colors\s*[:=]\s*(\d+)",
            r"Number\s+of\s+Colors\s*[:=]\s*(\d+)",
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
            r"Runtime\s*[:=]\s*([0-9.]+)",
            r"Time\s*[:=]\s*([0-9.]+)",
        ]
    )

    valid = None
    if re.search(r"valid\s*[:=]\s*true", text, flags=re.IGNORECASE):
        valid = True
    elif re.search(r"valid\s*[:=]\s*false", text, flags=re.IGNORECASE):
        valid = False

    return {
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "num_colors": num_colors,
        "runtime": runtime,
        "valid": valid,
    }


def load_metadata():
    metadata = {}

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            metadata[row["graph_id"]] = row

    return metadata


def graph_id_and_ordering_from_filename(path: Path):
    stem = path.stem

    for ordering in sorted(ORDERINGS, key=len, reverse=True):
        suffix = "_" + ordering.lower()
        if stem.endswith(suffix):
            graph_id = stem[: -len(suffix)]
            return graph_id, ordering

    raise ValueError(f"Could not infer graph/order from filename: {path.name}")


def main():
    metadata = load_metadata()
    rows = []

    for txt_path in sorted(OUTPUT_DIR.glob("*.txt")):
        graph_id, ordering = graph_id_and_ordering_from_filename(txt_path)
        parsed = parse_colpack_output(txt_path)
        meta = metadata.get(graph_id, {})

        rows.append(
            {
                "graph_id": graph_id,
                "source_type": meta.get("source_type"),
                "source_name": meta.get("source_name"),
                "structure_family": meta.get("structure_family"),
                "matrix_file": meta.get("matrix_file"),
                "num_vertices": parsed["num_vertices"],
                "num_edges": parsed["num_edges"],
                "coloring_distance": 1,
                "method_family": "colpack",
                "method_name": "greedy_coloring",
                "ordering_name": ordering,
                "num_colors": parsed["num_colors"],
                "runtime": parsed["runtime"],
                "valid": parsed["valid"],
                "output_file": str(txt_path),
            }
        )

    RESULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []

    for graph_id in sorted(set(row["graph_id"] for row in rows)):
        graph_rows = [row for row in rows if row["graph_id"] == graph_id]
        meta = metadata.get(graph_id, {})

        colors_by_ordering = {
            row["ordering_name"]: row["num_colors"] for row in graph_rows
        }

        color_values = [
            color for color in colors_by_ordering.values() if color is not None
        ]

        best_colors = min(color_values)
        worst_colors = max(color_values)
        ordering_gap = worst_colors - best_colors

        best_orderings = [
            ordering
            for ordering, color in colors_by_ordering.items()
            if color == best_colors
        ]

        smallest_last_colors = colors_by_ordering.get("SMALLEST_LAST")
        smallest_last_gap = (
            smallest_last_colors - best_colors
            if smallest_last_colors is not None
            else None
        )

        if ordering_gap == 0:
            sensitivity_class = "ordering_insensitive"
        else:
            sensitivity_class = "ordering_sensitive"

        if smallest_last_gap == 0:
            smallest_last_status = "best_or_tied_best"
        else:
            smallest_last_status = "worse_than_best"

        summary_rows.append(
            {
                "graph_id": graph_id,
                "source_type": meta.get("source_type"),
                "source_name": meta.get("source_name"),
                "structure_family": meta.get("structure_family"),
                "num_vertices": meta.get("graph_vertices"),
                "num_edges": meta.get("graph_edges"),
                "SMALLEST_LAST": colors_by_ordering.get("SMALLEST_LAST"),
                "LARGEST_FIRST": colors_by_ordering.get("LARGEST_FIRST"),
                "NATURAL": colors_by_ordering.get("NATURAL"),
                "DYNAMIC_LARGEST_FIRST": colors_by_ordering.get(
                    "DYNAMIC_LARGEST_FIRST"
                ),
                "INCIDENCE_DEGREE": colors_by_ordering.get("INCIDENCE_DEGREE"),
                "best_colors": best_colors,
                "worst_colors": worst_colors,
                "ordering_gap": ordering_gap,
                "best_orderings": ", ".join(best_orderings),
                "smallest_last_gap_from_best": smallest_last_gap,
                "smallest_last_status": smallest_last_status,
                "sensitivity_class": sensitivity_class,
            }
        )

    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Parsed structured-matrix ColPack outputs.")
    print()

    for row in summary_rows:
        print(
            f"{row['graph_id']}: "
            f"family={row['structure_family']}, "
            f"SL={row['SMALLEST_LAST']}, "
            f"best={row['best_colors']}, "
            f"worst={row['worst_colors']}, "
            f"gap={row['ordering_gap']}, "
            f"best_orderings={row['best_orderings']}, "
            f"SL_status={row['smallest_last_status']}"
        )

    print()
    print(f"Saved full benchmark to: {RESULT_CSV_PATH}")
    print(f"Saved summary to: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()