from pathlib import Path
import csv
import re


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week17_bickle_exact_family"
)

EXACT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_bickle_exact_family_summary.csv"
)

RESULT_CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_bickle_exact_family_colpack_benchmark.csv"
)

SUMMARY_CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_bickle_exact_family_colpack_summary.csv"
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


def load_exact_summary():
    rows = {}
    with EXACT_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["graph_id"]] = row
    return rows


def graph_id_and_ordering_from_filename(path: Path):
    stem = path.stem

    for ordering in sorted(ORDERINGS, key=len, reverse=True):
        suffix = "_" + ordering.lower()
        if stem.endswith(suffix):
            graph_id = stem[: -len(suffix)]
            return graph_id, ordering

    raise ValueError(f"Could not infer graph/order from filename: {path.name}")


def main():
    exact_summary = load_exact_summary()
    rows = []

    for txt_path in sorted(OUTPUT_DIR.glob("*.txt")):
        graph_id, ordering = graph_id_and_ordering_from_filename(txt_path)
        parsed = parse_colpack_output(txt_path)
        exact_row = exact_summary.get(graph_id, {})

        known_chromatic_number = exact_row.get("known_chromatic_number")

        rows.append(
            {
                "graph_id": graph_id,
                "source_type": "bickle_cycle_square_exact",
                "num_vertices": parsed["num_vertices"],
                "num_edges": parsed["num_edges"],
                "coloring_distance": 1,
                "method_family": "colpack",
                "method_name": "greedy_coloring",
                "ordering_name": ordering,
                "num_colors": parsed["num_colors"],
                "runtime": parsed["runtime"],
                "valid": parsed["valid"],
                "known_chromatic_number": known_chromatic_number,
                "output_file": str(txt_path),
            }
        )

    if not rows:
        raise ValueError(f"No ColPack output files found in {OUTPUT_DIR}")

    RESULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    graph_order = sorted(set(row["graph_id"] for row in rows))

    summary_rows = []

    for graph_id in graph_order:
        graph_rows = [row for row in rows if row["graph_id"] == graph_id]

        colors_by_ordering = {
            row["ordering_name"]: row["num_colors"]
            for row in graph_rows
        }

        color_values = [
            color for color in colors_by_ordering.values()
            if color is not None
        ]

        best_colors = min(color_values)
        worst_colors = max(color_values)
        ordering_gap = worst_colors - best_colors

        best_orderings = [
            ordering
            for ordering, color in colors_by_ordering.items()
            if color == best_colors
        ]

        exact_row = exact_summary.get(graph_id, {})
        known_chromatic_number = int(exact_row.get("known_chromatic_number", 4))

        smallest_last_colors = colors_by_ordering.get("SMALLEST_LAST")

        summary_row = {
            "graph_id": graph_id,
            "split": exact_row.get("split"),
            "cycle_size": exact_row.get("cycle_size"),
            "known_chromatic_number": known_chromatic_number,
            "SMALLEST_LAST": colors_by_ordering.get("SMALLEST_LAST"),
            "LARGEST_FIRST": colors_by_ordering.get("LARGEST_FIRST"),
            "NATURAL": colors_by_ordering.get("NATURAL"),
            "DYNAMIC_LARGEST_FIRST": colors_by_ordering.get(
                "DYNAMIC_LARGEST_FIRST"
            ),
            "INCIDENCE_DEGREE": colors_by_ordering.get("INCIDENCE_DEGREE"),
            "best_colpack5_colors": best_colors,
            "worst_colpack5_colors": worst_colors,
            "ordering_gap": ordering_gap,
            "best_colpack5_orderings": ", ".join(best_orderings),
            "smallest_last_gap_from_chromatic": (
                smallest_last_colors - known_chromatic_number
                if smallest_last_colors is not None
                else None
            ),
            "best_colpack5_gap_from_chromatic": (
                best_colors - known_chromatic_number
            ),
            "colpack5_stuck_above_optimum": best_colors > known_chromatic_number,
        }

        summary_rows.append(summary_row)

    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Parsed ColPack outputs for exact Bickle cycle-square family.")
    print()

    for row in summary_rows:
        print(
            f"{row['graph_id']} | "
            f"split={row['split']} | "
            f"chi={row['known_chromatic_number']} | "
            f"SL={row['SMALLEST_LAST']} | "
            f"best5={row['best_colpack5_colors']} | "
            f"best5-gap-chi={row['best_colpack5_gap_from_chromatic']} | "
            f"best_orderings={row['best_colpack5_orderings']}"
        )

    print()
    print(f"Saved full benchmark to: {RESULT_CSV_PATH}")
    print(f"Saved summary to: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()