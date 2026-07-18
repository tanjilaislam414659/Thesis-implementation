from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week18_controlled_mixed_joins"
)

GRAPH_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_controlled_mixed_join_candidate_summary.csv"
)

BENCHMARK_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_controlled_mixed_join_colpack_benchmark.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_controlled_mixed_join_colpack_summary.csv"
)

ORDERINGS = [
    "SMALLEST_LAST",
    "LARGEST_FIRST",
    "NATURAL",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
]

EXPECTED_GRAPH_COUNT = 105
EXPECTED_OUTPUT_COUNT = EXPECTED_GRAPH_COUNT * len(ORDERINGS)


def find_integer(
    text: str,
    patterns: list[str],
) -> int | None:
    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    return None


def find_float(
    text: str,
    patterns: list[str],
) -> float | None:
    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return float(match.group(1))

    return None


def parse_colpack_output(
    output_path: Path,
) -> dict[str, int | float | None]:
    text = output_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    num_colors = find_integer(
        text,
        [
            r"Total\s+Colors\s*[:=]\s*(\d+)",
            r"Number\s+of\s+Colors\s*[:=]\s*(\d+)",
            r"Colors\s*[:=]\s*(\d+)",
        ],
    )

    num_vertices = find_integer(
        text,
        [
            r"Vertex\s+Count\s*[:=]\s*(\d+)",
            r"Vertices\s*[:=]\s*(\d+)",
        ],
    )

    num_edges = find_integer(
        text,
        [
            r"Edge\s+Count\s*[:=]\s*(\d+)",
            r"Edges\s*[:=]\s*(\d+)",
        ],
    )

    ordering_time = find_float(
        text,
        [
            r"Ordering\s+Time\s*[:=]\s*([0-9.eE+-]+)",
        ],
    )

    coloring_time = find_float(
        text,
        [
            r"Coloring\s+Time\s*[:=]\s*([0-9.eE+-]+)",
        ],
    )

    if num_colors is None:
        raise ValueError(
            f"Could not parse color count from: {output_path}"
        )

    if num_vertices is None:
        raise ValueError(
            f"Could not parse vertex count from: {output_path}"
        )

    if num_edges is None:
        raise ValueError(
            f"Could not parse edge count from: {output_path}"
        )

    total_runtime = None

    if (
        ordering_time is not None
        and coloring_time is not None
    ):
        total_runtime = (
            ordering_time
            + coloring_time
        )

    return {
        "num_colors": num_colors,
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "ordering_time": ordering_time,
        "coloring_time": coloring_time,
        "total_runtime": total_runtime,
    }


def graph_id_and_ordering_from_filename(
    output_path: Path,
) -> tuple[str, str]:
    stem = output_path.stem

    for ordering in sorted(
        ORDERINGS,
        key=len,
        reverse=True,
    ):
        suffix = (
            "_"
            + ordering.lower()
        )

        if stem.endswith(suffix):
            graph_id = stem[
                : -len(suffix)
            ]

            return (
                graph_id,
                ordering,
            )

    raise ValueError(
        "Could not infer graph ID and ordering from "
        f"filename: {output_path.name}"
    )


def load_candidate_summary() -> pd.DataFrame:
    if not GRAPH_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Candidate summary not found: "
            f"{GRAPH_SUMMARY_PATH}"
        )

    candidate_df = pd.read_csv(
        GRAPH_SUMMARY_PATH
    )

    if len(candidate_df) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} candidate graphs, "
            f"found {len(candidate_df)}."
        )

    if candidate_df[
        "graph_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in candidate summary."
        )

    if not bool(
        candidate_df[
            "known_coloring_valid"
        ].all()
    ):
        raise ValueError(
            "At least one candidate has an invalid "
            "known coloring."
        )

    expected_targets = (
        4
        * candidate_df[
            "num_components_joined"
        ]
    )

    if not (
        candidate_df[
            "known_chromatic_number"
        ]
        == expected_targets
    ).all():
        raise ValueError(
            "At least one known target is inconsistent "
            "with four colors per joined component."
        )

    return candidate_df


def build_full_benchmark(
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    output_paths = sorted(
        OUTPUT_DIR.glob("*.txt")
    )

    if len(output_paths) != EXPECTED_OUTPUT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_OUTPUT_COUNT} ColPack outputs, "
            f"found {len(output_paths)}."
        )

    candidate_lookup = (
        candidate_df
        .set_index("graph_id")
        .to_dict(orient="index")
    )

    benchmark_rows: list[
        dict[str, object]
    ] = []

    seen_combinations: set[
        tuple[str, str]
    ] = set()

    for output_path in output_paths:
        graph_id, ordering = (
            graph_id_and_ordering_from_filename(
                output_path
            )
        )

        combination = (
            graph_id,
            ordering,
        )

        if combination in seen_combinations:
            raise ValueError(
                "Duplicate graph-ordering output found: "
                f"{graph_id} | {ordering}"
            )

        seen_combinations.add(
            combination
        )

        if graph_id not in candidate_lookup:
            raise ValueError(
                f"Graph {graph_id} is missing from "
                "the candidate summary."
            )

        metadata = candidate_lookup[
            graph_id
        ]

        parsed = parse_colpack_output(
            output_path
        )

        expected_vertices = int(
            metadata["num_vertices"]
        )

        expected_edges = int(
            metadata["num_edges"]
        )

        if (
            parsed["num_vertices"]
            != expected_vertices
        ):
            raise ValueError(
                f"{graph_id}: parsed vertex count "
                f"{parsed['num_vertices']} does not match "
                f"summary count {expected_vertices}."
            )

        if (
            parsed["num_edges"]
            != expected_edges
        ):
            raise ValueError(
                f"{graph_id}: parsed edge count "
                f"{parsed['num_edges']} does not match "
                f"summary count {expected_edges}."
            )

        benchmark_rows.append(
            {
                "graph_id": graph_id,
                "family": metadata["family"],
                "construction": metadata[
                    "construction"
                ],
                "num_components_joined": int(
                    metadata[
                        "num_components_joined"
                    ]
                ),
                "component_cycle_sizes": metadata[
                    "component_cycle_sizes"
                ],
                "num_unique_component_sizes": int(
                    metadata[
                        "num_unique_component_sizes"
                    ]
                ),
                "minimum_component_size": int(
                    metadata[
                        "minimum_component_size"
                    ]
                ),
                "maximum_component_size": int(
                    metadata[
                        "maximum_component_size"
                    ]
                ),
                "num_vertices": int(
                    parsed["num_vertices"]
                ),
                "num_edges": int(
                    parsed["num_edges"]
                ),
                "known_chromatic_number": int(
                    metadata[
                        "known_chromatic_number"
                    ]
                ),
                "coloring_distance": 1,
                "method_family": "colpack",
                "method_name": "greedy_coloring",
                "ordering_name": ordering,
                "num_colors": int(
                    parsed["num_colors"]
                ),
                "ordering_time": parsed[
                    "ordering_time"
                ],
                "coloring_time": parsed[
                    "coloring_time"
                ],
                "total_runtime": parsed[
                    "total_runtime"
                ],
                "output_file": str(
                    output_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

    benchmark_df = pd.DataFrame(
        benchmark_rows
    )

    counts_per_graph = (
        benchmark_df
        .groupby("graph_id")
        .size()
    )

    if not (
        counts_per_graph
        == len(ORDERINGS)
    ).all():
        problematic = counts_per_graph[
            counts_per_graph
            != len(ORDERINGS)
        ]

        raise ValueError(
            "Some graphs do not have exactly five "
            "ColPack outputs:\n"
            + problematic.to_string()
        )

    return benchmark_df


def build_graph_summary(
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[
        dict[str, object]
    ] = []

    for graph_id, graph_df in (
        benchmark_df.groupby(
            "graph_id",
            sort=True,
        )
    ):
        first_row = graph_df.iloc[0]

        colors_by_ordering = {
            row["ordering_name"]: int(
                row["num_colors"]
            )
            for _, row in graph_df.iterrows()
        }

        missing_orderings = (
            set(ORDERINGS)
            - set(colors_by_ordering)
        )

        if missing_orderings:
            raise ValueError(
                f"{graph_id}: missing orderings "
                f"{sorted(missing_orderings)}"
            )

        known_target = int(
            first_row[
                "known_chromatic_number"
            ]
        )

        best_colors = min(
            colors_by_ordering.values()
        )

        worst_colors = max(
            colors_by_ordering.values()
        )

        best_orderings = sorted(
            ordering
            for ordering, colors
            in colors_by_ordering.items()
            if colors == best_colors
        )

        best_gap_from_target = (
            best_colors
            - known_target
        )

        worst_gap_from_target = (
            worst_colors
            - known_target
        )

        smallest_last_colors = (
            colors_by_ordering[
                "SMALLEST_LAST"
            ]
        )

        summary_rows.append(
            {
                "graph_id": graph_id,
                "family": first_row[
                    "family"
                ],
                "construction": first_row[
                    "construction"
                ],
                "num_components_joined": int(
                    first_row[
                        "num_components_joined"
                    ]
                ),
                "component_cycle_sizes": first_row[
                    "component_cycle_sizes"
                ],
                "num_unique_component_sizes": int(
                    first_row[
                        "num_unique_component_sizes"
                    ]
                ),
                "minimum_component_size": int(
                    first_row[
                        "minimum_component_size"
                    ]
                ),
                "maximum_component_size": int(
                    first_row[
                        "maximum_component_size"
                    ]
                ),
                "num_vertices": int(
                    first_row[
                        "num_vertices"
                    ]
                ),
                "num_edges": int(
                    first_row[
                        "num_edges"
                    ]
                ),
                "known_chromatic_number": (
                    known_target
                ),
                "SMALLEST_LAST": (
                    colors_by_ordering[
                        "SMALLEST_LAST"
                    ]
                ),
                "LARGEST_FIRST": (
                    colors_by_ordering[
                        "LARGEST_FIRST"
                    ]
                ),
                "NATURAL": (
                    colors_by_ordering[
                        "NATURAL"
                    ]
                ),
                "DYNAMIC_LARGEST_FIRST": (
                    colors_by_ordering[
                        "DYNAMIC_LARGEST_FIRST"
                    ]
                ),
                "INCIDENCE_DEGREE": (
                    colors_by_ordering[
                        "INCIDENCE_DEGREE"
                    ]
                ),
                "best_colpack5_colors": (
                    best_colors
                ),
                "worst_colpack5_colors": (
                    worst_colors
                ),
                "ordering_gap": (
                    worst_colors
                    - best_colors
                ),
                "best_colpack5_orderings": (
                    ", ".join(
                        best_orderings
                    )
                ),
                "num_best_colpack5_orderings": (
                    len(best_orderings)
                ),
                "smallest_last_gap_from_known": (
                    smallest_last_colors
                    - known_target
                ),
                "best_colpack5_gap_from_known": (
                    best_gap_from_target
                ),
                "worst_colpack5_gap_from_known": (
                    worst_gap_from_target
                ),
                "colpack5_reached_known_target": (
                    best_colors
                    == known_target
                ),
                "colpack5_stuck_above_known": (
                    best_colors
                    > known_target
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if len(summary_df) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} summary rows, "
            f"found {len(summary_df)}."
        )

    if (
        summary_df[
            "best_colpack5_gap_from_known"
        ]
        < 0
    ).any():
        raise ValueError(
            "A ColPack result is below the known "
            "chromatic number, which is inconsistent."
        )

    return summary_df


def print_analysis(
    summary_df: pd.DataFrame,
) -> None:
    reached_target = int(
        summary_df[
            "colpack5_reached_known_target"
        ].sum()
    )

    stuck_above_target = int(
        summary_df[
            "colpack5_stuck_above_known"
        ].sum()
    )

    gap_at_least_two = int(
        (
            summary_df[
                "best_colpack5_gap_from_known"
            ]
            >= 2
        ).sum()
    )

    print(
        "Week 18 controlled mixed-join "
        "ColPack analysis"
    )
    print(
        "-----------------------------------------"
    )
    print(
        f"Parsed graphs: {len(summary_df)}"
    )
    print(
        f"Reached known target: {reached_target}"
    )
    print(
        f"Stuck above known target: "
        f"{stuck_above_target}"
    )
    print(
        f"Gap of at least 2 colors: "
        f"{gap_at_least_two}"
    )
    print()

    print(
        "Best-of-five gap distribution:"
    )

    gap_distribution = (
        summary_df[
            "best_colpack5_gap_from_known"
        ]
        .value_counts()
        .sort_index()
        .rename_axis("gap")
        .rename("num_graphs")
    )

    print(
        gap_distribution.to_string()
    )
    print()

    print(
        "Difficulty by component count:"
    )

    component_summary = (
        summary_df
        .groupby(
            "num_components_joined"
        )
        .agg(
            num_graphs=(
                "graph_id",
                "count",
            ),
            mean_best_gap=(
                "best_colpack5_gap_from_known",
                "mean",
            ),
            minimum_best_gap=(
                "best_colpack5_gap_from_known",
                "min",
            ),
            maximum_best_gap=(
                "best_colpack5_gap_from_known",
                "max",
            ),
            mean_ordering_gap=(
                "ordering_gap",
                "mean",
            ),
            hard_gap_at_least_2=(
                "best_colpack5_gap_from_known",
                lambda values: int(
                    (values >= 2).sum()
                ),
            ),
        )
        .round(3)
    )

    print(
        component_summary.to_string()
    )
    print()

    ordering_win_counts = {
        ordering: 0
        for ordering in ORDERINGS
    }

    for best_ordering_text in summary_df[
        "best_colpack5_orderings"
    ]:
        best_orderings = [
            ordering.strip()
            for ordering
            in str(
                best_ordering_text
            ).split(",")
        ]

        for ordering in best_orderings:
            ordering_win_counts[
                ordering
            ] += 1

    print(
        "Best-ordering appearances "
        "(ties counted for every winner):"
    )

    ordering_win_series = pd.Series(
        ordering_win_counts,
        name="best_appearances",
    ).sort_values(
        ascending=False
    )

    print(
        ordering_win_series.to_string()
    )


def main() -> None:
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"ColPack output directory not found: "
            f"{OUTPUT_DIR}"
        )

    candidate_df = (
        load_candidate_summary()
    )

    benchmark_df = (
        build_full_benchmark(
            candidate_df
        )
    )

    summary_df = (
        build_graph_summary(
            benchmark_df
        )
    )

    BENCHMARK_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_df = benchmark_df.sort_values(
        [
            "graph_id",
            "ordering_name",
        ]
    ).reset_index(drop=True)

    summary_df = summary_df.sort_values(
        [
            "num_components_joined",
            "num_vertices",
            "graph_id",
        ]
    ).reset_index(drop=True)

    benchmark_df.to_csv(
        BENCHMARK_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print_analysis(
        summary_df
    )

    print()
    print(
        f"Saved full benchmark to: "
        f"{BENCHMARK_OUTPUT_PATH}"
    )
    print(
        f"Saved graph summary to: "
        f"{SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()