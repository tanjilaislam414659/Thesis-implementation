from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COLPACK_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_heuristic_gap_colpack_summary.csv"
)

TARGET_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week17_heuristic_gap_extra_c44_exact_optimal_ordering_targets.csv"
)

SPLIT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week17_heuristic_gap_extra_c44_split.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_heuristic_gap_extra_c44_exact_target_summary.csv"
)


BASE_SIZE = 44
GAP_LEVELS = [1, 2, 3, 4, 5]


def exact_coloring_cycle_square_3r_plus_2(n: int) -> dict[int, int]:
    if (n - 2) % 3 != 0:
        raise ValueError(f"Expected n = 3r + 2, got n={n}")

    colors = []

    for i in range(n - 5):
        colors.append(i % 3)

    colors.extend([0, 3, 1, 2, 3])

    return {node: colors[node] for node in range(n)}


def exact_coloring_join(base_n: int, copies: int) -> dict[int, int]:
    base_coloring = exact_coloring_cycle_square_3r_plus_2(base_n)
    coloring = {}

    for component_id in range(copies):
        node_offset = component_id * base_n
        color_offset = component_id * 4

        for old_node, old_color in base_coloring.items():
            new_node = node_offset + old_node
            coloring[new_node] = color_offset + old_color

    return coloring


def ordering_from_coloring(coloring: dict[int, int]) -> list[int]:
    return [
        node
        for node, _ in sorted(
            coloring.items(),
            key=lambda item: (item[1], item[0]),
        )
    ]


def graph_id_for(base_n: int, gap_level: int) -> str:
    if gap_level == 1:
        return f"week17_gap_cycle_square_c{base_n}"
    return f"week17_gap_join_c{base_n}_join_{gap_level}"


def coloring_for(base_n: int, gap_level: int) -> dict[int, int]:
    if gap_level == 1:
        return exact_coloring_cycle_square_3r_plus_2(base_n)
    return exact_coloring_join(base_n=base_n, copies=gap_level)


def main() -> None:
    colpack_summary = pd.read_csv(COLPACK_SUMMARY_PATH)

    target_rows = []
    split_rows = []
    summary_rows = []

    for gap_level in GAP_LEVELS:
        graph_id = graph_id_for(BASE_SIZE, gap_level)

        matching = colpack_summary[colpack_summary["graph_id"] == graph_id]
        if matching.empty:
            raise ValueError(f"Graph not found in ColPack summary: {graph_id}")

        row = matching.iloc[0]

        known_chromatic_number = int(row["known_chromatic_number"])
        best_colpack5_colors = int(row["best_colpack5_colors"])
        verified_gap = int(row["best_colpack5_gap_from_known"])

        if verified_gap != gap_level:
            raise ValueError(
                f"Unexpected gap for {graph_id}: "
                f"expected {gap_level}, got {verified_gap}"
            )

        coloring = coloring_for(base_n=BASE_SIZE, gap_level=gap_level)
        ordering = ordering_from_coloring(coloring)

        selected_num_colors = max(coloring.values()) + 1

        if selected_num_colors != known_chromatic_number:
            raise ValueError(
                f"Known color mismatch for {graph_id}: "
                f"target={selected_num_colors}, known={known_chromatic_number}"
            )

        num_nodes = len(ordering)

        for order_position, node_id in enumerate(ordering):
            if num_nodes == 1:
                target_score = 1.0
            else:
                target_score = 1.0 - (order_position / (num_nodes - 1))

            target_rows.append(
                {
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "order_position": order_position,
                    "target_score": target_score,
                    "selected_ordering": "EXACT_OPTIMAL_COLOR_CLASS_ORDER",
                    "selected_num_colors": selected_num_colors,
                    "known_chromatic_number": known_chromatic_number,
                    "split": "extra_test",
                    "graph_family": "heuristic_gap_controlled_extra_test",
                }
            )

        split_rows.append(
            {
                "graph_id": graph_id,
                "split": "extra_test",
                "graph_family": "heuristic_gap_controlled_extra_test",
                "base_cycle_size": BASE_SIZE,
                "gap_level": gap_level,
                "known_chromatic_number": known_chromatic_number,
                "best_colpack5_colors": best_colpack5_colors,
                "verified_gap": verified_gap,
            }
        )

        summary_rows.append(
            {
                "graph_id": graph_id,
                "split": "extra_test",
                "base_cycle_size": BASE_SIZE,
                "gap_level": gap_level,
                "num_vertices": len(ordering),
                "known_chromatic_number": known_chromatic_number,
                "best_colpack5_colors": best_colpack5_colors,
                "verified_gap": verified_gap,
                "selected_ordering": "EXACT_OPTIMAL_COLOR_CLASS_ORDER",
                "selected_num_colors": selected_num_colors,
            }
        )

    target_df = pd.DataFrame(target_rows)
    split_df = pd.DataFrame(split_rows)
    summary_df = pd.DataFrame(summary_rows)

    TARGET_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    target_df.to_csv(TARGET_OUTPUT_PATH, index=False)
    split_df.to_csv(SPLIT_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("Built extra C44 exact-optimal targets.")
    print()
    print(f"Graphs: {summary_df.shape[0]}")
    print(f"Target rows: {target_df.shape[0]}")
    print()
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved targets to: {TARGET_OUTPUT_PATH}")
    print(f"Saved split to: {SPLIT_OUTPUT_PATH}")
    print(f"Saved summary to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()