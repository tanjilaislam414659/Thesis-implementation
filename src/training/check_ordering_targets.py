"""
Validate node-level ordering targets extracted from ColPack outputs.

This script checks that the SMALLEST_LAST ordering target CSV is structurally valid:
1. one target row per node,
2. no duplicate node IDs,
3. consecutive ordering positions,
4. normalized target scores in [0, 1].
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


TARGET_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "ordering_targets/smallest_last_ordering_targets.csv"
)

EXPECTED_NODE_COUNTS = {
    "ash85": 85,
    "can_24": 24,
    "hess_pat": 43,
    "hess_pat_small": 10,
    "jac_pat": 43,
}


def load_targets() -> dict[str, list[dict[str, str]]]:
    """
    Load ordering target rows grouped by graph_id.
    """

    if not TARGET_CSV.exists():
        raise FileNotFoundError(f"Ordering target CSV not found: {TARGET_CSV}")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    with TARGET_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "graph_id",
            "node_id",
            "order_position",
            "target_score",
            "ordering_name",
            "output_file",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Target CSV must contain columns {required_columns}, "
                f"got {reader.fieldnames}."
            )

        for row in reader:
            grouped_rows[row["graph_id"]].append(row)

    return grouped_rows


def validate_graph_targets(graph_id: str, rows: list[dict[str, str]]) -> None:
    """
    Validate ordering target rows for one graph.
    """

    expected_count = EXPECTED_NODE_COUNTS[graph_id]

    if len(rows) != expected_count:
        raise ValueError(
            f"{graph_id}: expected {expected_count} rows, got {len(rows)}."
        )

    node_ids = [int(row["node_id"]) for row in rows]
    order_positions = [int(row["order_position"]) for row in rows]
    target_scores = [float(row["target_score"]) for row in rows]

    if len(set(node_ids)) != len(node_ids):
        raise ValueError(f"{graph_id}: duplicate node IDs found.")

    expected_node_ids = set(range(expected_count))
    actual_node_ids = set(node_ids)

    if actual_node_ids != expected_node_ids:
        raise ValueError(
            f"{graph_id}: node IDs do not match expected range "
            f"0 to {expected_count - 1}."
        )

    expected_positions = list(range(expected_count))
    actual_positions = sorted(order_positions)

    if actual_positions != expected_positions:
        raise ValueError(
            f"{graph_id}: ordering positions are not consecutive from "
            f"0 to {expected_count - 1}."
        )

    if min(target_scores) < 0.0 or max(target_scores) > 1.0:
        raise ValueError(
            f"{graph_id}: target scores must lie in [0, 1], "
            f"got min={min(target_scores)}, max={max(target_scores)}."
        )

    print(f"{graph_id}: OK")
    print(f"  target rows: {len(rows)}")
    print(f"  score range: [{min(target_scores):.1f}, {max(target_scores):.1f}]")
    print()


def main() -> None:
    grouped_rows = load_targets()

    print("Ordering target consistency check")
    print("---------------------------------")

    for graph_id in EXPECTED_NODE_COUNTS:
        if graph_id not in grouped_rows:
            raise ValueError(f"Missing ordering targets for graph: {graph_id}")

        validate_graph_targets(graph_id, grouped_rows[graph_id])

    unexpected_graphs = set(grouped_rows) - set(EXPECTED_NODE_COUNTS)
    if unexpected_graphs:
        raise ValueError(f"Unexpected graph IDs in target CSV: {unexpected_graphs}")

    print("All ordering target checks passed.")


if __name__ == "__main__":
    main()