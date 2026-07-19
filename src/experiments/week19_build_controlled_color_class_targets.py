from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import torch

from src.training.ordered_greedy_coloring import (
    count_colors,
    greedy_color_with_ordering,
    is_valid_coloring,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week18_controlled_data_scaling_manifest.csv"
)

OUTPUT_TARGET_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week19_controlled_color_class_targets.csv"
)

OUTPUT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week19_controlled_color_class_target_summary.csv"
)

FULL_CONDITION = "train_125_plus105"
EXPECTED_GRAPH_COUNT = 135
EXPECTED_SPLIT_COUNTS = {
    "train": 125,
    "validation": 5,
    "test": 5,
}
EXPECTED_RECONSTRUCTED_GRAPHS = 30
EXPECTED_STORED_GRAPHS = 105
EXPECTED_NODE_TARGET_ROWS = 15045


def load_torch_data(path: Path):
    try:
        return torch.load(
            path,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path)


def scalar_to_int(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(
            value.detach().cpu().item()
        )

    return int(value)


def exact_coloring_cycle_square_3r_plus_2(
    n: int,
) -> dict[int, int]:
    if (n - 2) % 3 != 0:
        raise ValueError(
            f"Expected n = 3r + 2, got n={n}."
        )

    colors = [
        node % 3
        for node in range(n - 5)
    ]
    colors.extend([0, 3, 1, 2, 3])

    return {
        node: colors[node]
        for node in range(n)
    }


def exact_coloring_join(
    base_n: int,
    copies: int,
) -> dict[int, int]:
    base_coloring = (
        exact_coloring_cycle_square_3r_plus_2(
            base_n
        )
    )

    coloring: dict[int, int] = {}

    for component_id in range(copies):
        node_offset = component_id * base_n
        color_offset = component_id * 4

        for old_node, old_color in (
            base_coloring.items()
        ):
            coloring[
                node_offset + old_node
            ] = color_offset + old_color

    return coloring


def recover_color_classes(
    data,
) -> tuple[torch.Tensor, str]:
    if hasattr(data, "known_colors"):
        colors = (
            data.known_colors
            .detach()
            .cpu()
            .view(-1)
            .to(torch.long)
        )
        return colors, "stored_week18_known_colors"

    required_attributes = {
        "base_cycle_size",
        "gap_level",
    }
    missing = [
        name
        for name in required_attributes
        if not hasattr(data, name)
    ]

    if missing:
        raise ValueError(
            f"{data.graph_id}: cannot reconstruct "
            f"color classes; missing {sorted(missing)}."
        )

    base_n = scalar_to_int(
        data.base_cycle_size
    )
    copies = scalar_to_int(
        data.gap_level
    )

    coloring = exact_coloring_join(
        base_n=base_n,
        copies=copies,
    )

    colors = torch.tensor(
        [
            coloring[node]
            for node in range(data.num_nodes)
        ],
        dtype=torch.long,
    )

    return colors, "reconstructed_week17_exact_formula"


def pyg_data_to_networkx_graph(data) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(
        range(data.num_nodes)
    )

    edge_index = (
        data.edge_index
        .detach()
        .cpu()
    )

    for source, target in (
        edge_index.t().tolist()
    ):
        graph.add_edge(
            int(source),
            int(target),
        )

    return graph


def validate_graph_labels(
    data,
    colors: torch.Tensor,
) -> dict[str, object]:
    graph_id = str(data.graph_id)
    num_nodes = int(data.num_nodes)

    if colors.numel() != num_nodes:
        raise ValueError(
            f"{graph_id}: found {colors.numel()} labels "
            f"for {num_nodes} nodes."
        )

    unique_colors = sorted(
        int(value)
        for value in colors.unique().tolist()
    )
    selected_num_colors = scalar_to_int(
        data.selected_num_colors
    )
    expected_colors = list(
        range(selected_num_colors)
    )

    if unique_colors != expected_colors:
        raise ValueError(
            f"{graph_id}: expected color labels "
            f"{expected_colors}, found {unique_colors}."
        )

    edge_index = (
        data.edge_index
        .detach()
        .cpu()
    )
    sources = edge_index[0]
    targets = edge_index[1]

    target_coloring_valid = bool(
        (
            colors[sources]
            != colors[targets]
        ).all().item()
    )

    if not target_coloring_valid:
        raise ValueError(
            f"{graph_id}: color-class labels do not "
            "form a valid coloring."
        )

    scores = (
        data.y
        .detach()
        .cpu()
        .view(-1)
        .to(torch.float32)
    )

    if scores.numel() != num_nodes:
        raise ValueError(
            f"{graph_id}: target-score count does not "
            "match the node count."
        )

    score_order_compatible = True

    for earlier_color, later_color in zip(
        unique_colors[:-1],
        unique_colors[1:],
    ):
        earlier_minimum = float(
            scores[
                colors == earlier_color
            ].min().item()
        )
        later_maximum = float(
            scores[
                colors == later_color
            ].max().item()
        )

        if earlier_minimum <= later_maximum:
            score_order_compatible = False
            break

    if not score_order_compatible:
        raise ValueError(
            f"{graph_id}: color-class order disagrees "
            "with the existing target-score direction."
        )

    graph = pyg_data_to_networkx_graph(data)
    class_ordering = sorted(
        range(num_nodes),
        key=lambda node: (
            int(colors[node].item()),
            node,
        ),
    )

    greedy_coloring = greedy_color_with_ordering(
        graph=graph,
        ordering=class_ordering,
    )
    greedy_num_colors = int(
        count_colors(greedy_coloring)
    )
    greedy_coloring_valid = bool(
        is_valid_coloring(
            graph,
            greedy_coloring,
        )
    )

    if not greedy_coloring_valid:
        raise ValueError(
            f"{graph_id}: class-based ordering produced "
            "an invalid greedy coloring."
        )

    if greedy_num_colors != selected_num_colors:
        raise ValueError(
            f"{graph_id}: class-based ordering used "
            f"{greedy_num_colors} colors; expected "
            f"{selected_num_colors}."
        )

    return {
        "num_nodes": num_nodes,
        "num_color_classes": len(unique_colors),
        "selected_num_colors": selected_num_colors,
        "target_coloring_valid": target_coloring_valid,
        "score_order_compatible": score_order_compatible,
        "greedy_num_colors": greedy_num_colors,
        "greedy_coloring_valid": greedy_coloring_valid,
    }


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest_df = pd.read_csv(MANIFEST_PATH)
    condition_df = manifest_df[
        manifest_df["condition"]
        == FULL_CONDITION
    ].copy()

    if len(condition_df) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} graphs, "
            f"found {len(condition_df)}."
        )

    split_counts = (
        condition_df["split"]
        .value_counts()
        .to_dict()
    )

    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected split counts: "
            f"{split_counts}."
        )

    target_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    condition_df = condition_df.sort_values(
        ["split", "graph_id"]
    ).reset_index(drop=True)

    for row in condition_df.itertuples(
        index=False
    ):
        source_path = (
            PROJECT_ROOT
            / str(row.source_pt_path)
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"PyG file not found: {source_path}"
            )

        data = load_torch_data(source_path)

        if str(data.graph_id) != str(row.graph_id):
            raise ValueError(
                f"Graph-ID mismatch for {source_path}."
            )

        colors, label_source = (
            recover_color_classes(data)
        )
        validation = validate_graph_labels(
            data=data,
            colors=colors,
        )

        scores = (
            data.y
            .detach()
            .cpu()
            .view(-1)
        )

        for node_id in range(data.num_nodes):
            target_rows.append(
                {
                    "graph_id": str(data.graph_id),
                    "node_id": node_id,
                    "known_color": int(
                        colors[node_id].item()
                    ),
                    "existing_target_score": float(
                        scores[node_id].item()
                    ),
                    "split": str(row.split),
                    "label_source": label_source,
                }
            )

        summary_rows.append(
            {
                "graph_id": str(data.graph_id),
                "split": str(row.split),
                "label_source": label_source,
                **validation,
                "source_pt_path": str(
                    row.source_pt_path
                ),
            }
        )

        print(
            f"Validated {data.graph_id}: "
            f"{validation['num_color_classes']} classes "
            f"({label_source})"
        )

    targets_df = pd.DataFrame(target_rows)
    summary_df = pd.DataFrame(summary_rows)

    if len(targets_df) != EXPECTED_NODE_TARGET_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_NODE_TARGET_ROWS} node "
            f"targets, found {len(targets_df)}."
        )

    source_counts = (
        summary_df["label_source"]
        .value_counts()
        .to_dict()
    )
    expected_source_counts = {
        "reconstructed_week17_exact_formula": (
            EXPECTED_RECONSTRUCTED_GRAPHS
        ),
        "stored_week18_known_colors": (
            EXPECTED_STORED_GRAPHS
        ),
    }

    if source_counts != expected_source_counts:
        raise ValueError(
            "Unexpected label-source counts: "
            f"{source_counts}."
        )

    if not bool(
        summary_df[
            [
                "target_coloring_valid",
                "score_order_compatible",
                "greedy_coloring_valid",
            ]
        ].all().all()
    ):
        raise ValueError(
            "At least one color-class validation failed."
        )

    OUTPUT_TARGET_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets_df.to_csv(
        OUTPUT_TARGET_CSV,
        index=False,
    )
    summary_df.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
    )

    print()
    print(
        "Week 19 controlled color-class targets "
        "validated successfully."
    )
    print(
        "------------------------------------------"
    )
    print(f"Graphs: {len(summary_df)}")
    print(f"Node targets: {len(targets_df)}")
    print(f"Split counts: {split_counts}")
    print(f"Label-source counts: {source_counts}")
    print(f"Saved targets to: {OUTPUT_TARGET_CSV}")
    print(f"Saved summary to: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()