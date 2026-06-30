from __future__ import annotations

import csv
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import torch

from src.models.gnn_node_scorer import GNNNodeScorer
from src.training.learned_ordering import scores_to_ordering
from src.training.load_pyg_splits import load_all_pyg_graphs, group_dataset_by_split
from src.training.ordered_greedy_coloring import (
    count_colors,
    greedy_color_with_ordering,
    is_valid_coloring,
)


PYG_DATA_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "pyg_data_week17_best_available_of_5_symmetry_breaking_features"
)

TRAINING_SUMMARY_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_symmetry_breaking_validation_color_selection_training_summary.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_validation_color_selected_checkpoint_per_graph_evaluation.csv"
)

SEED_SUMMARY_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_validation_color_selected_checkpoint_seed_summary.csv"
)


def load_training_rows() -> list[dict[str, str]]:
    if not TRAINING_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing training summary: {TRAINING_SUMMARY_CSV}")

    with TRAINING_SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Training summary has no rows.")

    return rows


def pyg_data_to_networkx_graph(data) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))

    edge_index = data.edge_index.detach().cpu()

    for source, target in edge_index.t().tolist():
        graph.add_edge(int(source), int(target))

    return graph


def load_model_from_checkpoint(checkpoint_path: Path) -> GNNNodeScorer:
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model = GNNNodeScorer(
        in_channels=checkpoint["input_dim"],
        hidden_channels=checkpoint["hidden_channels"],
        out_channels=checkpoint["out_channels"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def evaluate_model_on_graph(model: GNNNodeScorer, graph_data):
    graph = pyg_data_to_networkx_graph(graph_data)

    start_time = time.perf_counter()

    with torch.no_grad():
        predicted_scores = model(graph_data.x, graph_data.edge_index)

    learned_ordering = scores_to_ordering(predicted_scores)

    coloring = greedy_color_with_ordering(
        graph=graph,
        ordering=learned_ordering,
    )

    runtime_seconds = time.perf_counter() - start_time

    num_colors = count_colors(coloring)
    valid = is_valid_coloring(graph, coloring)

    return num_colors, valid, runtime_seconds


def main() -> None:
    training_rows = load_training_rows()

    dataset = load_all_pyg_graphs(PYG_DATA_DIR)
    grouped = group_dataset_by_split(dataset)

    evaluation_graphs = []
    for split_name in ["validation", "test"]:
        for data in grouped[split_name]:
            evaluation_graphs.append(data)

    print("Week 17 validation-color-selected checkpoint evaluation")
    print("------------------------------------------------------")
    print(f"Evaluation graphs: {[data.graph_id for data in evaluation_graphs]}")
    print()

    rows = []

    for training_row in training_rows:
        seed = int(training_row["seed"])
        checkpoint_path = Path(training_row["checkpoint_path"])

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        model = load_model_from_checkpoint(checkpoint_path)

        for graph_data in evaluation_graphs:
            num_colors, valid, runtime_seconds = evaluate_model_on_graph(
                model=model,
                graph_data=graph_data,
            )

            target_colors = int(graph_data.selected_num_colors)
            gap_from_target = int(num_colors) - target_colors

            result_row = {
                "graph_id": graph_data.graph_id,
                "split": graph_data.split,
                "group": graph_data.group,
                "seed": seed,
                "method_family": "gnn",
                "method_name": "GNNNodeScorer",
                "target_ordering": "WEEK17_BEST_AVAILABLE_OF_5",
                "feature_set": "WEEK17_SYMMETRY_BREAKING_25",
                "checkpoint_selection": "validation_total_colors_then_validation_loss",
                "ordering_name": "learned_ordering_from_predicted_scores",
                "num_vertices": int(graph_data.num_nodes),
                "num_edges": int(graph_data.edge_index.shape[1] // 2),
                "target_colors": target_colors,
                "num_colors": int(num_colors),
                "gap_from_target": gap_from_target,
                "valid": bool(valid),
                "runtime_seconds": runtime_seconds,
                "selected_teacher_ordering": graph_data.selected_ordering,
                "checkpoint_path": str(checkpoint_path),
            }

            rows.append(result_row)

            print(
                f"{graph_data.split:10s} | "
                f"{graph_data.graph_id:25s} | "
                f"seed {seed} | "
                f"colors {num_colors} | "
                f"target {target_colors} | "
                f"gap {gap_from_target} | "
                f"valid {valid}"
            )

    output_df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)

    seed_summary = (
        output_df.groupby(["seed", "split"])
        .agg(
            total_colors=("num_colors", "sum"),
            total_target_colors=("target_colors", "sum"),
            total_gap_from_target=("gap_from_target", "sum"),
            average_gap_from_target=("gap_from_target", "mean"),
            all_valid=("valid", "all"),
        )
        .reset_index()
        .sort_values(["split", "total_gap_from_target", "seed"])
    )

    seed_summary.to_csv(SEED_SUMMARY_CSV, index=False)

    print()
    print(f"Saved per-graph evaluation to: {OUTPUT_CSV}")
    print(f"Saved seed summary to: {SEED_SUMMARY_CSV}")
    print()

    print("Seed summary:")
    print(seed_summary.to_string(index=False))

    print()
    print("Best test seed:")
    test_summary = seed_summary[seed_summary["split"] == "test"].copy()
    print(
        test_summary.sort_values(["total_gap_from_target", "total_colors", "seed"])
        .head(1)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()