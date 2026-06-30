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
    "pyg_data_week17_bickle_exact_symmetry_breaking"
)

TRAINING_SUMMARY_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_bickle_exact_gnn_training_summary.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_bickle_exact_gnn_per_graph_evaluation.csv"
)

SEED_SUMMARY_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_bickle_exact_gnn_seed_summary.csv"
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

    num_colors = int(count_colors(coloring))
    valid = bool(is_valid_coloring(graph, coloring))

    return num_colors, valid, runtime_seconds


def main() -> None:
    training_rows = load_training_rows()

    dataset = load_all_pyg_graphs(PYG_DATA_DIR)
    grouped = group_dataset_by_split(dataset)

    evaluation_graphs = []
    for split_name in ["validation", "test"]:
        for data in grouped[split_name]:
            evaluation_graphs.append(data)

    print("Week 17 exact-optimal Bickle GNN per-graph evaluation")
    print("----------------------------------------------------")
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
            known_chromatic_number = int(graph_data.known_chromatic_number)
            best_colpack5_colors = int(graph_data.best_colpack5_colors)

            gap_from_target = num_colors - target_colors
            gap_from_chromatic = num_colors - known_chromatic_number
            gap_from_colpack5 = num_colors - best_colpack5_colors

            row = {
                "graph_id": graph_data.graph_id,
                "split": graph_data.split,
                "cycle_size": int(graph_data.cycle_size),
                "seed": seed,
                "method_family": "gnn",
                "method_name": "GNNNodeScorer",
                "experiment": "WEEK17_BICKLE_EXACT_OPTIMAL",
                "feature_set": "WEEK17_SYMMETRY_BREAKING_25",
                "checkpoint_selection": "validation_total_colors_then_validation_loss",
                "num_vertices": int(graph_data.num_nodes),
                "num_edges": int(graph_data.edge_index.shape[1] // 2),
                "known_chromatic_number": known_chromatic_number,
                "target_colors": target_colors,
                "best_colpack5_colors": best_colpack5_colors,
                "gnn_num_colors": num_colors,
                "gnn_gap_from_target": gap_from_target,
                "gnn_gap_from_chromatic": gap_from_chromatic,
                "gnn_gap_from_colpack5": gap_from_colpack5,
                "gnn_beats_colpack5": num_colors < best_colpack5_colors,
                "gnn_matches_chromatic": num_colors == known_chromatic_number,
                "valid": valid,
                "runtime_seconds": runtime_seconds,
                "checkpoint_path": str(checkpoint_path),
            }

            rows.append(row)

            print(
                f"{graph_data.split:10s} | "
                f"{graph_data.graph_id:25s} | "
                f"seed {seed} | "
                f"GNN {num_colors} | "
                f"chi {known_chromatic_number} | "
                f"ColPack-5 {best_colpack5_colors} | "
                f"gap ColPack-5 {gap_from_colpack5} | "
                f"valid {valid}"
            )

    output_df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)

    seed_summary = (
        output_df.groupby(["seed", "split"])
        .agg(
            total_gnn_colors=("gnn_num_colors", "sum"),
            total_chromatic=("known_chromatic_number", "sum"),
            total_colpack5_colors=("best_colpack5_colors", "sum"),
            total_gap_from_chromatic=("gnn_gap_from_chromatic", "sum"),
            total_gap_from_colpack5=("gnn_gap_from_colpack5", "sum"),
            all_valid=("valid", "all"),
            all_match_chromatic=("gnn_matches_chromatic", "all"),
        )
        .reset_index()
        .sort_values(["split", "total_gnn_colors", "seed"])
    )

    seed_summary.to_csv(SEED_SUMMARY_CSV, index=False)

    print()
    print(f"Saved per-graph evaluation to: {OUTPUT_CSV}")
    print(f"Saved seed summary to: {SEED_SUMMARY_CSV}")
    print()

    print("Seed summary:")
    print(seed_summary.to_string(index=False))

    print()
    print("Test-only result:")
    test_summary = seed_summary[seed_summary["split"] == "test"].copy()
    print(test_summary.to_string(index=False))


if __name__ == "__main__":
    main()