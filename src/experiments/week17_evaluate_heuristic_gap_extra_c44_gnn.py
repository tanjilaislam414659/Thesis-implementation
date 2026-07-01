from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import pandas as pd
import torch

from src.models.gnn_node_scorer import GNNNodeScorer
from src.training.learned_ordering import scores_to_ordering
from src.training.load_pyg_splits import load_all_pyg_graphs
from src.training.ordered_greedy_coloring import (
    count_colors,
    greedy_color_with_ordering,
    is_valid_coloring,
)


PYG_DATA_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "pyg_data_week17_heuristic_gap_extra_c44_symmetry_breaking"
)

CHECKPOINT_DIR = Path(
    "results/models/gnn_node_scorer/"
    "week17_heuristic_gap_gnn_runs"
)

PER_GRAPH_OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_extra_c44_gnn_per_graph_results.csv"
)

PER_GAP_OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_extra_c44_gnn_per_gap_summary.csv"
)

OVERALL_OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_gap_extra_c44_gnn_overall_summary.csv"
)

SEEDS = [0, 1, 2, 3, 4]


def pyg_data_to_networkx_graph(data) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))

    edge_index = data.edge_index.detach().cpu()

    for source, target in edge_index.t().tolist():
        graph.add_edge(int(source), int(target))

    return graph


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    

def load_pyg_graph(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_extra_test_graphs(directory: Path) -> list:
    graph_paths = sorted(directory.glob("*.pt"))

    if not graph_paths:
        raise ValueError(f"No .pt files found in {directory}")

    graphs = []

    for path in graph_paths:
        data = load_pyg_graph(path)
        graphs.append(data)

    return graphs


def load_model_for_seed(seed: int) -> GNNNodeScorer:
    checkpoint_path = CHECKPOINT_DIR / f"best_gnn_node_scorer_seed_{seed}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)

    model = GNNNodeScorer(
        in_channels=int(checkpoint["input_dim"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
        out_channels=int(checkpoint["out_channels"]),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def graph_sort_key(data) -> tuple[int, int]:
    return int(data.base_cycle_size), int(data.gap_level)


def evaluate_seed(seed: int, graphs: list) -> list[dict]:
    model = load_model_for_seed(seed)
    rows = []

    with torch.no_grad():
        for data in sorted(graphs, key=graph_sort_key):
            graph = pyg_data_to_networkx_graph(data)

            predicted_scores = model(data.x, data.edge_index)
            learned_ordering = scores_to_ordering(predicted_scores)

            coloring = greedy_color_with_ordering(
                graph=graph,
                ordering=learned_ordering,
            )

            gnn_colors = int(count_colors(coloring))
            valid_coloring = bool(is_valid_coloring(graph, coloring))

            target_colors = int(data.selected_num_colors)
            colpack5_colors = int(data.best_colpack5_colors)

            rows.append(
                {
                    "seed": seed,
                    "split": "extra_test",
                    "graph_id": data.graph_id,
                    "base_cycle_size": int(data.base_cycle_size),
                    "gap_level": int(data.gap_level),
                    "gnn_colors": gnn_colors,
                    "target_colors": target_colors,
                    "colpack5_colors": colpack5_colors,
                    "gap_from_target": gnn_colors - target_colors,
                    "gap_from_colpack5": gnn_colors - colpack5_colors,
                    "colors_saved_vs_colpack5": colpack5_colors - gnn_colors,
                    "reached_target": gnn_colors == target_colors,
                    "improved_over_colpack5": gnn_colors < colpack5_colors,
                    "valid_coloring": valid_coloring,
                }
            )

    return rows


def main() -> None:
    graphs = load_extra_test_graphs(PYG_DATA_DIR)

    if not graphs:
        raise ValueError(f"No PyG graphs found in {PYG_DATA_DIR}")

    per_graph_rows = []

    for seed in SEEDS:
        seed_rows = evaluate_seed(seed=seed, graphs=graphs)
        per_graph_rows.extend(seed_rows)

        total_gnn = sum(row["gnn_colors"] for row in seed_rows)
        total_target = sum(row["target_colors"] for row in seed_rows)
        total_colpack5 = sum(row["colpack5_colors"] for row in seed_rows)
        total_saved = total_colpack5 - total_gnn

        print(
            f"Seed {seed}: "
            f"GNN={total_gnn}, target={total_target}, "
            f"ColPack-5={total_colpack5}, saved={total_saved}"
        )

    per_graph_df = pd.DataFrame(per_graph_rows)

    per_gap_df = (
        per_graph_df
        .groupby("gap_level")
        .agg(
            num_seed_runs=("seed", "count"),
            target_colors=("target_colors", "first"),
            colpack5_colors=("colpack5_colors", "first"),
            mean_gnn_colors=("gnn_colors", "mean"),
            std_gnn_colors=("gnn_colors", "std"),
            min_gnn_colors=("gnn_colors", "min"),
            max_gnn_colors=("gnn_colors", "max"),
            mean_gap_from_target=("gap_from_target", "mean"),
            mean_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "mean"),
            min_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "min"),
            max_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "max"),
            target_reached_count=("reached_target", "sum"),
            improved_over_colpack5_count=("improved_over_colpack5", "sum"),
            valid_coloring_count=("valid_coloring", "sum"),
        )
        .reset_index()
    )

    seed_total_df = (
        per_graph_df
        .groupby("seed")
        .agg(
            total_gnn_colors=("gnn_colors", "sum"),
            total_target_colors=("target_colors", "sum"),
            total_colpack5_colors=("colpack5_colors", "sum"),
            total_colors_saved_vs_colpack5=("colors_saved_vs_colpack5", "sum"),
            total_gap_from_target=("gap_from_target", "sum"),
        )
        .reset_index()
    )

    overall_df = pd.DataFrame(
        [
            {
                "num_seeds": per_graph_df["seed"].nunique(),
                "num_extra_test_graphs_per_seed": per_graph_df["graph_id"].nunique(),
                "total_extra_test_runs": len(per_graph_df),
                "mean_total_gnn_colors": seed_total_df["total_gnn_colors"].mean(),
                "std_total_gnn_colors": seed_total_df["total_gnn_colors"].std(),
                "min_total_gnn_colors": seed_total_df["total_gnn_colors"].min(),
                "max_total_gnn_colors": seed_total_df["total_gnn_colors"].max(),
                "target_total_colors": seed_total_df["total_target_colors"].iloc[0],
                "colpack5_total_colors": seed_total_df["total_colpack5_colors"].iloc[0],
                "mean_total_colors_saved_vs_colpack5": seed_total_df[
                    "total_colors_saved_vs_colpack5"
                ].mean(),
                "min_total_colors_saved_vs_colpack5": seed_total_df[
                    "total_colors_saved_vs_colpack5"
                ].min(),
                "max_total_colors_saved_vs_colpack5": seed_total_df[
                    "total_colors_saved_vs_colpack5"
                ].max(),
                "mean_total_gap_from_target": seed_total_df[
                    "total_gap_from_target"
                ].mean(),
            }
        ]
    )

    PER_GRAPH_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    per_graph_df.to_csv(PER_GRAPH_OUTPUT_CSV, index=False)
    per_gap_df.to_csv(PER_GAP_OUTPUT_CSV, index=False)
    overall_df.to_csv(OVERALL_OUTPUT_CSV, index=False)

    print()
    print("Extra C44 GNN evaluation summary")
    print("--------------------------------")
    print()
    print("Per-gap summary:")
    print(per_gap_df.to_string(index=False))
    print()
    print("Overall summary:")
    print(overall_df.to_string(index=False))
    print()
    print(f"Saved per-graph results to: {PER_GRAPH_OUTPUT_CSV}")
    print(f"Saved per-gap summary to: {PER_GAP_OUTPUT_CSV}")
    print(f"Saved overall summary to: {OVERALL_OUTPUT_CSV}")


if __name__ == "__main__":
    main()