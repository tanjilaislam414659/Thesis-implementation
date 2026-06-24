from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results/tables/gnn_node_scorer")

INPUT_FILES = {
    "SMALLEST_LAST_GNN": RESULTS_DIR / "week15_expanded_learned_coloring_evaluation.csv",
    "BEST_AVAILABLE_OF_3_GNN": RESULTS_DIR / "week15_best_available_learned_coloring_evaluation.csv",
    "BEST_AVAILABLE_OF_5_GNN": RESULTS_DIR / "week15_best_available_of_5_learned_coloring_evaluation.csv",
    "BEST_AVAILABLE_OF_5_RANKING_GNN": RESULTS_DIR / "week16_best_available_of_5_ranking_learned_coloring_evaluation.csv",
    "RAW_IMPROVED_FEATURES_GNN": RESULTS_DIR / "week16_improved_features_best_available_of_5_learned_coloring_evaluation.csv",
    "NORMALIZED_FEATURES_GNN": RESULTS_DIR / "week16_normalized_features_best_available_of_5_learned_coloring_evaluation.csv",
    "EDGE_SEPARATION_GNN": RESULTS_DIR / "week16_normalized_features_edge_separation_learned_coloring_evaluation.csv",
    "LARGER_EXTENSION_GNN": RESULTS_DIR / "week16_larger_extension_learned_coloring_evaluation.csv",
    "LARGER_EXTENSION_10_SEEDS_GNN": RESULTS_DIR / "week16_larger_extension_10_seeds_learned_coloring_evaluation.csv",
}

OUTPUT_CSV = RESULTS_DIR / "week16_full_learned_strategy_comparison.csv"


def summarize_file(strategy_name: str, csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"Skipping missing file: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    if "graph_id" not in df.columns or "num_colors" not in df.columns:
        raise ValueError(f"Required columns missing in {csv_path}")

    summary = (
        df.groupby("graph_id")["num_colors"]
        .agg(["min", "max", "mean", "std"])
        .reset_index()
    )

    summary.insert(0, "strategy", strategy_name)
    summary["source_file"] = csv_path.name

    return summary


def main() -> None:
    all_summaries = []

    for strategy_name, csv_path in INPUT_FILES.items():
        summary = summarize_file(strategy_name, csv_path)
        if not summary.empty:
            all_summaries.append(summary)

    if not all_summaries:
        raise RuntimeError("No valid input files found.")

    result = pd.concat(all_summaries, ignore_index=True)
    result = result.sort_values(["graph_id", "strategy"]).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved full Week 16 learned-strategy comparison to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()