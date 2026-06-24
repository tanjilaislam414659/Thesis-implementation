from pathlib import Path

import pandas as pd
import torch

PYG_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "pyg_data_week16_improved_features_best_available_of_5"
)


OUTPUT_CSV = PYG_DIR / "pyg_week16_improved_features_dataset_summary.csv"


def main() -> None:
    rows = []

    for path in sorted(PYG_DIR.glob("*.pt")):
        data = torch.load(path, weights_only=False)

        rows.append(
            {
                "graph_id": data.graph_id,
                "split": data.split,
                "num_nodes": data.num_nodes,
                "num_directed_edges": data.edge_index.shape[1],
                "num_undirected_edges": data.edge_index.shape[1] // 2,
                "num_features": data.x.shape[1],
                "num_targets": data.y.shape[0],
                "path": str(path),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["split", "graph_id"])

    summary.to_csv(OUTPUT_CSV, index=False)

    print(summary.to_string(index=False))
    print()
    print(f"Saved summary to: {OUTPUT_CSV}")
    print(f"Total graphs: {len(summary)}")


if __name__ == "__main__":
    main()