"""
Summarize repeated GNN node scorer training runs.

This script reads the multiple-run training summary CSV and computes
simple aggregate statistics over the runs.
"""

from __future__ import annotations

import csv
from pathlib import Path


INPUT_CSV = Path(
    "results/tables/gnn_node_scorer/multiple_run_training_summary.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/multiple_run_training_aggregate_summary.csv"
)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    rows = []

    with INPUT_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("No rows found in multiple-run training summary.")

    validation_losses = [
        float(row["best_validation_loss"])
        for row in rows
    ]

    test_losses = [
        float(row["final_test_loss_best_model"])
        for row in rows
    ]

    train_losses = [
        float(row["final_train_loss_best_model"])
        for row in rows
    ]

    best_test_index = min(
        range(len(rows)),
        key=lambda index: test_losses[index],
    )

    best_validation_index = min(
        range(len(rows)),
        key=lambda index: validation_losses[index],
    )

    summary_row = {
        "num_runs": len(rows),
        "mean_train_loss": sum(train_losses) / len(train_losses),
        "mean_validation_loss": sum(validation_losses) / len(validation_losses),
        "mean_test_loss": sum(test_losses) / len(test_losses),
        "min_test_loss": min(test_losses),
        "max_test_loss": max(test_losses),
        "best_test_seed": rows[best_test_index]["seed"],
        "best_validation_seed": rows[best_validation_index]["seed"],
        "best_validation_loss": min(validation_losses),
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(summary_row.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

    print("Multiple-run training aggregate summary")
    print("---------------------------------------")
    print(f"Number of runs: {summary_row['num_runs']}")
    print(f"Mean train loss: {summary_row['mean_train_loss']:.6f}")
    print(f"Mean validation loss: {summary_row['mean_validation_loss']:.6f}")
    print(f"Mean test loss: {summary_row['mean_test_loss']:.6f}")
    print(f"Min test loss: {summary_row['min_test_loss']:.6f}")
    print(f"Max test loss: {summary_row['max_test_loss']:.6f}")
    print(f"Best test seed: {summary_row['best_test_seed']}")
    print(f"Best validation seed: {summary_row['best_validation_seed']}")
    print()
    print(f"Saved aggregate summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()