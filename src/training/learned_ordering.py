"""
Utilities for converting GNN node scores into vertex orderings.

The learned ordering follows the convention:

    higher predicted score -> earlier vertex in the ordering
"""

from __future__ import annotations

import torch


def scores_to_ordering(scores: torch.Tensor) -> list[int]:
    """
    Convert node scores into a vertex ordering.

    Parameters
    ----------
    scores:
        Tensor of shape [num_nodes] or [num_nodes, 1].

    Returns
    -------
    list[int]
        Node IDs sorted by descending predicted score.
    """

    if scores.ndim == 2:
        if scores.shape[1] != 1:
            raise ValueError(
                f"Expected scores with shape [num_nodes, 1], got {tuple(scores.shape)}."
            )
        scores = scores.view(-1)

    if scores.ndim != 1:
        raise ValueError(
            f"Expected scores with shape [num_nodes] or [num_nodes, 1], got {tuple(scores.shape)}."
        )

    ordering = torch.argsort(scores, descending=True).tolist()

    return ordering


def main() -> None:
    example_scores = torch.tensor([[0.2], [0.9], [0.5], [0.1]])

    ordering = scores_to_ordering(example_scores)

    print("Learned ordering utility check")
    print("------------------------------")
    print(f"Example scores: {example_scores.view(-1).tolist()}")
    print(f"Ordering: {ordering}")
    print("Expected ordering: [1, 2, 0, 3]")

    if ordering != [1, 2, 0, 3]:
        raise ValueError("scores_to_ordering returned an unexpected ordering.")

    print("Status: OK")


if __name__ == "__main__":
    main()