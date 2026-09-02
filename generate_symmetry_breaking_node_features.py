"""Generate the thesis figure illustrating deterministic symmetry breaking.

The example uses the square of an eight-vertex cycle.  Every vertex has the
same label-independent structural feature vector, while the normalized
position in ascending vertex-label order distinguishes the node inputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


OUTPUT_STEM = "symmetry_breaking_node_features"


def cycle_square_adjacency(num_nodes: int) -> np.ndarray:
    """Return the adjacency matrix of the square of a cycle."""

    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
    for source in range(num_nodes):
        for target in range(source + 1, num_nodes):
            distance = min(abs(source - target), num_nodes - abs(source - target))
            if distance in (1, 2):
                adjacency[source, target] = 1
                adjacency[target, source] = 1
    return adjacency


def structural_features(adjacency: np.ndarray) -> np.ndarray:
    """Return the 15 label-independent structural features used in the thesis."""

    num_nodes = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    max_degree = max(float(degrees.max()), 1.0)

    # C_8^2 is 4-regular and is itself a 4-core.
    core_numbers = np.full(num_nodes, 4.0)
    max_core = 4.0

    # Each triangle containing v contributes two closed length-three walks at v.
    triangles = np.diag(adjacency @ adjacency @ adjacency).astype(float) / 2.0
    max_triangles = max(float(triangles.max()), 1.0)
    possible_neighbor_edges = degrees * (degrees - 1.0) / 2.0
    clustering = np.divide(
        triangles,
        possible_neighbor_edges,
        out=np.zeros_like(triangles),
        where=possible_neighbor_edges > 0,
    )

    rows: list[list[float]] = []
    for node in range(num_nodes):
        neighbors = np.flatnonzero(adjacency[node])
        neighbor_degrees = degrees[neighbors]
        neighbor_cores = core_numbers[neighbors]
        degree = float(degrees[node])

        if degree <= 1:
            local_edge_density = 0.0
        else:
            neighbor_edges = adjacency[np.ix_(neighbors, neighbors)].sum() / 2.0
            local_edge_density = neighbor_edges / (degree * (degree - 1) / 2)

        rows.append(
            [
                float(degree),
                degree / max(num_nodes - 1, 1),
                degree / max_degree,
                float(clustering[node]),
                float(core_numbers[node]),
                core_numbers[node] / max_core,
                float(triangles[node]),
                triangles[node] / max_triangles,
                float(neighbor_degrees.mean()) if neighbors.size else 0.0,
                float(neighbor_degrees.max()) if neighbors.size else 0.0,
                float(neighbor_degrees.min()) if neighbors.size else 0.0,
                float(neighbor_degrees.std()) if neighbors.size else 0.0,
                float(neighbor_cores.mean()) if neighbors.size else 0.0,
                float(neighbor_cores.max()) if neighbors.size else 0.0,
                float(local_edge_density),
            ]
        )

    return np.asarray(rows, dtype=float)


def draw_graph(
    axis: plt.Axes,
    adjacency: np.ndarray,
    positions: np.ndarray,
    node_values: np.ndarray,
    title: str,
    annotation: str,
    color_map: str,
    value_min: float,
    value_max: float,
) -> object:
    """Draw one panel using a fixed graph layout."""

    num_nodes = adjacency.shape[0]
    for source in range(num_nodes):
        for target in range(source + 1, num_nodes):
            if adjacency[source, target]:
                axis.plot(
                    positions[[source, target], 0],
                    positions[[source, target], 1],
                    color="#7a7a7a",
                    linewidth=1.0,
                    alpha=0.75,
                    zorder=1,
                )

    nodes = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=node_values,
        cmap=plt.get_cmap(color_map),
        vmin=value_min,
        vmax=value_max,
        s=560,
        edgecolors="#222222",
        linewidths=0.8,
        zorder=2,
    )
    for node, (horizontal, vertical) in enumerate(positions):
        label = axis.text(
            horizontal,
            vertical,
            str(node),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            zorder=3,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=1.4, foreground="white")]
        )
    axis.set_title(title, fontsize=10, pad=8)
    axis.text(
        0.5,
        -0.03,
        annotation,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )
    axis.set_axis_off()
    axis.set_aspect("equal")
    axis.set_xlim(-1.28, 1.28)
    axis.set_ylim(-1.20, 1.20)
    return nodes


def main() -> None:
    num_nodes = 8
    adjacency = cycle_square_adjacency(num_nodes)
    structural = structural_features(adjacency)
    structural_signature_count = np.unique(
        np.round(structural, decimals=12), axis=0
    ).shape[0]

    normalized_position = np.arange(num_nodes, dtype=float) / (num_nodes - 1)
    augmented_signature_count = np.unique(
        np.column_stack((structural, normalized_position)), axis=0
    ).shape[0]

    # Circular coordinates with vertex 0 at the top.
    angles = np.pi / 2.0 - 2.0 * np.pi * np.arange(num_nodes) / num_nodes
    positions = np.column_stack((np.cos(angles), np.sin(angles)))

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)

    draw_graph(
        axes[0],
        adjacency,
        positions,
        np.zeros(num_nodes),
        r"(a) Label-independent structural features",
        f"15 structural features: {structural_signature_count} unique vector",
        "Greys",
        -1.0,
        1.0,
    )
    node_artist = draw_graph(
        axes[1],
        adjacency,
        positions,
        normalized_position,
        r"(b) Added normalized label position $p_v$",
        f"Structural features + $p_v$: {augmented_signature_count} unique vectors",
        "viridis",
        0.0,
        1.0,
    )

    colorbar = figure.colorbar(
        node_artist,
        ax=axes[1],
        location="right",
        fraction=0.055,
        pad=0.03,
    )
    colorbar.set_label(r"Normalized position $p_v$", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)

    output_dir = Path(__file__).resolve().parent
    figure.savefig(output_dir / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    figure.savefig(
        output_dir / f"{OUTPUT_STEM}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
