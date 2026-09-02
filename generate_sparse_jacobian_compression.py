"""Generate a reproducible schematic of sparse Jacobian compression.

The example uses a 6-by-5 structural Jacobian pattern.  Its
column-intersection graph is three-colourable with classes
{c1, c3}, {c2, c4}, and {c5}.  The corresponding seed matrix groups
structurally orthogonal columns, so the pattern of J @ S has three columns.

Run from any directory:

    python generate_sparse_jacobian_compression.py

The script writes a vector PDF for LaTeX and a PNG for visual inspection.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


# Colour-blind-friendly Okabe--Ito colours.
CLASS_COLORS = ("#0072B2", "#E69F00", "#009E73")
GRID_COLOR = "#B8B8B8"
TEXT_COLOR = "#222222"


def configure_matplotlib() -> None:
    """Use stable, publication-oriented Matplotlib settings."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def draw_structural_matrix(
    ax: plt.Axes,
    pattern: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    nonzero_colors: list[str],
    title: str,
    panel_label: str,
    panel_label_x: float = -0.12,
) -> None:
    """Draw a binary structural pattern as a labelled cell grid."""

    n_rows, n_cols = pattern.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")

    for x in np.arange(-0.5, n_cols, 1.0):
        ax.plot([x, x], [-0.5, n_rows - 0.5], color=GRID_COLOR, lw=0.55, zorder=0)
    for y in np.arange(-0.5, n_rows, 1.0):
        ax.plot([-0.5, n_cols - 0.5], [y, y], color=GRID_COLOR, lw=0.55, zorder=0)

    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            n_cols,
            n_rows,
            fill=False,
            edgecolor="#555555",
            linewidth=0.8,
            zorder=1,
        )
    )

    for row, col in np.argwhere(pattern != 0):
        ax.scatter(
            col,
            row,
            marker="s",
            s=78,
            color=nonzero_colors[col],
            edgecolor="#333333",
            linewidth=0.35,
            zorder=3,
        )

    ax.set_xticks(range(n_cols), column_labels)
    ax.set_yticks(range(n_rows), row_labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="both", length=0, pad=3)
    for tick_label, color in zip(ax.get_xticklabels(), nonzero_colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, pad=20)
    ax.text(
        panel_label_x,
        1.13,
        panel_label,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def draw_seed_matrix(ax: plt.Axes, seed: np.ndarray) -> None:
    """Draw the seed matrix with colour-coded compressed directions."""

    n_rows, n_cols = seed.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")

    for x in np.arange(-0.5, n_cols, 1.0):
        ax.plot([x, x], [-0.5, n_rows - 0.5], color=GRID_COLOR, lw=0.55, zorder=0)
    for y in np.arange(-0.5, n_rows, 1.0):
        ax.plot([-0.5, n_cols - 0.5], [y, y], color=GRID_COLOR, lw=0.55, zorder=0)
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            n_cols,
            n_rows,
            fill=False,
            edgecolor="#555555",
            linewidth=0.8,
            zorder=1,
        )
    )

    for row in range(n_rows):
        for col in range(n_cols):
            value = int(seed[row, col])
            if value:
                ax.add_patch(
                    Rectangle(
                        (col - 0.36, row - 0.36),
                        0.72,
                        0.72,
                        facecolor=CLASS_COLORS[col],
                        edgecolor="#333333",
                        linewidth=0.35,
                        zorder=2,
                    )
                )
                text_color = "white" if col in (0, 2) else TEXT_COLOR
            else:
                text_color = "#777777"
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
                zorder=3,
            )

    ax.set_xticks(range(n_cols), [r"$d_1$", r"$d_2$", r"$d_3$"])
    ax.set_yticks(range(n_rows), [r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$c_5$"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="both", length=0, pad=3)
    for tick_label, color in zip(ax.get_xticklabels(), CLASS_COLORS):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    for tick_label, color in zip(
        ax.get_yticklabels(),
        (CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]),
    ):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(r"Seed matrix $S$", pad=20)
    ax.text(
        -0.22,
        1.13,
        "(c)",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def draw_column_intersection_graph(ax: plt.Axes, edges: list[tuple[int, int]]) -> None:
    """Draw the column-intersection graph with a fixed deterministic layout."""

    positions = {
        0: (-0.82, 0.55),
        1: (0.00, 0.95),
        2: (0.82, 0.55),
        3: (0.58, -0.48),
        4: (-0.58, -0.48),
    }
    node_colors = [
        CLASS_COLORS[0],
        CLASS_COLORS[1],
        CLASS_COLORS[0],
        CLASS_COLORS[1],
        CLASS_COLORS[2],
    ]

    for left, right in edges:
        x_values = (positions[left][0], positions[right][0])
        y_values = (positions[left][1], positions[right][1])
        ax.plot(x_values, y_values, color="#555555", linewidth=1.2, zorder=1)

    for node, (x_position, y_position) in positions.items():
        ax.scatter(
            x_position,
            y_position,
            s=660,
            color=node_colors[node],
            edgecolor="#333333",
            linewidth=0.8,
            zorder=2,
        )
        label_color = TEXT_COLOR if node_colors[node] == CLASS_COLORS[1] else "white"
        ax.text(
            x_position,
            y_position,
            rf"$c_{node + 1}$",
            ha="center",
            va="center",
            fontsize=10,
            color=label_color,
            zorder=3,
        )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.82, 1.23)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"Column-intersection graph $G$", pad=5)
    ax.text(
        -0.04,
        1.04,
        "(b)",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_COLORS[index],
            markeredgecolor="#333333",
            markersize=7,
            label=label,
        )
        for index, label in enumerate(
            (
                r"$d_1=\{c_1,c_3\}$",
                r"$d_2=\{c_2,c_4\}$",
                r"$d_3=\{c_5\}$",
            )
        )
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.3,
        fontsize=8,
    )


def main() -> None:
    configure_matplotlib()

    # Each row contains the columns that structurally intersect in that row.
    jacobian_pattern = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=int,
    )

    seed_matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )

    compressed_pattern = (jacobian_pattern @ seed_matrix > 0).astype(int)

    # Derive the column-intersection edges directly from the Jacobian pattern.
    edge_set: set[tuple[int, int]] = set()
    for row in jacobian_pattern:
        active_columns = np.flatnonzero(row)
        for left_index, left in enumerate(active_columns):
            for right in active_columns[left_index + 1 :]:
                edge_set.add((int(left), int(right)))
    edges = sorted(edge_set)

    # Verify the mathematical relationships used by the diagram.
    class_ids = np.argmax(seed_matrix, axis=1)
    assert np.all(seed_matrix.sum(axis=1) == 1)
    assert all(class_ids[left] != class_ids[right] for left, right in edges)
    assert compressed_pattern.shape == (6, 3)
    assert np.all(compressed_pattern.sum(axis=1) == 2)

    figure = plt.figure(figsize=(7.1, 5.55), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.975,
        bottom=0.09,
        top=0.94,
        width_ratios=(1.0, 1.12),
        height_ratios=(1.0, 1.0),
        wspace=0.34,
        hspace=0.52,
    )

    ax_j = figure.add_subplot(grid[0, 0])
    ax_g = figure.add_subplot(grid[0, 1])
    ax_s = figure.add_subplot(grid[1, 0])
    ax_js = figure.add_subplot(grid[1, 1])

    column_colors = [
        CLASS_COLORS[0],
        CLASS_COLORS[1],
        CLASS_COLORS[0],
        CLASS_COLORS[1],
        CLASS_COLORS[2],
    ]
    draw_structural_matrix(
        ax_j,
        jacobian_pattern,
        [rf"$r_{index}$" for index in range(1, 7)],
        [rf"$c_{index}$" for index in range(1, 6)],
        column_colors,
        r"Sparse Jacobian pattern $J$",
        "(a)",
    )
    draw_column_intersection_graph(ax_g, edges)
    draw_seed_matrix(ax_s, seed_matrix)
    draw_structural_matrix(
        ax_js,
        compressed_pattern,
        [rf"$r_{index}$" for index in range(1, 7)],
        [r"$d_1$", r"$d_2$", r"$d_3$"],
        list(CLASS_COLORS),
        r"Compressed product $JS$",
        "(d)",
        panel_label_x=-0.36,
    )

    figure.text(
        0.5,
        0.018,
        "Columns assigned to the same direction never share a structural nonzero row.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=TEXT_COLOR,
    )

    output_directory = Path(__file__).resolve().parent
    pdf_path = output_directory / "sparse_jacobian_compression.pdf"
    png_path = output_directory / "sparse_jacobian_compression.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
