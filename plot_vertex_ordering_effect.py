from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch


# Graph adapted from the cited Baeldung example.
EDGES = [
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 3),
    (2, 4),
    (2, 7),
    (3, 5),
    (3, 7),
    (4, 6),
    (5, 6),
    (6, 7),
]

# Fixed positions ensure the same layout across repeated runs.
POSITIONS = {
    1: (-0.9, 1.0),
    3: (0.3, 1.0),
    4: (-1.7, -0.45),
    6: (-0.3, 0.15),
    5: (0.8, -0.45),
    2: (-0.3, -1.15),
    7: (1.55, 0.0),
}

NATURAL_ORDER = (1, 2, 3, 4, 5, 6, 7)
ALTERNATIVE_ORDER = (6, 1, 4, 5, 2, 3, 7)

COLOR_PALETTE = {
    1: "#0072B2",  # blue
    2: "#009E73",  # green
    3: "#F0E442",  # yellow
    4: "#D62728",  # red
}


def greedy_coloring(
    graph: nx.Graph,
    ordering: tuple[int, ...],
) -> dict[int, int]:
    """Apply sequential greedy coloring using positive color indices."""

    if len(ordering) != graph.number_of_nodes():
        raise ValueError("The ordering has an incorrect length.")

    if set(ordering) != set(graph.nodes):
        raise ValueError(
            "The ordering must contain every vertex exactly once."
        )

    coloring: dict[int, int] = {}

    for vertex in ordering:
        used_colors = {
            coloring[neighbor]
            for neighbor in graph.neighbors(vertex)
            if neighbor in coloring
        }

        color = 1
        while color in used_colors:
            color += 1

        coloring[vertex] = color

    return coloring


def verify_coloring(
    graph: nx.Graph,
    coloring: dict[int, int],
) -> None:
    """Check vertex coverage and the absence of edge conflicts."""

    if set(coloring) != set(graph.nodes):
        raise AssertionError("The coloring does not cover every vertex.")

    for u, v in graph.edges:
        if coloring[u] == coloring[v]:
            raise AssertionError(
                f"Vertices {u} and {v} have the same color."
            )


def draw_panel(
    axis: plt.Axes,
    graph: nx.Graph,
    coloring: dict[int, int],
    panel_title: str,
    ordering_symbol: str,
    ordering: tuple[int, ...],
) -> None:
    """Draw one ordering and its greedy coloring."""

    node_colors = [
        COLOR_PALETTE[coloring[vertex]]
        for vertex in graph.nodes
    ]

    nx.draw_networkx_edges(
        graph,
        POSITIONS,
        ax=axis,
        width=1.5,
        edge_color="#4D4D4D",
    )

    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        ax=axis,
        node_color=node_colors,
        node_size=900,
        edgecolors="#222222",
        linewidths=1.2,
    )

    # Black labels are used on yellow nodes for readability.
    yellow_nodes = [
        vertex
        for vertex in graph.nodes
        if coloring[vertex] == 3
    ]

    other_nodes = [
        vertex
        for vertex in graph.nodes
        if coloring[vertex] != 3
    ]

    nx.draw_networkx_labels(
        graph,
        POSITIONS,
        labels={
            vertex: str(vertex)
            for vertex in yellow_nodes
        },
        ax=axis,
        font_color="black",
        font_size=11,
        font_weight="bold",
    )

    nx.draw_networkx_labels(
        graph,
        POSITIONS,
        labels={
            vertex: str(vertex)
            for vertex in other_nodes
        },
        ax=axis,
        font_color="white",
        font_size=11,
        font_weight="bold",
    )

    ordering_text = ",".join(
        str(vertex)
        for vertex in ordering
    )
    color_count = max(coloring.values())

    axis.set_title(
        f"{panel_title} "
        f"${ordering_symbol}=({ordering_text})$\n"
        f"Greedy color count: {color_count}",
        fontsize=11,
        pad=9,
    )

    axis.set_aspect("equal")
    axis.axis("off")


def main() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(1, 8))
    graph.add_edges_from(EDGES)

    natural_coloring = greedy_coloring(
        graph,
        NATURAL_ORDER,
    )
    alternative_coloring = greedy_coloring(
        graph,
        ALTERNATIVE_ORDER,
    )

    verify_coloring(graph, natural_coloring)
    verify_coloring(graph, alternative_coloring)

    # Verify the claims stated in the Supporting Materials.
    assert max(natural_coloring.values()) == 4
    assert max(alternative_coloring.values()) == 3

    # Vertices 1, 4, and 6 form a triangle.
    triangle_edges = [
        (1, 4),
        (1, 6),
        (4, 6),
    ]
    assert all(
        graph.has_edge(u, v)
        for u, v in triangle_edges
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
    })

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.4),
    )

    draw_panel(
        axes[0],
        graph,
        natural_coloring,
        "(a) Natural ordering",
        r"\pi_{\mathrm{N}}",
        NATURAL_ORDER,
    )

    draw_panel(
        axes[1],
        graph,
        alternative_coloring,
        "(b) Alternative ordering",
        r"\pi_{\mathrm{A}}",
        ALTERNATIVE_ORDER,
    )

    legend_handles = [
        Patch(
            facecolor=COLOR_PALETTE[color],
            edgecolor="#222222",
            label=f"Color {color}",
        )
        for color in sorted(COLOR_PALETTE)
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.88,
        bottom=0.17,
        wspace=0.08,
    )

    output_directory = Path(__file__).resolve().parent
    pdf_path = output_directory / "vertex_ordering_effect.pdf"
    png_path = output_directory / "vertex_ordering_effect.png"

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(
        f"Natural ordering: {natural_coloring} "
        f"({max(natural_coloring.values())} colors)"
    )
    print(
        f"Alternative ordering: {alternative_coloring} "
        f"({max(alternative_coloring.values())} colors)"
    )
    print("Both colorings are valid.")
    print(
        "Vertices 1, 4, and 6 form a triangle; "
        "therefore, the three-color result is optimal."
    )
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()