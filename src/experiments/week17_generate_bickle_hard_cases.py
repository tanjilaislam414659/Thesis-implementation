from pathlib import Path
import csv


def build_adjacency(num_vertices, edges):
    adjacency = [set() for _ in range(num_vertices)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def greedy_color_count(adjacency, ordering):
    colors = {}

    for vertex in ordering:
        used_colors = {
            colors[neighbor]
            for neighbor in adjacency[vertex]
            if neighbor in colors
        }

        color = 0
        while color in used_colors:
            color += 1

        colors[vertex] = color

    return max(colors.values()) + 1 if colors else 0, colors


def smallest_last_ordering(adjacency):
    num_vertices = len(adjacency)
    remaining = set(range(num_vertices))
    current_degree = {v: len(adjacency[v]) for v in range(num_vertices)}
    removed_order = []

    while remaining:
        vertex = min(remaining, key=lambda v: (current_degree[v], v))
        removed_order.append(vertex)
        remaining.remove(vertex)

        for neighbor in adjacency[vertex]:
            if neighbor in remaining:
                current_degree[neighbor] -= 1

    return list(reversed(removed_order))


def largest_first_ordering(adjacency):
    return sorted(range(len(adjacency)), key=lambda v: (-len(adjacency[v]), v))


def natural_ordering(adjacency):
    return list(range(len(adjacency)))


def exact_chromatic_number(adjacency):
    num_vertices = len(adjacency)
    vertices = sorted(range(num_vertices), key=lambda v: len(adjacency[v]), reverse=True)

    def can_color_with_k(k):
        colors = [-1] * num_vertices

        def backtrack(index):
            if index == num_vertices:
                return True

            vertex = vertices[index]
            forbidden = {
                colors[neighbor]
                for neighbor in adjacency[vertex]
                if colors[neighbor] != -1
            }

            for color in range(k):
                if color not in forbidden:
                    colors[vertex] = color
                    if backtrack(index + 1):
                        return True
                    colors[vertex] = -1

            return False

        return backtrack(0)

    for k in range(1, num_vertices + 1):
        if can_color_with_k(k):
            return k

    return num_vertices


def build_g10_edges():
    """
    Literature-defined G10 graph from Bickle's Smallest-Last hard-case paper.

    Vertex mapping:
        a1, a2, a3 -> 0, 1, 2
        b1, b2, b3, b4 -> 3, 4, 5, 6
        c1, c2, c3 -> 7, 8, 9

    Structure:
        A union B induces K_{3,4}
        C induces K_3
        extra edges: b1-c1, b2-c2, b3-c3, b4-c3
    """
    A = [0, 1, 2]
    B = [3, 4, 5, 6]
    C = [7, 8, 9]

    edges = set()

    # A union B induces K_{3,4}
    for a in A:
        for b in B:
            edges.add(tuple(sorted((a, b))))

    # C induces K_3
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            edges.add(tuple(sorted((C[i], C[j]))))

    # Extra edges
    extra_edges = [
        (3, 7),  # b1-c1
        (4, 8),  # b2-c2
        (5, 9),  # b3-c3
        (6, 9),  # b4-c3
    ]

    for u, v in extra_edges:
        edges.add(tuple(sorted((u, v))))

    return sorted(edges)


def build_cycle_square_edges(n):
    """
    Build C_n^2.

    Vertices are 0, 1, ..., n-1.
    Each vertex is connected to vertices at cyclic distance 1 and 2.
    """
    edges = set()

    for v in range(n):
        for distance in [1, 2]:
            u = (v + distance) % n
            edges.add(tuple(sorted((v, u))))

    return sorted(edges)


def write_matrix_market(num_vertices, edges, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for u, v in sorted(edges):
        entries.append((u + 1, v + 1, 1))
        entries.append((v + 1, u + 1, 1))

    with output_path.open("w", encoding="utf-8") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write("% Week 17 Bickle Smallest-Last hard-case graph\n")
        f.write(f"{num_vertices} {num_vertices} {len(entries)}\n")

        for i, j, value in entries:
            f.write(f"{i} {j} {value}\n")


def evaluate_graph(graph_id, num_vertices, edges, matrix_path):
    adjacency = build_adjacency(num_vertices, edges)

    orderings = {
        "SMALLEST_LAST": smallest_last_ordering(adjacency),
        "LARGEST_FIRST": largest_first_ordering(adjacency),
        "NATURAL": natural_ordering(adjacency),
    }

    color_counts = {
        name: greedy_color_count(adjacency, ordering)[0]
        for name, ordering in orderings.items()
    }

    chromatic_number = exact_chromatic_number(adjacency)

    write_matrix_market(num_vertices, edges, matrix_path)

    return {
        "graph_id": graph_id,
        "num_vertices": num_vertices,
        "num_edges": len(edges),
        "chromatic_number_python": chromatic_number,
        "smallest_last_colors_python": color_counts["SMALLEST_LAST"],
        "largest_first_colors_python": color_counts["LARGEST_FIRST"],
        "natural_colors_python": color_counts["NATURAL"],
        "smallest_last_gap_from_chromatic": color_counts["SMALLEST_LAST"] - chromatic_number,
        "matrix_path": str(matrix_path),
    }


def main():
    output_dir = Path("data/raw/matrices/week17_bickle_hard_cases")
    summary_path = Path(
        "results/tables/initial_graph_coloring_benchmarks/"
        "week17_bickle_hard_cases_python_summary.csv"
    )

    graph_specs = []

    # Single explanatory G10 graph
    graph_specs.append(
        {
            "graph_id": "week17_bickle_g10",
            "num_vertices": 10,
            "edges": build_g10_edges(),
            "matrix_path": output_dir / "week17_bickle_g10.mtx",
        }
    )

    # Cycle-square hard-case family C_{3r+2}^2
    for n in [8, 11, 14, 17, 20]:
        graph_specs.append(
            {
                "graph_id": f"week17_cycle_square_c{n}",
                "num_vertices": n,
                "edges": build_cycle_square_edges(n),
                "matrix_path": output_dir / f"week17_cycle_square_c{n}.mtx",
            }
        )

    rows = []
    for spec in graph_specs:
        row = evaluate_graph(
            graph_id=spec["graph_id"],
            num_vertices=spec["num_vertices"],
            edges=spec["edges"],
            matrix_path=spec["matrix_path"],
        )
        rows.append(row)

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Generated Bickle hard-case graphs.")
    print()
    for row in rows:
        print(
            f"{row['graph_id']}: "
            f"n={row['num_vertices']}, "
            f"edges={row['num_edges']}, "
            f"chi={row['chromatic_number_python']}, "
            f"SL={row['smallest_last_colors_python']}, "
            f"LF={row['largest_first_colors_python']}, "
            f"NATURAL={row['natural_colors_python']}, "
            f"SL-gap={row['smallest_last_gap_from_chromatic']}"
        )

    print()
    print(f"Saved matrices to: {output_dir}")
    print(f"Saved Python summary to: {summary_path}")


if __name__ == "__main__":
    main()