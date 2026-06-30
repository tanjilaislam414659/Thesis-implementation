from pathlib import Path
import random
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
        used_neighbor_colors = {
            colors[neighbor]
            for neighbor in adjacency[vertex]
            if neighbor in colors
        }

        color = 0
        while color in used_neighbor_colors:
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

        success = backtrack(0)
        return success, colors if success else None

    for k in range(1, num_vertices + 1):
        success, colors = can_color_with_k(k)
        if success:
            return k, colors

    return num_vertices, list(range(num_vertices))


def write_matrix_market(num_vertices, edges, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write a symmetric adjacency pattern with both (u, v) and (v, u).
    entries = []
    for u, v in sorted(edges):
        entries.append((u + 1, v + 1, 1))
        entries.append((v + 1, u + 1, 1))

    with output_path.open("w", encoding="utf-8") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write("% Diagnostic graph for Week 17 ordering-sensitivity experiment\n")
        f.write(f"{num_vertices} {num_vertices} {len(entries)}\n")
        for i, j, value in entries:
            f.write(f"{i} {j} {value}\n")


def save_summary(output_path, row):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def find_counterexample():
    rng = random.Random(42)

    probability_values = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    for num_vertices in range(8, 15):
        for probability in probability_values:
            for trial in range(5000):
                edges = []

                for u in range(num_vertices):
                    for v in range(u + 1, num_vertices):
                        if rng.random() < probability:
                            edges.append((u, v))

                if not edges:
                    continue

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

                smallest_last_colors = color_counts["SMALLEST_LAST"]

                best_other_name = min(
                    ["LARGEST_FIRST", "NATURAL"],
                    key=lambda name: color_counts[name],
                )
                best_other_colors = color_counts[best_other_name]

                if smallest_last_colors > best_other_colors:
                    chromatic_number, exact_coloring = exact_chromatic_number(adjacency)

                    # Prefer a graph where another greedy ordering reaches the exact optimum.
                    if best_other_colors == chromatic_number:
                        return {
                            "num_vertices": num_vertices,
                            "num_edges": len(edges),
                            "probability": probability,
                            "trial": trial,
                            "edges": edges,
                            "chromatic_number": chromatic_number,
                            "smallest_last_colors": smallest_last_colors,
                            "largest_first_colors": color_counts["LARGEST_FIRST"],
                            "natural_colors": color_counts["NATURAL"],
                            "best_other_ordering": best_other_name,
                            "best_other_colors": best_other_colors,
                            "smallest_last_ordering": orderings["SMALLEST_LAST"],
                            "largest_first_ordering": orderings["LARGEST_FIRST"],
                            "natural_ordering": orderings["NATURAL"],
                        }

    raise RuntimeError("No counterexample found. Increase search range or trials.")


def main():
    result = find_counterexample()

    graph_id = "week17_sl_counterexample"

    matrix_path = Path("data/raw/matrices/week17_sl_counterexample.mtx")
    summary_path = Path(
        "results/tables/initial_graph_coloring_benchmarks/"
        "week17_sl_counterexample_python_summary.csv"
    )

    write_matrix_market(
        num_vertices=result["num_vertices"],
        edges=result["edges"],
        output_path=matrix_path,
    )

    save_summary(
        summary_path,
        {
            "graph_id": graph_id,
            "num_vertices": result["num_vertices"],
            "num_edges": result["num_edges"],
            "chromatic_number": result["chromatic_number"],
            "smallest_last_colors_python": result["smallest_last_colors"],
            "largest_first_colors_python": result["largest_first_colors"],
            "natural_colors_python": result["natural_colors"],
            "best_other_ordering_python": result["best_other_ordering"],
            "best_other_colors_python": result["best_other_colors"],
            "matrix_path": str(matrix_path),
        },
    )

    print("Found diagnostic graph.")
    print(f"Graph ID: {graph_id}")
    print(f"Vertices: {result['num_vertices']}")
    print(f"Edges: {result['num_edges']}")
    print(f"Exact chromatic number: {result['chromatic_number']}")
    print(f"Python SMALLEST_LAST colors: {result['smallest_last_colors']}")
    print(f"Python LARGEST_FIRST colors: {result['largest_first_colors']}")
    print(f"Python NATURAL colors: {result['natural_colors']}")
    print(f"Best non-SL ordering: {result['best_other_ordering']}")
    print()
    print(f"Saved matrix to: {matrix_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()