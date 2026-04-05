import networkx as nx


def make_cycle_graph(n: int) -> nx.Graph:
    return nx.cycle_graph(n)


def make_wheel_graph(n: int) -> nx.Graph:
    return nx.wheel_graph(n)


def make_ladder_graph(n: int) -> nx.Graph:
    return nx.ladder_graph(n)


def make_balanced_tree(r: int, h: int) -> nx.Graph:
    return nx.balanced_tree(r, h)


def make_barbell_graph(m1: int, m2: int) -> nx.Graph:
    return nx.barbell_graph(m1, m2)


def make_grid_graph(rows: int, cols: int) -> nx.Graph:
    g = nx.grid_2d_graph(rows, cols)
    return nx.convert_node_labels_to_integers(g)


def graph_summary(g: nx.Graph) -> dict:
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "density": nx.density(g),
    }


if __name__ == "__main__":
    demo_graph = make_wheel_graph(8)
    print("Demo graph summary:", graph_summary(demo_graph))

