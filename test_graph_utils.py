import networkx as nx

from src.utils.graph_utils import (
    compute_graph_stats,
    count_colors,
    greedy_coloring,
    is_valid_coloring,
)

G = nx.cycle_graph(5)
order = list(G.nodes)

coloring = greedy_coloring(G, order)
stats = compute_graph_stats(G)

print("Graph stats:", stats)
print("Coloring:", coloring)
print("Valid coloring:", is_valid_coloring(G, coloring))
print("Number of colors:", count_colors(coloring))