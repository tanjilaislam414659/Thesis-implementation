import networkx as nx

from src.graphs.ordering_strategies import get_ordering_strategies

G = nx.Graph()
G.add_edges_from([
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (3, 4),
    (4, 5),
])

strategies = get_ordering_strategies(seed=42)

for name, strategy in strategies.items():
    order = strategy(G)
    print(f"{name}: {order}")