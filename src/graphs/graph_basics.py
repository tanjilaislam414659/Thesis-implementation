import networkx as nx
import matplotlib.pyplot as plt

# Create a simple graph
G = nx.cycle_graph(6)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()