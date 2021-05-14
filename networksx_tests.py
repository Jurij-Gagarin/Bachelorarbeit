import networkx as nx
import matplotlib.pyplot as plt


G = nx.hexagonal_lattice_graph(30, 11)
plt.figure(figsize=(20, 20))
#pos = {(x, y): (y, -x) for x, y in G.nodes()}
nx.draw(G, with_labels=True, node_size=100)
plt.show()