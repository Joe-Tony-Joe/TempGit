import networkx as nx
import matplotlib.pyplot as plt

# G = nx.Graph([(1, 2, {"color": "yellow"})])
# print(G[1]) # same as G.adj[1]
# nx.draw(G,with_labels=True)
# plt.show()

# H = nx.complete_graph(6)
# K_3_5 = nx.complete_bipartite_graph(3, 5)
# barbell = nx.barbell_graph(10, 10)
# nx.draw(barbell,with_labels=True)
# plt.show()
# print(barbell)

G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3)])
G.add_node("spam")       # adds node "spam"
ll = list(nx.connected_components(G))
# print(ll)
nx.draw(G,with_labels=True)
plt.show()
print(G.degree())
# l2 = sorted(d for n,d in G.degree())
# print(l2)
# print(nx.clustering(G))