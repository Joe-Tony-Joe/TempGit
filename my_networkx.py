import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph() #Make the graph
G.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11]) #Add nodes, although redundant because of the line below
G.add_edges_from([(1,2),(1,3),(2,3),(1,4),(2,4),(1,5),(5,6),(5,7),(6,7)]) # Adding the edges
# G.add_edges_from([(10,11),(7,10),(8,9),(3,10),(1,9)])
connected_subgraphs = [G.subgraph(cc) for cc in nx.connected_components(G)]

# ll=list(nx.clique.find_cliques(G))
# print(connected_subgraphs.pop())
# sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)




Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
print([Gcc])
G0 = G.subgraph(Gcc[0])
nx.draw(G0,with_labels=True)
plt.show()
nx.draw(G,with_labels=True)
plt.show()
# H = nx.connected_components(G)

# G = nx.path_graph(4)
# nx.add_path(G, [10, 11, 12])
# print( sorted(nx.connected_components(G), key=len, reverse=True))

# G = nx.path_graph(4)
#
# G.add_path([10, 11, 12])
# list1 = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
# print(list1)
# Gc =list( max(nx.connected_component_subgraphs(G), key=len))
# print(Gc)

# nx.draw(G,with_labels=True)
# plt.show()
# nx.draw(G.subgraph([1,2,3,4,7]),with_labels=True)
# plt.show()
# list1 = list(nx.clique.find_cliques(G))
# print(list1)
