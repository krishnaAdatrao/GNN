import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

dataset = pd.read_csv("n2e.csv")
# # print(dataset)
# node1 = list(dataset['Node1'])
# node2 = list(dataset['Node2'])
# edges = list(dataset['EdgeWeight'])
# nodes = set(node1+node2)
# nodes_edges = list(zip(node1, node2, edges))
# # print(nodes_edges)
# # print(len(nodes))

sample_data = dataset.iloc[:1000, :]
# print(sample_data)
node1 = list(sample_data['Node1'])
node2 = list(sample_data['Node2'])
edges = list(sample_data['EdgeWeight'])
nodes = set(node1+node2)
nodes_edges = list(zip(node1, node2, edges))
# print(nodes_edges)
# print(len(nodes))

graph_network = nx.Graph()
graph_network.add_nodes_from(nodes)
graph_network.add_weighted_edges_from(nodes_edges)
graph_nodes = graph_network.nodes
graph_edges = graph_network.edges
pos = nx.spring_layout(graph_network)
print(nx.info(graph_network))
weight = nx.get_edge_attributes(graph_network, 'weight')
plt.figure(figsize = (50,20))
nx.draw(graph_network, pos, edge_color = 'black', width = 1, linewidths = 1, node_size = 500, with_labels = True)
nx.draw_networkx_edge_labels(graph_network, pos, edge_labels = weight, font_color = 'red')
plt.axis('off')
# plt.tight_layout()
plt.savefig('fig1.png')
plt.show()
print("Is the graph is weighted Graph? \n\n", nx.is_weighted(graph_network))
print("\n Is the graph is directed graph? \n\n", nx.is_directed(graph_network))

# graph_network = nx.from_pandas_edgelist(sample_data, 'Node1', 'Node2', edge_attr = 'EdgeWeight')
# print(graph_network)
# print(type(graph_network)) 
# nx.draw(graph_network, with_labels=True, font_weight='bold')