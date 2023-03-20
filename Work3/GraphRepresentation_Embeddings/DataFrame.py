import lxml
import pickle
import pandas as pd

data = pickle.load(open('edgelist.pkl', 'rb'))

# data_length_edges = len(data['edges'])
# data_length_ent2idx = len(data['ent2idx'])

# data_most_common = data['edges'].most_common(20)
# print(data_most_common)

# print("\n", f'Length of Edges: {data_length_edges}', "\n\n", f'Length of Entity to Index: {data_length_ent2idx}')

# print(type(data_edges))
# print(type(data_ent2idx))

# data_ent2idx = data['ent2idx']
data_edges = data['edges']

data_keys = data['edges'].keys()
data_values = data['edges'].values()
list_keys = list(data_keys)
data_edges_values_list = list(data_values)

node1 = []
node2 = []
edge = data_edges_values_list
len_lst_keys = len(list_keys)
for i, j in enumerate(list_keys):
    node1.append(list_keys[i][0])
    node2.append(list_keys[i][1])
    
nodes_edges_df = pd.DataFrame(list(zip(node1, node2, edge)), columns=['Node1','Node2', 'EdgeWeight'])

dataset = nodes_edges_df.to_csv("n2e.csv", index = False)

print(dataset)

