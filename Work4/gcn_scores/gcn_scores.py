import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Load the cora dataset
dataset = Planetoid(root='./data/cora', name='cora')
data = dataset[0]

# Convert the data to NetworkX graph
G = nx.Graph()
G.add_nodes_from(range(data.x.shape[0]))
G.add_edges_from(zip(data.edge_index[0], data.edge_index[1]))

# Define the GCN model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize the model and optimizer
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Train the model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluate the model
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)
print("F1 score:", f1_score(data.y.cpu().numpy(), pred.cpu().numpy(), average="micro"))
print("Precision score:", precision_score(data.y.cpu().numpy(), pred.cpu().numpy(), average="micro"))
print("Recall score:", recall_score(data.y.cpu().numpy(), pred.cpu().numpy(), average="micro"))
print("Accuracy score:", accuracy_score(data.y.cpu().numpy(), pred.cpu().numpy()))

