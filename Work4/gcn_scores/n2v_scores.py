from stellargraph.datasets import Cora
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph import StellarGraph
from stellargraph import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, optimizers, losses, layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cora_dataset = datasets.Cora()
print(cora_dataset.description)


G, _ = cora_dataset.load(largest_connected_component_only=True)

node_ids = G.nodes()
labels = {node: label for node, label in G.nodes(data='label')}

# Split the dataset
train_nodes, test_nodes = train_test_split(
    node_ids, train_size=0.1, test_size=None, stratify=labels
)
val_nodes, test_nodes = train_test_split(
    test_nodes, train_size=0.5, test_size=None, stratify={node: labels[node] for node in test_nodes}
)

# Initialize the node2vec model
rw = BiasedRandomWalk(G)

node2vec = Node2Vec(
    G,
    dimensions=128,
    walk_length=80,
    num_walks=10,
    p=1,
    q=1,
    weight_key="weight",
    workers=4,
)

# Compute the node embeddings
n2v_model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get the train and test node embeddings
train_embeddings = n2v_model.wv[train_nodes]
test_embeddings = n2v_model.wv[test_nodes]

# Train logistic regression classifier
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(train_embeddings, [labels[node] for node in train_nodes])

# Predict the labels of the test nodes
predicted_labels = clf.predict(test_embeddings)

# Compute the evaluation metrics
accuracy = accuracy_score([labels[node] for node in test_nodes], predicted_labels)
precision = precision_score([labels[node] for node in test_nodes], predicted_labels, average='weighted')
recall = recall_score([labels[node] for node in test_nodes], predicted_labels, average='weighted')
f1 = f1_score([labels[node] for node in test_nodes], predicted_labels, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1:.4f}")
