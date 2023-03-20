# DeepWalk Graph Embeddings
import pandas as pd
import numpy as np
import re, random, warnings
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from typing import List, Callable
from functools import partial

# ignoring warnings
warnings.filterwarnings("ignore")

# importing DBLP dataset
dblp_dataset = pd.read_csv("DBLP.csv")
# print("\n\n DBLP Dataset: \n\n", dblp_dataset)
print("\n Columns in a DBLP-Dataset: \n\n", dblp_dataset.columns, "\n")

# Getting Authors and Titles Dataset
db = dblp_dataset.loc[:10000,["Author", "Title"]] # Adjust the no. of samples user want
dataset = db.drop_duplicates()
# new_ds = db.drop_duplicates()
# new_ds.drop_duplicates().iloc[28273:28277, :]
# new_ds.set_index('Author', drop=True, inplace=True)
# new_ds.loc["øystein Tråsdahl":, :]
# new_ds
# print("\n\n Dataset with Authors and Titles: \n\n", dataset)

# Making Authors and Titles Lists
authors_list = np.array(dataset['Author'])
titles_list = np.array(dataset['Title'])
# print("\n\nNumber of Authors: ", len(authors_list), "\n") 
# print("Number of Titles: ", len(titles_list), "\n")
# print("Authors List and Titles List: ")
# print(authors_list[:20], "\n\n", titles_list[:20])

# providing unique ID's for Authors and Titles
unique_model = defaultdict(lambda: len(unique_model))
aut_keys = [unique_model[ele] for ele in authors_list] # unique key ID for each Author name
tit_keys = [unique_model[ele] for ele in titles_list] # uniqe key ID for each Title name

dataset['title_keys'] = tit_keys
dataset['author_keys'] = aut_keys

dict_titles = dict(zip(tit_keys, titles_list))
dict_authors = dict(zip(aut_keys, authors_list))

aut_tit_keys_group = dataset.groupby(["author_keys", "title_keys"])
#print("\nGrouping of Author_Keys and Title_Keys: \n\n", aut_tit_keys_group.first().loc[:,:])

dictionary_aut_tit_keys = dataset.groupby('author_keys')['title_keys'].apply(list).to_dict()
#print("\nDictionary of Author_Keys and Title_Keys: \n\n", dictionary_aut_tit_keys)

# Graphing a network 
graph_network = nx.Graph()
graph_network = nx.from_pandas_edgelist(dataset, 'author_keys', 'title_keys')
plt.rcParams["figure.figsize"] = (200,180)
nx.draw(graph_network, with_labels=True)
print(graph_network)


class Scatters: # First-Order Random Walks # RandomWalk

    def __init__(self, random_walklen: int, truncated_walklen: int):
        self.random_walklen = random_walklen # No. of Random Walks
        self.truncated_walklen = truncated_walklen # No. of nodes in truncated walk

    def scattering_move(self, node): # single truncated walk from a source node
        # node -> source node
        # walk -> truncated random walk
        walk = [node]
        for temp in range(self.random_walklen - 1):
            weight_node = [weigh for weigh in self.graph.neighbors(walk[-1])]
            if len(weight_node) > 0:
                walk = walk + random.sample(weight_node, 1)
        walk = [str(move) for move in walk]
        return walk
    
    def scattering_moves(self, graph): # certain no. of truncated walks from every node in the graph
        self.walks, self.graph = [], graph  # graph -> graph_networkx
        for weight in self.graph.nodes():
            for temp in range(self.truncated_walklen):
                movement = self.scattering_move(weight)
                self.walks.append(movement)

class Predictor(object):

    def seed_adjustment(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def provide_sustainability(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        #Ensure walk traversal conditions.
        edges = [(ids, ids) for ids in range(graph.number_of_nodes())]
        graph.add_edges_from(edges)
        return graph

    @staticmethod
    def scan_ids(graph: nx.classes.graph.Graph):
        num_ids = [ids for ids in range(graph.number_of_nodes())]
        node_ids = sorted([weigh for weigh in graph.nodes()])
        assert num_ids == node_ids, "The node indexing is wrong."

    def scan_network(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        self.scan_ids(graph)
        graph = self.provide_sustainability(graph)
        return graph

    def scan_networks(self, graphs: List[nx.classes.graph.Graph]):
        #Check the Karate Club assumptions for a list of graphs.
        graphs = [self.scan_network(graph) for graph in graphs]
        return graphs

class DeepWalk(Predictor): # DeepWalk Method

    def __init__(
        self,
        truncated_walklen: int = 10,
        random_walklen: int = 80,
        dimensions: int = 64,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        self.truncated_walklen = truncated_walklen
        self.random_walklen = random_walklen
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        
        self.seed_adjustment()
        graph = self.scan_network(graph)
        walker = Scatters(self.random_walklen, self.truncated_walklen)
        walker.scattering_moves(graph)

        wv_model = Word2Vec(
            walker.walks,
            hs=1,
            alpha=self.learning_rate,
            epochs=self.epochs,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
        )

        total_nodes = graph.number_of_nodes()
        self._embedding = [wv_model.wv[str(k)] for k in range(total_nodes)]

    def get_embedding(self) -> np.array:
        
        return np.array(self._embedding)

model_deepwalk = DeepWalk()
model_deepwalk.fit(graph_network)
embedding_deepwalk = model_deepwalk.get_embedding()
print("\nDeepwalk Model Embeddings:\n\n", embedding_deepwalk) # DeepWalk Embeddings
