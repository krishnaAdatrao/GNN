import pandas as pd
import numpy as np
import re, random, warnings, os, gensim, pkg_resources
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from typing import List, Callable
from functools import partial
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm import tqdm
# from .parallel import parallel_generate_walks
# from node2vec import Node2Vec
# from node2vec.edges import HadamardEmbedder

# loading dataset
dataset = pd.read_csv("/home/krishna/Projects/GNN/Work/Work3/n2e.csv")

# Sample Dataset with 300 rows
sample_data = dataset.iloc[:1000, :]
# print(sample_data)
node1 = list(sample_data['Node1'])
node2 = list(sample_data['Node2'])
edges = list(sample_data['EdgeWeight'])
nodes = set(node1+node2)
node_pairs = list(zip(node1, node2))
nodes_edges = list(zip(node1, node2, edges))
# print(nodes_edges)
# print(len(nodes))

# Graph Representation:
graph_network = nx.Graph()
graph_network.add_nodes_from(nodes)
graph_network.add_weighted_edges_from(nodes_edges)
graph_nodes = graph_network.nodes
graph_edges = graph_network.edges
pos = nx.spring_layout(graph_network)
# print("\n\n Graph Contains: ", nx.info(graph_network))
weight = nx.get_edge_attributes(graph_network, 'weight')
plt.figure(figsize = (50,20))
nx.draw(graph_network, pos, edge_color = 'black', width = 1, linewidths = 1, node_size = 500, with_labels = True)
nx.draw_networkx_edge_labels(graph_network, pos, edge_labels = weight, font_color = 'red')
plt.axis('off')
# plt.savefig('Graph_300_pb.png')
# plt.show()

# graph_network = nx.from_pandas_edgelist(sample_data, 'Node1', 'Node2', edge_attr = 'EdgeWeight')
# print(graph_network)
# print(type(graph_network)) 
# nx.draw(graph_network, with_labels=True, font_weight='bold')

# NODE Embeddings:
def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
                            quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks


class Node2Vec:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph: nx.Graph, dimensions: int = 128, walk_length: int = 80, num_walks: int = 10, p: float = 1,
                 q: float = 1, weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 quiet: bool = False, temp_folder: str = None, seed: int = None):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        :param seed: Seed for the random number generator.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        """

        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    try:
                        if self.graph[current_node][destination].get(self.weight_key):
                            weight = self.graph[current_node][destination].get(self.weight_key, 1)
                        else: 
                            ## Example : AtlasView({0: {'type': 1, 'weight':0.1}})- when we have edge weight
                            edge = list(self.graph[current_node][destination])[-1]
                            weight = self.graph[current_node][destination][edge].get(self.weight_key, 1)
                            
                    except:
                        weight = 1 
                    
                    if destination == source:  # Backwards probability
                        ss_weight = weight * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = weight
                    else:
                        ss_weight = weight * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

            # Calculate first_travel weights for source
            first_travel_weights = []

            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

            first_travel_weights = np.array(first_travel_weights)
            d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

            # Save neighbors
            d_graph[source][self.NEIGHBORS_KEY] = list(self.graph.neighbors(source))

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks

    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameters for gensim.models.Word2Vec - do not supply 'size' / 'vector_size' it is
            taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        # Figure out gensim version, naming of output dimensions changed from size to vector_size in v4.0.0
        gensim_version = pkg_resources.get_distribution("gensim").version
        size = 'size' if gensim_version < '4.0.0' else 'vector_size'
        if size not in skip_gram_params:
            skip_gram_params[size] = self.dimensions

        if 'sg' not in skip_gram_params:
            skip_gram_params['sg'] = 1

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)
    
# EDGE Embeddings:

from abc import ABC, abstractmethod
from functools import reduce
from itertools import combinations_with_replacement

import numpy as np
import pkg_resources
from gensim.models import KeyedVectors
from tqdm import tqdm


class EdgeEmbedder(ABC):
    INDEX_MAPPING_KEY = 'index2word' if pkg_resources.get_distribution("gensim").version < '4.0.0' else 'index_to_key'

    def __init__(self, keyed_vectors: KeyedVectors, quiet: bool = False):
        """
        :param keyed_vectors: KeyedVectors containing nodes and embeddings to calculate edges for
        """

        self.kv = keyed_vectors
        self.quiet = quiet

    @abstractmethod
    def _embed(self, edge: tuple) -> np.ndarray:
        """
        Abstract method for implementing the embedding method
        :param edge: tuple of two nodes
        :return: Edge embedding
        """
        pass

    def __getitem__(self, edge) -> np.ndarray:
        if not isinstance(edge, tuple) or not len(edge) == 2:
            raise ValueError('edge must be a tuple of two nodes')

        if edge[0] not in getattr(self.kv, self.INDEX_MAPPING_KEY):
            raise KeyError('node {} does not exist in given KeyedVectors'.format(edge[0]))

        if edge[1] not in getattr(self.kv, self.INDEX_MAPPING_KEY):
            raise KeyError('node {} does not exist in given KeyedVectors'.format(edge[1]))

        return self._embed(edge)

    def as_keyed_vectors(self) -> KeyedVectors:
        """
        Generated a KeyedVectors instance with all the possible edge embeddings
        :return: Edge embeddings
        """

        edge_generator = combinations_with_replacement(getattr(self.kv, self.INDEX_MAPPING_KEY), r=2)

        if not self.quiet:
            vocab_size = len(getattr(self.kv, self.INDEX_MAPPING_KEY))
            total_size = reduce(lambda x, y: x * y, range(1, vocab_size + 2)) / \
                         (2 * reduce(lambda x, y: x * y, range(1, vocab_size)))

            edge_generator = tqdm(edge_generator, desc='Generating edge features', total=total_size)

        # Generate features
        tokens = []
        features = []
        for edge in edge_generator:
            token = str(tuple(sorted(edge)))
            embedding = self._embed(edge)

            tokens.append(token)
            features.append(embedding)

        # Build KV instance
        edge_kv = KeyedVectors(vector_size=self.kv.vector_size)
        if pkg_resources.get_distribution("gensim").version < '4.0.0':
            edge_kv.add(
                entities=tokens,
                weights=features)
        else:
            edge_kv.add_vectors(
                keys=tokens,
                weights=features)

        return edge_kv


class AverageEmbedder(EdgeEmbedder):
    """
    Average node features
    """

    def _embed(self, edge: tuple):
        return (self.kv[edge[0]] + self.kv[edge[1]]) / 2


class HadamardEmbedder(EdgeEmbedder):
    """
    Hadamard product node features
    """

    def _embed(self, edge: tuple):
        return self.kv[edge[0]] * self.kv[edge[1]]


class WeightedL1Embedder(EdgeEmbedder):
    """
    Weighted L1 node features
    """

    def _embed(self, edge: tuple):
        return np.abs(self.kv[edge[0]] - self.kv[edge[1]])


class WeightedL2Embedder(EdgeEmbedder):
    """
    Weighted L2 node features
    """

    def _embed(self, edge: tuple):
        return (self.kv[edge[0]] - self.kv[edge[1]]) ** 2

# Train the Node2Vec model
node2vec = Node2Vec(graph_network, dimensions=64, walk_length=34, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

print("\n\n Is the graph is weighted Graph? \n\n", nx.is_weighted(graph_network))
print("\n Is the graph is directed graph? \n\n", nx.is_directed(graph_network), "\n\n")

# Node 1 and Node 2 represent the two publications.
# Edge1 or Edge Weight represent the strength of the citation relationship between the two publications.
print("\n\nTotal number of Nodes (Publication IDs): \n", len(nodes), "\n\n")
print("\n\n All the Nodes (Publication IDs): \n", nodes, "\n\n") # Will provide all the nodes in the graph
print(f'Total number of Node Pairs (Publication Pairs): \n\n {len(node_pairs)}', "\n\n")
print(f'Node Pairs (Publication Pairs): \n\n {node_pairs}', "\n\n") # Will illustrate the pairing of nodes

edge_nod1 = input("To visualise edge embeddings, \n\nPlease enter Publication ID (Node 1): ")
edge_nod2 = input("Now, please enter another Publication ID (Node 2): ")
# edge_nod1 = '4'
# edge_nod2 = '39'
print(f"\n\nEdge Embedding of Publication ID_pair ~ [{edge_nod1}, {edge_nod2}]: \n\n", edges_embs[(edge_nod1, edge_nod2)], "\n\n")

# 23 is the node number, there are only 34 nodes in the first 300 rows of the data.
# Else we can also give the node name as '23679' which is 23rd node in the data.
# node_nod = '4'
node_nod = input("To visualise node embeddings, \n\nPlease enter your preferred Publication ID (Node): ")
print(f"\n\nNode Embedding of Publication ID ~ {node_nod}: \n\n", model.wv.get_vector(node_nod), "\n\n")

# sim_nod = '39'
sim_nod = input("Enter the Publication ID (Node) to generate the simlar list of publications: ")
similar_pub_df = pd.DataFrame(model.wv.most_similar(sim_nod), columns=['Similar Publication IDs', 'Similarity Percentage (%)'])
print(f'\n\nList of similar results for publication ID - {sim_nod}: \n\n', similar_pub_df, "\n\n")

# model.wv.rank: This function finds the rank of a given node or edge in the model vocabulary based on its similarity to other nodes or edges.
# rank_nod1 = '4'
# rank_nod2 = '39'
rank_nod1 = input("To find the rank of the node pair, \n\nPlease enter Node1: ")
rank_nod2 = input("Now, please enter Node2: ")
print(f'\nThe rank of the given Node Pair ~ [{rank_nod1}, {rank_nod2}]:', model.wv.rank(rank_nod1, rank_nod2))

# model.wv.doesnt_match: This function finds the node or edge that does not belong in a given set of nodes or edges.
# doesnt_nod1 = '4'
# doesnt_nod2 = '39'
doesnt_nod1 = input("\n\nTo find non-matched of the node pair, \n\nPlease enter Node1: ")
doesnt_nod2 = input("Now, please enter Node2: ")
non_match = model.wv.doesnt_match([doesnt_nod1, doesnt_nod2])
print(f'\nThe non-matching node in the given node pair [{doesnt_nod1}, {doesnt_nod2}] is: {non_match}')

# Storing Embeddings
Node_Embeddings = [model.wv.get_vector(str(i)) for i in nodes]
Edge_Embeddings = [edges_embs[(str(j[0]), str(j[1]))] for i, j in enumerate(node_pairs)]
print("\n\nTotal number of Node Embeddings", len(Node_Embeddings))
print("\n\nTotal number of Edge Embeddings", len(Edge_Embeddings))

# K-Means Clustering
from sklearn.cluster import KMeans
# Node Embeddings ~ Clustering
num_clusters = 5 # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters)
node_labels = kmeans.fit_predict(Node_Embeddings)
# Edge Embeddings ~ Clustering
num_clusters = 5 # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters)
edge_labels = kmeans.fit_predict(Edge_Embeddings)
# print("\n\nnode_labels:", node_labels)
# print("\n\nedge_labels:", edge_labels)
cluster_node_df = pd.DataFrame({'Publication IDs': list(nodes), 'Clustering (K=5)': node_labels})
print(f'\n\nK-Means Custering results for publication IDs: \n\n', cluster_node_df, "\n\n")
cluster_edge_df = pd.DataFrame({'Publication ID_Pairs': node_pairs, 'Clustering (K=5)': edge_labels})
print(f'\n\nK-Means Custering results for publication IDs: \n\n', cluster_edge_df, "\n\n")

