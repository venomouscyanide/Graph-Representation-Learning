import random

import networkx as nx
import torch

import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.optim import SGD


def explore():
    G = nx.Graph()
    G.add_node(0, feature=0, label=0)

    # Get attributes of the node 0
    node_0_attr = G.nodes[0]
    print("Node 0 has the attributes {}".format(node_0_attr))

    # Add multiple nodes with attributes
    G.add_nodes_from([
        (1, {"feature": 1, "label": 1}),
        (2, {"feature": 2, "label": 2})
    ])

    # Loop through all the nodes
    # Set data=True will return node attributes
    for node in G.nodes(data=True):
        print(node)

    for node in G.nodes(data=True):
        print(node)

    # Get number of nodes
    num_nodes = G.number_of_nodes()
    print("G has {} nodes".format(num_nodes))


def page_rank_one_iteration():
    G = nx.karate_club_graph()
    beta = 0.8
    r0 = 1 / G.number_of_nodes()
    node_id = 0

    def _page_rank_eq():
        return beta * (r0 / G.degree(node_id)) + (1 - beta) * (1 / G.number_of_nodes())

    r1 = sum([_page_rank_eq() for _ in G.neighbors(node_id)])
    return round(r1, 2)


def cc():
    G = nx.karate_club_graph()
    focus_node = 5
    closeness_centrality_denom = sum(nx.single_source_shortest_path_length(G, focus_node).values())

    def _sanity_check_cc(raw_value):
        nx_value = nx.closeness_centrality(G, u=focus_node, wf_improved=False)
        unnorm_value = (len(G.nodes())) / closeness_centrality_denom - nx_value
        assert round(unnorm_value, 2) == raw_value

    raw_value = round(1 / closeness_centrality_denom, 2)
    _sanity_check_cc(raw_value)
    return raw_value


def edges():
    G = nx.karate_club_graph()
    edge_list = list(G.edges())

    edge_list_tensor = torch.tensor(edge_list)
    return edge_list_tensor


def negative_edge_samples():
    num_samples = 50
    G = nx.karate_club_graph()
    num_nodes = len(G.nodes())
    fully_connected_set = set([(i, j) for i in range(num_nodes) for j in range(num_nodes)])

    actual_edges = set()
    for node_id in range(num_nodes):
        actual_edges |= set(G.edges(node_id))

    negative_samples = fully_connected_set.difference(actual_edges)
    n_samples = random.sample(negative_samples, k=num_samples)

    def _sanity_check_neg_samples(n_samples):
        has_edge = [G.has_edge(*neg_sample) for neg_sample in n_samples]
        assert not all(has_edge)

    _sanity_check_neg_samples(n_samples)
    return n_samples


def create_embeddings(num_node=34, embedding_dim=16):
    emb_sample = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim,
                              _weight=torch.rand(num_node, embedding_dim))
    print('Sample embedding layer: {}'.format(emb_sample))
    ids = torch.LongTensor([0, 3])

    # Print the embedding layer
    print("Embedding: {}".format(emb_sample))

    # An example that gets the embeddings for node 0 and 3
    print(emb_sample(ids))


if __name__ == '__main__':
    # explore()
    # page_rank_one_iteration()
    # print(cc())
    # print(edges())
    # negative_edge_samples()
    # create_embeddings()
    test1 = torch.rand(2, 78)
    test2 = torch.rand(2, 78)

    concat = torch.cat([test1, test2], dim=0)
    breakpoint()
