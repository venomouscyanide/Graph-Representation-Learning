import os

import torch
import torch_geometric

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import copy

from pylab import show

G = nx.karate_club_graph()
community_map = {}
for node in G.nodes(data=True):
    if node[1]["club"] == "Mr. Hi":
        community_map[node[0]] = 0
    else:
        community_map[node[0]] = 1
node_color = []
color_map = {0: 0, 1: 1}
node_color = [color_map[community_map[node]] for node in G.nodes()]
pos = nx.spring_layout(G)
plt.figure(figsize=(7, 7))
nx.draw(G, pos=pos, cmap=plt.get_cmap('coolwarm'), node_color=node_color)


# show()


def assign_node_types(G, community_map):
    # TODO: Implement a function that takes in a NetworkX graph
    # G and community map assignment (mapping node id --> 0/1 label)
    # and adds 'node_type' as a node_attribute in G.

    ############# Your code here ############
    ## (~2 line of code)
    ## Note
    ## 1. Look up NetworkX `nx.classes.function.set_node_attributes`
    ## 2. Look above for the two node type values!
    community_map = {k: "n0" if not v else "n1" for k, v in community_map.items()}
    nx.classes.function.set_node_attributes(G, community_map, 'node_type')
    #########################################


def assign_node_labels(G, community_map):
    # TODO: Implement a function that takes in a NetworkX graph
    # G and community map assignment (mapping node id --> 0/1 label)
    # and adds 'node_label' as a node_attribute in G.

    ############# Your code here ############
    ## (~2 line of code)
    ## Note
    ## 1. Look up NetworkX `nx.classes.function.set_node_attributes`
    nx.classes.function.set_node_attributes(G, community_map, 'node_label')
    #########################################


def assign_node_features(G):
    # TODO: Implement a function that takes in a NetworkX graph
    # G and adds 'node_feature' as a node_attribute in G. Each node
    # in the graph has the same feature vector - a torchtensor with
    # data [1., 1., 1., 1., 1.]

    ############# Your code here ############
    ## (~2 line of code)
    ## Note
    ## 1. Look up NetworkX `nx.classes.function.set_node_attributes`
    nx.classes.function.set_node_attributes(G, torch.ones(5), 'node_feature')
    #########################################


assign_node_types(G, community_map)
assign_node_labels(G, community_map)
assign_node_features(G)

# Explore node properties for the node with id: 20
node_id = 20


# print(f"Node {node_id} has properties:", G.nodes(data=True)[node_id])


def assign_edge_types(G, community_map):
    # TODO: Implement a function that takes in a NetworkX graph
    # G and community map assignment (mapping node id --> 0/1 label)
    # and adds 'edge_type' as a edge_attribute in G.

    ############# Your code here ############
    ## (~5 line of code)
    ## Note
    ## 1. Create an edge assignment dict following rules above
    ## 2. Look up NetworkX `nx.classes.function.set_edge_attributes`
    edges = G.edges

    def _get_label(edge):
        label = "e0"
        if all([community_map[edge[0]], community_map[edge[1]]]):
            return "e1"
        if any([community_map[edge[0]], community_map[edge[1]]]):
            return "e2"
        return label

    edge_label_map = {edge: _get_label(edge) for edge in edges}
    nx.classes.set_edge_attributes(G, edge_label_map, "edge_type")
    #########################################


if 'IS_GRADESCOPE_ENV' not in os.environ:
    assign_edge_types(G, community_map)

    # Explore edge properties for a sampled edge and check the corresponding
    # node types
    edge_idx = 15
    n1 = 0
    n2 = 31
    edge = list(G.edges(data=True))[edge_idx]
    print(f"Edge ({edge[0]}, {edge[1]}) has properties:", edge[2])
    print(f"Node {n1} has properties:", G.nodes(data=True)[n1])
    print(f"Node {n2} has properties:", G.nodes(data=True)[n2])

edge_color = {}
for edge in G.edges():
    n1, n2 = edge
    edge_color[edge] = community_map[n1] if community_map[n1] == community_map[n2] else 2
    if community_map[n1] == community_map[n2] and community_map[n1] == 0:
        edge_color[edge] = 'blue'
    elif community_map[n1] == community_map[n2] and community_map[n1] == 1:
        edge_color[edge] = 'red'
    else:
        edge_color[edge] = 'green'

G_orig = copy.deepcopy(G)
nx.classes.function.set_edge_attributes(G, edge_color, name='color')
colors = nx.get_edge_attributes(G, 'color').values()
labels = nx.get_node_attributes(G, 'node_type')
plt.figure(figsize=(8, 8))
nx.draw(G, pos=pos, cmap=plt.get_cmap('coolwarm'), node_color=node_color, edge_color=colors, labels=labels,
        font_color='white')
# show()

from deepsnap.hetero_graph import HeteroGraph


def get_nodes_per_type(hete):
    # TODO: Implement a function that takes a DeepSNAP dataset object
    # and return the number of nodes per `node_type`.

    num_nodes_n0 = 0
    num_nodes_n1 = 0

    ############# Your code here ############
    ## (~2 line of code)
    ## Note
    ## 1. Colab autocomplete functionality might be useful.

    num_nodes_n0 = len(hete.node_type['n0'])
    num_nodes_n1 = len(hete.node_type['n1'])
    #########################################

    return num_nodes_n0, num_nodes_n1


hete = HeteroGraph(G_orig)
num_nodes_n0, num_nodes_n1 = get_nodes_per_type(hete)
print("Node type n0 has {} nodes".format(num_nodes_n0))
print("Node type n1 has {} nodes".format(num_nodes_n1))


def get_num_message_edges(hete):
    # TODO: Implement this function that takes a DeepSNAP dataset object
    # and return the number of edges for each message type.
    # You should return a list of tuples as
    # (message_type, num_edge)

    message_type_edges = []

    ############# Your code here ############
    ## (~2 line of code)
    ## Note
    ## 1. Colab autocomplete functionality might be useful.
    message_type_edges = [(msg_type, len(hete.edge_type[msg_type])) for msg_type in hete.message_types]
    #########################################

    return message_type_edges


message_type_edges = get_num_message_edges(hete)
for (message_type, num_edges) in message_type_edges:
    print("Message type {} has {} edges".format(message_type, num_edges))

from deepsnap.dataset import GraphDataset


def compute_dataset_split_counts(datasets):
    # TODO: Implement a function that takes a dict of datasets in the form
    # {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
    # and returns a dict mapping dataset names to the number of labeled
    # nodes used for supervision in that respective dataset.

    data_set_splits = {}

    ############# Your code here ############
    ## (~3 line of code)
    ## Note
    ## 1. The DeepSNAP `node_label_index` dictionary will be helpful.
    ## 2. Remember to count both node_types
    ## 3. Remember each dataset only has one graph that we need to access
    ##    (i.e. dataset[0])
    data_set_splits = {
        k: sum([len(item) for item in v.graphs[0].node_label_index.values()]) for k, v in datasets.items()
    }
    #########################################

    return data_set_splits


dataset = GraphDataset([hete], task='node')
# Splitting the dataset
dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])
datasets = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}

data_set_splits = compute_dataset_split_counts(datasets)
for dataset_name, num_nodes in data_set_splits.items():
    print("{} dataset has {} nodes".format(dataset_name, num_nodes))

import copy
import torch
import deepsnap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from sklearn.metrics import f1_score
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul


class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        # To simplify implementation, please initialize both self.lin_dst
        # and self.lin_src out_features to out_channels
        self.lin_dst = None
        self.lin_src = None

        self.lin_update = None

        ############# Your code here #############
        ## (~3 lines of code)
        ## Note:
        ## 1. Initialize the 3 linear layers.
        ## 2. Think through the connection between the mathematical
        ##    definition of the update rule and torch linear layers!
        self.lin_src = self.nn.Linear(in_channels_src, out_channels)
        self.lin_dst = self.nn.Linear(in_channels_dst, out_channels)
        self.lin_update = self.nn.Linear(out_channels + out_channels, out_channels)
        ##########################################

    def forward(
            self,
            node_feature_src,
            node_feature_dst,
            edge_index,
            size=None
    ):
        ############# Your code here #############
        ## (~1 line of code)
        ## Note:
        ## 1. Unlike Colabs 3 and 4, we just need to call self.propagate with
        ## proper/custom arguments.
        self.propagate(edge_index=edge_index, size=size, node_feature_src=node_feature_src,
                       node_feature_dst=node_feature_dst)
        ##########################################

    def message_and_aggregate(self, edge_index, node_feature_src):
        out = None
        ############# Your code here #############
        ## (~1 line of code)
        ## Note:
        ## 1. Different from what we implemented in Colabs 3 and 4, we use message_and_aggregate
        ##    to combine the previously seperate message and aggregate functions.
        ##    The benefit is that we can avoid materializing x_i and x_j
        ##    to make the implementation more efficient.
        ## 2. To implement efficiently, refer to PyG documentation for message_and_aggregate
        ##    and sparse-matrix multiplication:
        ##    https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
        ## 3. Here edge_index is torch_sparse SparseTensor. Although interesting, you
        ##    do not need to deeply understand SparseTensor represenations!
        ## 4. Conceptually, think through how the message passing and aggregation
        ##    expressed mathematically can be expressed through matrix multiplication.
        out = matmul(edge_index, node_feature_src, reduce=self.aggr)
        ##########################################

        return out

    def update(self, aggr_out, node_feature_dst):
        ############# Your code here #############
        ## (~4 lines of code)
        ## Note:
        ## 1. The update function is called after message_and_aggregate
        ## 2. Think through the one-one connection between the mathematical update
        ##    rule and the 3 linear layers defined in the constructor.
        aggr_out = self.lin_update(torch.cat(self.lin_dst(node_feature_dst), self.lin_src(aggr_out)))

        ##########################################

        return aggr_out


class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":
            ############# Your code here #############
            ## (~1 line of code)
            ## Note:
            ## 1. Initialize self.attn_proj, where self.attn_proj should include
            ##    two linear layers. Note, make sure you understand
            ##    which part of the equation self.attn_proj captures.
            ## 2. You should use nn.Sequential for self.attn_proj
            ## 3. nn.Linear and nn.Tanh are useful.
            ## 4. You can model a weight vector (rather than matrix) by using:
            ##    nn.Linear(some_size, 1, bias=False).
            ## 5. The first linear layer should have out_features as args['attn_size']
            ## 6. You can assume we only have one "head" for the attention.
            ## 7. We recommend you to implement the mean aggregation first. After
            ##    the mean aggregation works well in the training, then you can
            ##    implement this part.

            self.attn_proj = nn.Sequential(nn.Linear(args['hidden_size'], args['attn_size'], bias=False), nn.Tanh(),
                                           nn.Linear(args['attn_size'], 1, bias=False))

            ##########################################

    def reset_parameters(self):
        super().reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                )
            )
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb

    def aggregate(self, xs):
        # TODO: Implement this function that aggregates all message type results.
        # Here, xs is a list of tensors (embeddings) with respect to message
        # type aggregation results.

        if self.aggr == "mean":

            ############# Your code here #############
            ## (~2 lines of code)
            ## Note:
            ## 1. Explore the function parameter `xs`!

            return torch.mul(1 / len(xs), torch.sum(xs, dim=-1))

            ##########################################

        elif self.aggr == "attn":
            N = xs[0].shape[0]  # Number of nodes for that node type
            M = len(xs)  # Number of message types for that node type

            x = torch.cat(xs, dim=0).view(M, N, -1)  # M * N * D
            z = self.attn_proj(x).view(M, N)  # M * N * 1
            z = z.mean(1)  # M * 1
            alpha = torch.softmax(z, dim=0)  # M * 1

            # Store the attention result to self.alpha as np array
            self.alpha = alpha.view(-1).data.cpu().numpy()

            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)


def generate_convs(hetero_graph: HeteroGraph, conv, hidden_size, first_layer=False):
    """
    In general, we need to create a dictionary of HeteroGNNConv layers where the keys are message types.

    To get all message types, deepsnap.hetero_graph.HeteroGraph.message_types is useful.
    If we are initializing the first conv layers, we need to get the feature dimension of each node type.
    Using deepsnap.hetero_graph.HeteroGraph.num_node_features(node_type) will return the node feature dimension of node_type.
    In this function, we will set each HeteroGNNConv out_channels to be hidden_size.
    If we are not initializing the first conv layers, all node types will have the smae embedding dimension hidden_size
    and we still set HeteroGNNConv out_channels to be hidden_size for simplicity.

    """
    # TODO: Implement this function that returns a dictionary of `HeteroGNNConv`
    # layers where the keys are message types. `hetero_graph` is deepsnap `HeteroGraph`
    # object and the `conv` is the `HeteroGNNConv`.

    convs = {}

    ############# Your code here #############
    ## (~9 lines of code)
    ## Note:
    ## 1. See the hints above!
    ## 2. conv is of type `HeteroGNNConv`
    for message_type in hetero_graph.message_types:
        node_features_src, node_features_dest = hetero_graph.num_node_features(
            message_type[0]), hetero_graph.num_node_features(message_type[-1])
        if first_layer:
            gnn = conv(node_features_src, node_features_dest, hidden_size)
        else:
            gnn = conv(hidden_size, hidden_size, hidden_size)
        convs.update({message_type: gnn})
    ##########################################

    return convs


class HeteroGNN(torch.nn.Module):
    """
    Now we will make a simple HeteroGNN model which contains only two HeteroGNNWrapperConv layers.
    For the forward function in HeteroGNN, the model is going to be run as following:
    self.convs1→self.bns1→self.relus1→self.convs2→self.bns2→self.relus2→self.post_mps
    """

    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.convs1 = None
        self.convs2 = None

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        ############# Your code here #############
        ## (~10 lines of code)
        ## Note:
        ## 1. For self.convs1 and self.convs2, call generate_convs at first and then
        ##    pass the returned dictionary of `HeteroGNNConv` to `HeteroGNNWrapperConv`.
        ## 2. For self.bns, self.relus and self.post_mps, the keys are node_types.
        ##    `deepsnap.hetero_graph.HeteroGraph.node_types` will be helpful.
        ## 3. Initialize all batchnorms to torch.nn.BatchNorm1d(hidden_size, eps=1).
        ## 4. Initialize all relus to nn.LeakyReLU().
        ## 5. For self.post_mps, each value in the ModuleDict is a linear layer
        ##    where the `out_features` is the number of classes for that node type.
        ##    `deepsnap.hetero_graph.HeteroGraph.num_node_labels(node_type)` will be
        ##    useful.
        self.convs1 = generate_convs(hetero_graph, HeteroGNNConv, args['hidden_size'], True)
        self.convs2 = generate_convs(hetero_graph, HeteroGNNConv, args['hidden_size'], False)
        for node_type in hetero_graph.node_types:
            self.bns1.update({node_type: torch.nn.BatchNorm1d(args['hidden_size'], eps=1)})
            self.bns2.update({node_type: torch.nn.BatchNorm1d(args['hidden_size'], eps=1)})
            self.relus1.update({node_type: torch.nn.LeakyReLU()})
            self.relus2.update({node_type: torch.nn.LeakyReLU()})
            self.post_mps.update(
                {node_type: torch.nn.Linear(args['hidden_size'], hetero_graph.num_node_labels(node_type))}
            )
        ##########################################

    def forward(self, node_feature, edge_index):
        # TODO: Implement the forward function. Notice that `node_feature` is
        # a dictionary of tensors where keys are node types and values are
        # corresponding feature tensors. The `edge_index` is a dictionary of
        # tensors where keys are message types and values are corresponding
        # edge index tensors (with respect to each message type).

        x = node_feature

        ############# Your code here #############
        ## (~7 lines of code)
        ## Note:
        ## 1. `deepsnap.hetero_gnn.forward_op` can be helpful.
        temp_dict = {}
        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        x = forward_op(x, self.bns2)
        x = forward_op(x, self.relus2)
        x = forward_op(x, self.post_mps)
        ##########################################

        return x

    def loss(self, preds, y, indices):
        loss = 0
        loss_func = F.cross_entropy

        ############# Your code here #############
        ## (~3 lines of code)
        ## Note:
        ## 1. For each node type in preds, accumulate computed loss to `loss`
        ## 2. Loss need to be computed with respect to the given index
        ## 3. preds is a dictionary of model predictions keyed by node_type.
        ## 4. indeces is a dictionary of labeled supervision nodes keyed
        ##    by node_type

        pass

        ##########################################

        return loss
