"""
Authored by Rui F David https://github.com/rfdavid
"""
import torch
import networkx as nx
import matplotlib.pyplot as plt


# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None, figsize=(8, 8), labels=False):
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=labels,
                         node_color=color, cmap="Set2")
    plt.show()


# from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

# dataset = KarateClub()
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]  # Get the first graph object.

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y, figsize=(15, 15))

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


model = GCN()
print(model)

import time

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(data):
    optimizer.zero_grad()  # Clear gradients.
    # data.x = torch.Size([34, 34])
    # tensor([1., 0., 0., 0., ...])
    # data.edge_index = torch.Size([2, 156])
    # 256 connections between the nodes
    #
    # out => linear classifier
    # h   => final GNN embedding space
    # out, h = model(data.x, data.edge_index)
    out = model(data)

    # Compute the loss solely based on the training nodes.
    # out[data.train_mask] example:
    #   tensor([[0.0069, 0.4383, 0.6527, 0.5311],
    #           [0.0751, 0.4670, 0.6761, 0.5893],
    #           [0.0644, 0.4349, 0.6395, 0.5877],
    #           [0.0887, 0.4226, 0.6209, 0.6145]], grad_fn=<IndexBackward>)
    #
    # data.y[data.train_mask] example:
    #   tensor([1, 3, 0, 2])
    # loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    # Derive gradients
    loss.backward()

    # Update parameters based on gradients
    optimizer.step()

    return out, loss


def run_training(data):
    total = int(data.test_mask.sum())
    max_accuracy = 0

    for epoch in range(401):
        model.train()
        out, loss = train(data)

        model.eval()
        correct = (out[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).sum()
        accuracy = correct / total
        max_accuracy = max(max_accuracy, accuracy)

        if epoch % 10 == 0:
            # visualize(h, color=data.y, epoch=epoch, loss=loss)
            print('Accuracy: {:.2f}%'.format(accuracy * 100))

    print("Max accuracy: {:.2f}%".format(max_accuracy * 100))


from torch_geometric.data import Data

# Centralities
betweness_centrality = nx.betweenness_centrality(G)
degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Clustering
triangles = nx.triangles(G)
clustering = nx.clustering(G)
square_clustering = nx.square_clustering(G)

x = data.x

# for i, value in enumerate(x):
#     # print(x[i][0])
#     x[i][0] = betweness_centrality[i]
#     x[i][1] = degree_centrality[i]
#     x[i][2] = eigenvector_centrality[i]
#     x[i][3] = closeness_centrality[i]
#     x[i][4] = triangles[i]
#     x[i][5] = clustering[i]
#     x[i][7] = square_clustering[i]

data_with_metrics = Data(x=x, edge_index=data.edge_index, y=data.y, train_mask=data.train_mask,
                         test_mask=data.test_mask)
run_training(data_with_metrics)
