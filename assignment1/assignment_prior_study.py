import time

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.nn import Linear
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx


def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(h, pos=None, with_labels=True,
                         node_color=color, cmap="Dark2")
    plt.show()


def show_dataset_as_networkx_graph(data):
    G = to_networkx(data, to_undirected=True)
    visualize(G, color=data.y)


def train_helper(model, data):
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

    for epoch in range(401):
        loss, h = train(data, optimizer, model, criterion)
        # Visualize the node embeddings every 10 epochs
        if epoch % 50 == 0:
            visualize(h, color=data.y, epoch=epoch, loss=loss)

            time.sleep(0.3)


def train(data, optimizer, model, criterion):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


class GCN(torch.nn.Module):
    def __init__(self, dataset):
        dataset = dataset
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h


def explore_gcn():
    dataset = KarateClub()
    model = GCN(dataset)

    data = dataset[0]
    _, h = model(data.x, data.edge_index)

    show_dataset_as_networkx_graph(data)
    visualize(h, color=data.y)
    train_helper(model, data)
    print("Completed training 400 epochs")


if __name__ == '__main__':
    explore_gcn()
