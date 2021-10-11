import torch
from networkx import degree_centrality, eigenvector_centrality
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import torch.nn.functional as F

from assignment1.utils import visualize


class GCN(torch.nn.Module):
    def __init__(self, dataset):
        dataset = dataset
        super(GCN, self).__init__()
        torch.manual_seed(666)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x


def train(data, optimizer, model, criterion, target):
    optimizer.zero_grad()
    h = model(data.x, data.edge_index)
    loss = criterion(h[data.train_mask], target[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h


class CreateNodeFeatureTensor:
    def create(self, data):
        nx_graph = to_networkx(data)
        # features captured
        degree_cent = list(degree_centrality(nx_graph).values())
        eigen_cent = list(eigenvector_centrality(nx_graph).values())
        all_features = [degree_cent, eigen_cent]
        features = [[i, j] for i, j in zip(*all_features)]
        target_tensor = torch.tensor(features, dtype=torch.float)
        return target_tensor


def train_helper():
    dataset = KarateClub()
    data = dataset[0]

    model = GCN(data)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    target_tensor = CreateNodeFeatureTensor().create(data)

    for epoch in range(401):
        loss, h = train(data, optimizer, model, criterion, target_tensor)
        # Visualize the node embeddings every 10 epochs
        if epoch % 50 == 0:
            visualize(h, color=data.y, epoch=epoch, loss=loss)

    return model


def inference(model):
    model.eval()
    dataset = KarateClub()
    data = dataset[0]
    target = CreateNodeFeatureTensor().create(data)
    out = model(data.x[data.test_mask], data.edge_index[data.test_mask])
    breakpoint()


if __name__ == '__main__':
    model = train_helper()
    inference(model)
