# ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
import os.path as osp

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression


class DeepWalk(Node2Vec):
    # Deep walk is nothing but node2vec when p and q are 1
    def __init__(self, **kwargs):
        # override p and q to 1 as dw is just a special case of node2vec
        kwargs['p'] = 1
        kwargs['q'] = 1
        super().__init__(**kwargs)


def train_node2vec(data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        # testing here is based on logistic regression using scikit learn
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    for epoch in range(1, 15):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    # plot_points(colors)
    return model


def node_classification_prediction(model, data):
    model = model()
    clf = LogisticRegression(). \
        fit(
        model[data.train_mask].detach().cpu().numpy(),
        data.y[data.train_mask].detach().cpu().numpy()
    )
    return clf.score(model[data.test_mask].detach().cpu().numpy(),
                     data.y[data.test_mask].detach().cpu().numpy())


if __name__ == "__main__":
    dataset = 'Cora'
    path = osp.join('temp_data', dataset)
    dataset = Planetoid(path, dataset)
    data = dataset[0]

    node2vec_model = train_node2vec(data)
    print(f"Node classification score: f{node_classification_prediction(node2vec_model, data)}")
