# ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
import os.path as osp

import torch
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
import torch_geometric.transforms as T
import multiprocessing

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import warnings

warnings.simplefilter(action="ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepWalk(Node2Vec):
    # Deep walk is nothing but node2vec when p and q are 1
    def __init__(self, **kwargs):
        # override p and q to 1 as dw is just a special case of node2vec
        kwargs['p'] = 1
        kwargs['q'] = 1
        super().__init__(**kwargs)


def train_node2vec(data):
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    num_workers = int(multiprocessing.cpu_count() / 2)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
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


def link_prediction(model, train_data, test_data):
    model = model()

    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=1500)
    link_pred_pipeline = Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

    link_features_train_128 = model[train_data.edge_label_index[0]] * model[train_data.edge_label_index[1]]
    link_features_test_128 = model[test_data.edge_label_index[0]] * model[test_data.edge_label_index[1]]

    link_pred_pipeline.fit(link_features_train_128.detach().cpu().numpy(), train_data.edge_label.detach().cpu().numpy())
    final_prediction = link_pred_pipeline.predict_proba(link_features_test_128.detach().cpu().numpy())

    positive_column = list(link_pred_pipeline.classes_).index(1)
    roc_scores = roc_auc_score(test_data.edge_label.detach().cpu().numpy(), final_prediction[:, positive_column])
    # print(roc_scores)
    # print(link_pred_pipeline.score(link_features_test_128.detach().cpu().numpy(), test_data.edge_label))
    return roc_scores


if __name__ == "__main__":
    dataset = 'Cora'
    path = osp.join('temp_data', dataset)

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=True, split_labels=False),
        T.ToDevice(device)
    ])
    print(f"Using device: {device}")
    dataset = Planetoid(path, name='Cora', transform=transform)
    train_data, val_data, test_data = dataset[0]

    node2vec_model = train_node2vec(train_data)

    print(f"Node classification score: {node_classification_prediction(node2vec_model, test_data)}")

    print(f"Link prediction score on train: {link_prediction(node2vec_model, train_data, train_data)}")
    print(f"Link prediction score on test: {link_prediction(node2vec_model, train_data, test_data)}")
    print(f"Link prediction score on val: {link_prediction(node2vec_model, train_data, val_data)}")
