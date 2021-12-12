import warnings

warnings.filterwarnings(action='ignore')
import torch_geometric.transforms as T
from pylab import show
import networkx as nx
import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_networkx, negative_sampling

from project.dataset_loader_factory import DatasetLoaderFactory
from project.shallow_models.utils import LinkOperators
from project.statistics.mmd import compute_mmd
from project.utils import CustomDataLoader, MaxDegreeMapping
import matplotlib.pyplot as plt
from project.statistics.mmd import emd, l2, gaussian_emd


def link_prediction_cross_validation(model, train_data, test_data, dataset, operator=LinkOperators.hadamard):
    # test_data can be train/test/validation test. The name is rather deceiving.
    cv = 10
    if dataset == 'karate':
        cv = 2

    lr_clf = LogisticRegressionCV(Cs=10, cv=cv, scoring="roc_auc", max_iter=1500)
    link_pred_pipeline = Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

    z = model.encode(train_data.x, train_data.edge_index)
    link_features_train_d_dim = operator(z[train_data.edge_label_index[0]], z[train_data.edge_label_index[1]])
    link_features_test_d_dim = operator(z[test_data.edge_label_index[0]], z[test_data.edge_label_index[1]])

    link_pred_pipeline.fit(link_features_train_d_dim.detach().cpu().numpy(),
                           train_data.edge_label.detach().cpu().numpy())
    final_prediction = link_pred_pipeline.predict_proba(link_features_test_d_dim.detach().cpu().numpy())

    positive_column = list(link_pred_pipeline.classes_).index(1)
    roc_scores = roc_auc_score(test_data.edge_label.detach().cpu().numpy(), final_prediction[:, positive_column])

    print("Trained linear_reg and roc_scores: ", roc_scores)
    return link_pred_pipeline


def deep_model_graph_reconstruction(data, model, viz_graph=False):
    neg_edge_index = negative_sampling(data.edge_index, method='dense')

    edge_label_index = torch.cat(
        [data.edge_index, neg_edge_index],
        dim=-1,
    )

    edge_label = torch.cat([
        torch.ones(neg_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0)

    data.edge_label_index = edge_label_index
    data.edge_label = edge_label

    clf = link_prediction_cross_validation(model, data, data, dataset='cora')
    operator = LinkOperators.hadamard

    # model = model()
    z = model.encode(data.x, data.edge_index)
    test_data = operator(z[data.edge_index[0]], z[data.edge_index[1]])
    predicted_classes = clf.predict(test_data.detach().cpu().numpy())
    predicted_edge_indices = data.edge_index.t().detach().numpy()[predicted_classes.astype(np.bool)]

    actual_graph = to_networkx(data, to_undirected=True)
    actual_graph.remove_edges_from(nx.selfloop_edges(actual_graph))

    print(f"Predicted Edge Indices: {len(predicted_edge_indices)}, Actual Edge Indices: {len(data.edge_index.t())}")

    predicted_graph = nx.Graph()
    predicted_graph.add_nodes_from([_ for _ in range(data.num_nodes)])
    predicted_graph.add_edges_from([tuple(item) for item in predicted_edge_indices])

    predicted_graph.remove_edges_from(nx.selfloop_edges(predicted_graph))
    predicted_graph = predicted_graph.to_undirected()

    predicted_graph_deg_histogram = np.array(nx.degree_histogram(predicted_graph))
    actual_graph_deg_histogram = np.array(nx.degree_histogram(actual_graph))

    mmd = compute_mmd([actual_graph_deg_histogram], [predicted_graph_deg_histogram], kernel=gaussian_emd,
                      is_parallel=False, is_hist=True)

    print("MMD", mmd)

    if viz_graph:
        render_graph(actual_graph)
        render_graph(predicted_graph)

    return mmd


def render_graph(nx_graph):
    print("Starting to viz graph")
    nx_graph = nx_graph.to_directed()
    pos = nx.spring_layout(nx_graph)
    plt.figure(figsize=(15, 15))
    nx.draw(nx_graph, pos=pos, cmap=plt.get_cmap('coolwarm'))
    show()


if __name__ == '__main__':
    # Best trial test set accuracy: 0.8744999621933611
    data = DatasetLoaderFactory().get('cora', 'delete_me', T.OneHotDegree(max_degree=MaxDegreeMapping.MAPPING['cora'])
                                      )[0]
    model = torch.load('gcn_norm_degree_information_cora_best_model.model', map_location=torch.device('cpu'))

    mmd = deep_model_graph_reconstruction(data, model)

    # Best trial test set accuracy: 0.9356783051103775
    data = DatasetLoaderFactory().get('cora', 'delete_me', None)[0]
    model = torch.load('id_gcn_no_norm_no_degree_information_cora_best_model.model', map_location=torch.device('cpu'))

    mmd = deep_model_graph_reconstruction(data, model)

"""
# TODO refer below and complete the clustering coefficient as well
clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
sample_pred.append(hist)
"""
