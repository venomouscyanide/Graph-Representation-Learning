import networkx as nx
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import show
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import negative_sampling


def prep_dataset(data):
    sample_size = data.edge_index.size(1)
    neg_edge_index = negative_sampling(data.edge_index, method='dense', num_nodes=data.num_nodes,
                                       num_neg_samples=sample_size)

    edge_label_index = torch.cat(
        [data.edge_index, neg_edge_index],
        dim=-1,
    )

    edge_label = torch.cat([
        torch.ones(sample_size),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0)

    data.edge_label_index = edge_label_index
    data.edge_label = edge_label


def render_graph(nx_graph):
    print("Starting to viz graph")
    nx_graph = nx_graph.to_directed()
    pos = nx.spring_layout(nx_graph)
    plt.figure(figsize=(15, 15))
    nx.draw(nx_graph, pos=pos, cmap=plt.get_cmap('coolwarm'))
    show()


def get_link_embedding(type_of_model, operator, model, data, arg):
    if type_of_model == TypeOfModel.DEEP:
        z = model.encode(data.x, data.edge_index)
        link_embeddings = operator(z[getattr(data, arg)[0]], z[getattr(data, arg)[1]])
    elif type_of_model == TypeOfModel.SHALLOW:
        model = model()
        link_embeddings = operator(model[getattr(data, arg)[0]], model[getattr(data, arg)[1]])
    elif type_of_model == TypeOfModel.MLP:
        out = model(data.x)
        link_embeddings = operator(out[getattr(data, arg)[0]], out[getattr(data, arg)[1]])
    else:
        raise NotImplementedError
    return link_embeddings


def link_prediction_cross_validation(model, train_data, test_data, cv, type_of_model, operator):
    # test_data can be train/test/validation test. The name is rather deceiving.
    if cv > 1:
        lr_clf = LogisticRegressionCV(Cs=10, cv=cv, scoring="roc_auc", max_iter=1500)
    else:
        lr_clf = LogisticRegression()
    link_pred_pipeline = Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

    link_features_train_d_dim = get_link_embedding(type_of_model, operator, model, train_data, arg='edge_label_index')
    link_features_test_d_dim = get_link_embedding(type_of_model, operator, model, test_data, arg='edge_label_index')

    link_pred_pipeline.fit(link_features_train_d_dim.detach().cpu().numpy(),
                           train_data.edge_label.detach().cpu().numpy())
    final_prediction = link_pred_pipeline.predict_proba(link_features_test_d_dim.detach().cpu().numpy())

    positive_column = list(link_pred_pipeline.classes_).index(1)
    roc_scores = roc_auc_score(test_data.edge_label.detach().cpu().numpy(), final_prediction[:, positive_column])

    print("Trained linear_reg and roc_scores: ", roc_scores)
    return link_pred_pipeline


class TypeOfModel:
    DEEP: str = 'deep'
    SHALLOW: str = 'shallow'
    MLP: str = 'mlp'
