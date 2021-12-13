import warnings

from torch_geometric import seed_everything

from project.statistics.reconstruction_utils import render_graph, link_prediction_cross_validation, TypeOfModel, \
    get_link_embedding, prep_dataset

seed_everything(49)
warnings.filterwarnings(action='ignore')
import torch_geometric.transforms as T
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx

from project.dataset_loader_factory import DatasetLoaderFactory
from project.shallow_models.utils import LinkOperators
from project.statistics.mmd import compute_mmd
from project.utils import MaxDegreeMapping
from project.statistics.mmd import gaussian_emd


class GraphReconstructionMMD:
    def model_graph_reconstruction(self, data, model, dataset, type_of_model: str, viz_graph=False):
        print(f"Model reconstruction for dataset: {dataset}")
        operator = LinkOperators.hadamard
        cv = 10
        clf = link_prediction_cross_validation(model, data, data, cv, type_of_model, operator)

        link_pred_data = get_link_embedding(type_of_model, operator, model, data, arg='edge_index')

        predicted_classes = clf.predict(link_pred_data.detach().cpu().numpy())
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


if __name__ == '__main__':
    data_sans_deg = DatasetLoaderFactory().get('cora', 'delete_me', None)[0]
    prep_dataset(data_sans_deg)

    data_with_deg = DatasetLoaderFactory().get('cora', 'delete_me',
                                               T.OneHotDegree(max_degree=MaxDegreeMapping.MAPPING['cora']))[0]
    prep_dataset(data_with_deg)

    # Link prediction score on test: 0.877564100256005
    print("Running on n2v")
    model = torch.load('node_2_vec_no_degree_information_cora_best_model.model', map_location=torch.device('cpu'))
    mmd = GraphReconstructionMMD().model_graph_reconstruction(data_sans_deg, model, 'cora', TypeOfModel.SHALLOW)

    # Link prediction score on test: 0.8951243838418026
    print("Running on dw")
    model = torch.load('dw_no_degree_information_cora_best_model.model', map_location=torch.device('cpu'))
    mmd = GraphReconstructionMMD().model_graph_reconstruction(data_sans_deg, model, 'cora', TypeOfModel.SHALLOW)

    # Best trial test set accuracy: 0.8744999621933611
    print("Running on gcn")
    model = torch.load('gcn_norm_degree_information_cora_best_model.model', map_location=torch.device('cpu'))
    mmd = GraphReconstructionMMD().model_graph_reconstruction(data_with_deg, model, 'cora', TypeOfModel.DEEP)

    # Best trial test set accuracy: 0.9356783051103775
    print("Running on id_gcn")
    model = torch.load('id_gcn_no_norm_no_degree_information_cora_best_model.model', map_location=torch.device('cpu'))

    mmd = GraphReconstructionMMD().model_graph_reconstruction(data_sans_deg, model, 'cora', TypeOfModel.DEEP)

    """
    # TODO refer below and complete the clustering coefficient et al. well
    clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
    hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    sample_pred.append(hist)
    """
