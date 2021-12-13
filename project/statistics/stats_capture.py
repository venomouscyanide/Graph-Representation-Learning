import warnings

from project.statistics.reconstruction_utils import render_graph, link_prediction_cross_validation, TypeOfModel, \
    get_link_embedding, prep_dataset

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


class MMDTypes:
    CLUSTERING_COEF: str = 'clustering'
    DEG_DISTRIBUTION: str = 'deg_distribution'
    BETWEENESS: str = 'betweeness_centrality'
    CLOSENESS: str = 'closeness_centrality'


class HistogramGenerator:
    def generate(self, mmd_type, predicted_graph, actual_graph):
        if mmd_type == MMDTypes.DEG_DISTRIBUTION:
            actual_histogram = np.array(nx.degree_histogram(actual_graph))
            predicted_histogram = np.array(nx.degree_histogram(predicted_graph))
        elif mmd_type == MMDTypes.CLUSTERING_COEF:
            bins = 25
            actual_stats = list(nx.clustering(actual_graph).values())
            predicted_stats = list(nx.clustering(predicted_graph).values())
            actual_histogram, _ = np.histogram(actual_stats, bins=bins, range=(0.0, 1.0), density=False)
            predicted_histogram, _ = np.histogram(predicted_stats, bins=bins, range=(0.0, 1.0), density=False)
        elif mmd_type == MMDTypes.BETWEENESS:
            bins = 1000
            actual_stats = list(nx.betweenness_centrality(actual_graph).values())
            predicted_stats = list(nx.betweenness_centrality(predicted_graph).values())
            actual_histogram, _ = np.histogram(actual_stats, bins=bins, range=(0.0, 1.0), density=False)
            predicted_histogram, _ = np.histogram(predicted_stats, bins=bins, range=(0.0, 1.0), density=False)
        elif mmd_type == MMDTypes.CLOSENESS:
            bins = 25
            actual_stats = list(nx.closeness_centrality(actual_graph).values())
            predicted_stats = list(nx.closeness_centrality(predicted_graph).values())
            actual_histogram, _ = np.histogram(actual_stats, bins=bins, range=(0.0, 1.0), density=False)
            predicted_histogram, _ = np.histogram(predicted_stats, bins=bins, range=(0.0, 1.0), density=False)
        else:
            raise NotImplementedError
        return actual_histogram, predicted_histogram


class GraphReconstructionMMD:
    def model_graph_reconstruction(self, data, model, dataset, type_of_model: str, mmd_type, viz_graph=False, cv=10):
        print(f"Model reconstruction for dataset: {dataset}")
        operator = LinkOperators.hadamard
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

        predicted_graph_histogram, actual_graph_histogram = HistogramGenerator().generate(mmd_type, actual_graph,
                                                                                          predicted_graph)

        mmd = compute_mmd([actual_graph_histogram], [predicted_graph_histogram], kernel=gaussian_emd, is_parallel=False,
                          is_hist=True)

        print(f"MMD for {mmd_type} on {dataset} using {model} is {mmd}")

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
    mmd = GraphReconstructionMMD().model_graph_reconstruction(data_sans_deg, model, 'cora', TypeOfModel.SHALLOW,
                                                              MMDTypes.DEG_DISTRIBUTION, cv=1)

    # Link prediction score on test: 0.8951243838418026
    print("Running on dw")
    model = torch.load('dw_no_degree_information_cora_best_model.model', map_location=torch.device('cpu'))
    mmd = GraphReconstructionMMD().model_graph_reconstruction(data_sans_deg, model, 'cora', TypeOfModel.SHALLOW,
                                                              MMDTypes.CLUSTERING_COEF, cv=1)

    # Best trial test set accuracy: 0.8744999621933611
    print("Running on gcn")
    model = torch.load('gcn_norm_degree_information_cora_best_model.model', map_location=torch.device('cpu'))
    mmd = GraphReconstructionMMD().model_graph_reconstruction(data_with_deg, model, 'cora', TypeOfModel.DEEP,
                                                              MMDTypes.CLOSENESS, cv=1)

    # Best trial test set accuracy: 0.9356783051103775
    print("Running on id_gcn")
    model = torch.load('id_gcn_no_norm_no_degree_information_cora_best_model.model', map_location=torch.device('cpu'))

    mmd = GraphReconstructionMMD().model_graph_reconstruction(data_sans_deg, model, 'cora', TypeOfModel.DEEP,
                                                              MMDTypes.BETWEENESS, cv=1)
