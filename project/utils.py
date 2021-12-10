import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.utils import to_networkx

from project.dataset_loader_factory import DatasetLoaderFactory


@torch.no_grad()
def plot_points(model, data, dataset):
    # ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
    colors = ['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
              '#ffd700']
    model.eval()
    z = model(torch.arange(data.num_nodes))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()


def show_dataset_stats(dataset: Dataset):
    # Taken from https://colab.research.google.com/drive/1CILdAekIkIh-AX2EXwZ3ZsZ6VcCbwc0t?usp=sharing
    data = dataset[0]
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    print(f'Contains self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print('======================')


def max_degree():
    from project.dataset_loader_factory import DatasetLoaderFactory

    datasets = ['cs', 'physics', 'computers', 'photo', 'ego-facebook', 'karate', 'cora', 'citeseer', 'pubmed']
    path = 'delete_me'
    transform = None

    max_degree_values = {}

    for data in datasets:
        print(f"Processing {data}")
        dataset = DatasetLoaderFactory().get(data, path, transform)
        nx_graph = to_networkx(dataset[0])
        max_degree = max(nx_graph.degree(), key=lambda x: x[1])[0]
        max_degree_values[data] = max_degree

    print(max_degree_values)


def viz_dataset_stats():
    from project.dataset_loader_factory import DatasetLoaderFactory

    datasets = ['cs', 'physics', 'computers', 'photo', 'ego-facebook', 'karate', 'cora', 'citeseer', 'pubmed']
    path = 'delete_me'
    transform = None

    for data in datasets:
        dataset = DatasetLoaderFactory().get(data, path, transform)
        # from project.shallow_models.model_trainer import DataLoader
        # train_data, test_data, val_data = DataLoader().load_data(data, path, 'cpu')
        print(f"Printing stats for :{dataset}")
        show_dataset_stats(dataset)


if __name__ == '__main__':
    # viz_dataset_stats()
    max_degree()


class MaxDegreeMapping:
    MAPPING = {'cs': 11127, 'physics': 23597, 'computers': 12888, 'photo': 2198, 'ego-facebook': 346, 'karate': 33,
               'cora': 1358, 'citeseer': 1422, 'pubmed': 11450}


class DataLoader:
    def load_data(self, dataset: str, path: str, device: str, norm_features: bool, augment_degree_info: bool, num_val,
                  num_test, add_negative_during_load=True):
        """
        add_negative_during_load = False for deep models as negative sampling is uniquely created for each epoch
        """
        transforms = []
        if norm_features:
            transforms.append(T.NormalizeFeatures())

        if augment_degree_info:
            transforms.append(T.OneHotDegree(max_degree=MaxDegreeMapping.MAPPING[dataset]))

        transforms.append(T.RandomLinkSplit(num_val=num_val,
                                            num_test=num_test,
                                            is_undirected=True,
                                            add_negative_train_samples=add_negative_during_load, split_labels=False))
        transforms.append(T.ToDevice(device))

        transform = T.Compose(transforms)
        dataset = DatasetLoaderFactory().get(dataset, path, transform)
        # all datasets contain only one graph, hence the indexing by 0
        train_data, val_data, test_data = dataset[0]
        return train_data, val_data, test_data
