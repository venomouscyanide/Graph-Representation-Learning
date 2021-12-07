import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import Dataset


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
    viz_dataset_stats()
