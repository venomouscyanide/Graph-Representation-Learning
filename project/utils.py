import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


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


class LinkEmbedder:
    # TODO: complete edge prediction metrics
    @staticmethod
    def hadamard():
        pass

    @staticmethod
    def max():
        pass

    @staticmethod
    def min():
        pass

    @staticmethod
    def mean():
        pass

    def dot(self):
        pass
