import random
from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import *
from torch_geometric.nn import GCNConv
from networkx.algorithms.centrality import *
from networkx.algorithms.cluster import *
from torch_geometric.utils import to_networkx

from assignment1.utils import show_dataset_as_networkx_graph, show_dataset_stats


class Hyperparameters:
    # Ref for centrality APIs: https://networkx.org/documentation/stable/reference/algorithms/centrality.html
    # Ref for clustering APIs:
    # I just focus on learning centrality/clustering related info
    # target tensor size is 8
    ALL_STATISTICS_CAPTURED: List[str] = ['degree_centrality', 'in_degree_centrality',
                                          'eigenvector_centrality', 'harmonic_centrality',
                                          'closeness_centrality', 'betweenness_centrality', 'load_centrality',
                                          'clustering']
    # upto 8 dimensional vector captured for each node.
    # Each dimension corresponds to a centrality measure in the same order as they are listed above
    EMBEDDINGS_SIZE: int = 8
    LEARNING_RATE: float = 0.01
    WEIGHT_DECAY: float = 1e-5
    TRAINING_RATIO: float = 0.10
    DATASET: str = "KarateClub"
    EPOCHS: int = 500
    ERROR_THRESHOLD: float = 1


class CreateNodeFeatureTensor:
    def create(self, data: Data):
        nx_graph = to_networkx(data)
        random.seed(13)
        # features captured
        feature_functions: List[str] = random.sample(Hyperparameters.ALL_STATISTICS_CAPTURED,
                                                     Hyperparameters.EMBEDDINGS_SIZE)
        # ref for globals()["function_name"](): https://stackoverflow.com/a/834451
        feature_values: List[List[float]] = [
            list(globals()[feature](nx_graph).values()) for feature in feature_functions
        ]
        features = [list(node_embedding) for node_embedding in zip(*feature_values)]
        target_tensor = torch.tensor(features, dtype=torch.float)
        return target_tensor


class GCN(torch.nn.Module):
    # reference: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

    def __init__(self, dataset):
        super().__init__()

        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 8)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=Hyperparameters.LEARNING_RATE,
                                          weight_decay=Hyperparameters.WEIGHT_DECAY)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)

        x = F.relu(x)
        x = self.conv2(x, edge_index)

        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return x

    def train_helper(self, target: torch.tensor, epoch: int, viz_training: bool):
        self.optimizer.zero_grad()
        h = self(data.x, data.edge_index)
        loss = self.criterion(h[data.train_mask], target[data.train_mask])
        loss.backward()
        if viz_training and epoch % 50 == 0:
            print(f"epoch: {epoch}, loss: {loss}")
        self.optimizer.step()


class TrainAndEvaluate:
    def __init__(self, data: Dataset):
        self.data = data
        self.target = CreateNodeFeatureTensor().create(data)

    def train_helper(self, viz_data: bool = False, viz_training: bool = False):
        if viz_data:
            show_dataset_as_networkx_graph(self.data)
        self._set_train_test_mask()
        model = GCN(self.data)
        for epoch in range(Hyperparameters.EPOCHS):
            model.train_helper(self.target, epoch, viz_training)
        return model

    def evaluate(self, trained_model: GCN):
        trained_model.eval()
        out_tensor = trained_model(self.data.x, self.data.edge_index)
        out_numpy = out_tensor.detach().numpy()
        target_numpy = self.target.detach().numpy()
        print(self._get_accuracy(out_tensor))

    def _set_train_test_mask(self):
        # set train and test mask according to how you set the training ratio hyperparameter
        set_of_all_indices = set(range(len(data.x)))

        train_mask_indices = torch.randperm(len(data.x))[:int(Hyperparameters.TRAINING_RATIO * len(data.x))]
        set_of_training_indices = set(train_mask_indices.detach().tolist())
        test_mask_indices = torch.tensor(list(set_of_all_indices.difference(set_of_training_indices)), dtype=int)

        train_mask = torch.tensor([False for _ in range(len(data.x))], dtype=bool)
        test_mask = torch.tensor([False for _ in range(len(data.x))], dtype=bool)

        train_mask[train_mask_indices] = True
        test_mask[test_mask_indices] = True

        # sanity check
        assert len(set_of_training_indices.intersection(set(test_mask_indices.detach().tolist()))) == 0 \
            , "train and test sets are not mutually exclusive"

        # overwrite the test and train boolean masks
        self.data.test_mask = test_mask
        self.data.train_mask = train_mask

    def _get_accuracy(self, out):
        # for some reason cosine sim is giving high similarity no matter what
        # define a custom accuracy metric
        # cosine_sim = torch.nn.functional.cosine_similarity(x1=out[self.data.test_mask],
        #                                                    x2=self.target[self.data.test_mask], dim=1)
        a = out[self.data.train_mask].detach().numpy()
        b = self.target[self.data.train_mask].detach().numpy()
        # cosine_sim_numpy = cosine_sim.detach().numpy()
        # print(cosine_sim_numpy)
        diff_tensor = torch.abs(out - self.target)
        denom = int(torch.bincount(data.test_mask.to(int))[1])
        num_tensor = torch.flatten(diff_tensor.le(Hyperparameters.ERROR_THRESHOLD).to(torch.int32))
        num = int(torch.bincount(num_tensor)[1])
        # print(data.train_mask)
        # cdist = torch.cdist(out, self.target, p=2)
        # cdist = torch.cdist(out[self.data.test_mask], self.target[self.data.test_mask], p=2)
        # print(cdist)
        accuracy = num / denom * 100
        return accuracy


if __name__ == '__main__':
    # data = Planetoid(root="delete_me/", name="Cora")[0]
    show_dataset_stats(KarateClub())
    data = KarateClub()[0]
    train_and_eval = TrainAndEvaluate(data)
    trained_model = train_and_eval.train_helper(False)
    train_and_eval.evaluate(trained_model)
