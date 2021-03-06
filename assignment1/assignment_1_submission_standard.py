import itertools
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from cachetools import Cache
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
    TRAINING_RATIO: float = 0.70
    DATASET: str = "KarateClub"
    EPOCHS: int = 1000

    @staticmethod
    def override_defaults(*args):
        Hyperparameters.EMBEDDINGS_SIZE = args[0]
        Hyperparameters.LEARNING_RATE = args[1]
        Hyperparameters.WEIGHT_DECAY = args[2]
        Hyperparameters.TRAINING_RATIO = args[3]
        Hyperparameters.DATASET = args[4]
        Hyperparameters.EPOCHS = args[5]


class CreateNodeFeatureTensor:
    def create(self, data: Dataset) -> torch.tensor:
        dataset_id = str(data)
        if type(cache.get(dataset_id)) == torch.Tensor:
            return cache.get(dataset_id)
        nx_graph = to_networkx(data)
        # features captured
        feature_functions: List[str] = Hyperparameters.ALL_STATISTICS_CAPTURED

        # ref for globals()["function_name"](): https://stackoverflow.com/a/834451
        feature_values: List[List[float]] = [
            list(globals()[feature](nx_graph).values()) for feature in feature_functions
        ]

        # norm harmonic centrality as values are larger than other stats
        self._norm_harmonic_centrality_values(feature_values, nx_graph)

        features = [list(node_embedding) for node_embedding in zip(*feature_values)]
        target_tensor = torch.tensor(features, dtype=torch.float)
        cache[dataset_id] = target_tensor
        return target_tensor

    def _norm_harmonic_centrality_values(self, feature_values, nx_graph):
        """
        normalize harmonic centrality values
        normalized harmonic centrality(node) = sum(1 / distance from node to every other node excluding itself) / (number of nodes - 1)
        ref: https://neo4j.com/docs/graph-data-science/current/algorithms/harmonic-centrality/
        """

        normalization_val = nx_graph.number_of_nodes() - 1
        idx_of_hc_values = Hyperparameters.ALL_STATISTICS_CAPTURED.index("harmonic_centrality")
        feature_values[idx_of_hc_values] = [hc / normalization_val for hc in feature_values[idx_of_hc_values]]


class GCN(torch.nn.Module):
    # TODO: multiple regressor heads?
    # reference: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

    def __init__(self, dataset: Data):
        super().__init__()

        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, Hyperparameters.EMBEDDINGS_SIZE)

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

    def train_helper(self, target: torch.tensor, epoch: int, viz_training: bool, data):
        self.optimizer.zero_grad()
        h = self(data.x, data.edge_index)
        loss = self.criterion(h[data.train_mask], target[data.train_mask])
        loss.backward()
        if viz_training and epoch % 50 == 0:
            print(f"epoch: {epoch}, loss: {loss}")
        self.optimizer.step()


cache = Cache(maxsize=4)


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
            model.train_helper(self.target, epoch, viz_training, self.data)
        return model

    def evaluate(self, trained_model: GCN) -> torch.tensor:
        trained_model.eval()
        out_tensor = trained_model(self.data.x, self.data.edge_index)
        return self._get_accuracy(out_tensor)

    def _set_train_test_mask(self):
        # set train and test mask according to how you set the training ratio hyperparameter
        set_of_all_indices = set(range(len(self.data.x)))

        train_mask_indices = torch.randperm(len(self.data.x))[:int(Hyperparameters.TRAINING_RATIO * len(self.data.x))]
        set_of_training_indices = set(train_mask_indices.detach().tolist())
        test_mask_indices = torch.tensor(list(set_of_all_indices.difference(set_of_training_indices)), dtype=int)

        train_mask = torch.tensor([False for _ in range(len(self.data.x))], dtype=bool)
        test_mask = torch.tensor([False for _ in range(len(self.data.x))], dtype=bool)

        train_mask[train_mask_indices] = True
        test_mask[test_mask_indices] = True

        # sanity check
        assert len(set_of_training_indices.intersection(set(test_mask_indices.detach().tolist()))) == 0 \
            , "train and test sets are not mutually exclusive"

        # overwrite the test and train boolean masks
        self.data.test_mask = test_mask
        self.data.train_mask = train_mask

    def _get_accuracy(self, out: torch.tensor) -> torch.tensor:
        # for some reason cosine sim is giving high similarity no matter what
        # cosine_sim = torch.nn.functional.cosine_similarity(x1=out[self.data.test_mask],
        #                                                    x2=self.target[self.data.test_mask], dim=1)
        # define a custom accuracy metric
        # all the output values must be within the threshold defined in Hyperparameters.ERROR_THRESHOLD

        ## ~~~~START OF OLD ACCURACY CODE~~~
        # diff_tensor = torch.abs(out[self.data.test_mask] - self.target[self.data.test_mask])
        # loss = torch.nn.MSELoss()
        # for dim in range(Hyperparameters.EMBEDDINGS_SIZE):
        #     expectation_for_statistic = self.target[self.data.test_mask][dim]
        #     prediction_for_statistic = out[self.data.test_mask][dim]
        #     with torch.no_grad():
        #         loss(prediction_for_statistic, expectation_for_statistic)
        # within_threshold = torch.le(diff_tensor, Hyperparameters.ERROR_THRESHOLD)
        # numerator = int(torch.count_nonzero(within_threshold == True))
        # denominator = diff_tensor.shape[0] * diff_tensor.shape[1]
        # accuracy = round(numerator / denominator * 100, 4)
        ## ~~~~END OF OLD ACCURACY CODE~~~

        """
        Calculate the MSE across each statistic(column) and return the 8-d tensor which contains the MSE across each statistic
        """
        loss = torch.nn.MSELoss()
        accuracy = []
        for dim in range(Hyperparameters.EMBEDDINGS_SIZE):
            expectation_for_statistic = self.target[self.data.test_mask][:, dim]
            prediction_for_statistic = out[self.data.test_mask][:, dim]
            with torch.no_grad():
                accuracy.append(float(loss(prediction_for_statistic, expectation_for_statistic)))
        return torch.tensor(accuracy)


class HyperParameterCombinations:
    """
    To be manually configured
    """
    ES = [8]
    LR = [0.01, 0.001]
    WD = [1e-5, 1e-3]
    TR = [x / 10 for x in range(1, 9)]
    DATASET = [KarateClub()]
    EPOCHS = [250, 500, 1000]
    ET = [torch.FloatTensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10])]


class TrainAndCaptureResults:
    def __init__(self):
        self.seeds: List[int] = [7, 13, 666]
        self.data = pd.DataFrame(
            columns=["Embedding Size", "Learning Rate", "Weight Decay", "Training Ratio", "Dataset", "Epochs",
                     *[f'{stat}() error' for stat in Hyperparameters.ALL_STATISTICS_CAPTURED],
                     f"Average Error across {Hyperparameters.EMBEDDINGS_SIZE}-d"])

    def _setup(self) -> List:
        combinations = [HyperParameterCombinations.ES, HyperParameterCombinations.LR, HyperParameterCombinations.WD,
                        HyperParameterCombinations.TR, HyperParameterCombinations.DATASET,
                        HyperParameterCombinations.EPOCHS]
        return combinations

    def run(self) -> pd.DataFrame:
        errors = []
        all_combinations = list(itertools.product(*self._setup()))
        total_combinations = len(all_combinations)
        print(f"Total combinations to run exp on: {total_combinations}")
        for index, combination in enumerate(all_combinations):
            print(f"Running combination: {index + 1} of {total_combinations}")
            combination = list(combination)
            data = combination[4]
            error_tensor_sum = torch.tensor([0.0 for _ in range(Hyperparameters.EMBEDDINGS_SIZE)])
            Hyperparameters.override_defaults(*combination)
            for seed in self.seeds:
                torch.manual_seed(seed)
                train_and_eval = TrainAndEvaluate(data[0])
                trained_model = train_and_eval.train_helper(False)
                error_tensor_sum += train_and_eval.evaluate(trained_model)
            avg_mse = torch.div(error_tensor_sum, 3.0)
            avg_mse_across_8d = torch.mean(avg_mse)
            errors.append(avg_mse_across_8d)
            combination.extend([float(x) for x in avg_mse])
            combination.append(float(avg_mse_across_8d))
            combination[4] = str(combination[4])
            self.data.loc[len(self.data)] = combination
        least_error = min(errors)
        print(f"Least avg error across 8-d: {least_error}")
        self.data.sort_values(by=f"Average Error across {Hyperparameters.EMBEDDINGS_SIZE}-d", ascending=True,
                              inplace=True)
        return self.data


if __name__ == '__main__':
    # data = Planetoid(root="delete_me/", name="Cora")[0]
    # torch.manual_seed(13)
    # show_dataset_stats(KarateClub())
    # data = KarateClub()[0]
    # train_and_eval = TrainAndEvaluate(data)
    # trained_model = train_and_eval.train_helper(False)
    # print(train_and_eval.evaluate(trained_model))
    show_dataset_stats(Planetoid(root="delete_me/", name="Cora"))
    # show_dataset_stats(Planetoid(root="delete_me/", name="Cora"))
    # show_dataset_stats(Planetoid(root="delete_me/", name="PubMed"))
    # show_dataset_stats(Planetoid(root="delete_me/", name="CiteSeer"))
    df = TrainAndCaptureResults().run()
    print(df)
