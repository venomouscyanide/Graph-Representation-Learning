from copy import copy, deepcopy

import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.datasets import Planetoid, KarateClub

from assignment1.assignment_1_submission_standard import CreateNodeFeatureTensor, TrainAndEvaluate
from assignment1.assignment_prior_study import GCN, train_helper


class CompareTriplets:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.data = self.dataset[0]
        self.df = pd.DataFrame(columns=["Dataset", "Method", "Accuracy"])

    def _get_accuracy(self, model: GCN, data) -> float:
        with torch.no_grad():
            model_output, h = model(data.x, data.edge_index)
            num = int(
                torch.count_nonzero(
                    torch.argmax(model_output[~ data.train_mask], dim=1) == data.y[~ data.train_mask])
            )
            denom = int(torch.count_nonzero(~ data.train_mask))
            return round(num / denom * 100, 4)

    def execute(self) -> pd.DataFrame:
        vanilla_accuracy = self._run_vanilla_x()
        self.df.loc[len(self.df)] = [str(self.dataset), "Vanilla Implementation", vanilla_accuracy]
        nx_accuracy = self._run_nx_stat_x()
        self.df.loc[len(self.df)] = [str(self.dataset), "Pure NX Stats Implementation", nx_accuracy]
        model_stat_accuracy = self._run_model_stat_x()
        self.df.loc[len(self.df)] = [str(self.dataset), "Learnt NX Stats Implementation", model_stat_accuracy]
        return self.df

    def _run_vanilla_x(self):
        dataset = deepcopy(self.dataset)
        model = GCN(dataset=dataset)
        data = deepcopy(self.data)
        train_helper(model, data)
        _, h = model(data.x, data.edge_index)
        accuracy = self._get_accuracy(model, data)
        return accuracy

    def _run_nx_stat_x(self):
        dataset = deepcopy(self.dataset)
        data = deepcopy(self.data)
        model = GCN(dataset=dataset, dim=8)
        x = CreateNodeFeatureTensor().create(data)
        data.x = x
        train_helper(model, data)
        _, h = model(data.x, data.edge_index)
        accuracy = self._get_accuracy(model, data)
        return accuracy

    def _run_model_stat_x(self):
        dataset = deepcopy(self.dataset)
        model = GCN(dataset=dataset, dim=8)
        data = deepcopy(self.data)

        train_and_eval = TrainAndEvaluate(data)
        trained_model = train_and_eval.train_helper(False)
        accuracy = train_and_eval.evaluate(trained_model)
        print(f"Node statistics learning accuracy: {accuracy}")

        trained_model.eval()
        copy_data = deepcopy(self.data)
        copy_data.x = trained_model(data.x, data.edge_index).detach()

        train_helper(model, copy_data)
        _, h = model(copy_data.x, copy_data.edge_index)
        accuracy = self._get_accuracy(model, copy_data)
        return accuracy


if __name__ == '__main__':
    print(CompareTriplets(Planetoid(root="delete_me/", name="Cora")).execute())
