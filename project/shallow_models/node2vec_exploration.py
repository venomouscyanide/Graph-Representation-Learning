# ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
# ray tune: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
"""
    Usage : python3.9 node2vec_exploration.py --gpu_count 0

"""
import math
import os
import os.path as osp
from functools import partial
from typing import Dict

import argparse
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
import torch_geometric.transforms as T
import multiprocessing

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import warnings

warnings.simplefilter(action="ignore")


class TrainNode2Vec:
    @staticmethod
    def train_node2vec(config: Dict, cpu_count: int, data_dir: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        train_data, val_data, test_data = DataLoader().load_data("Cora", data_dir, device)
        model = Node2Vec(test_data.edge_index, embedding_dim=config['embedding_dim'], walk_length=config['walk_length'],
                         context_size=config['context_size'], walks_per_node=config['walks_per_node'],
                         num_negative_samples=1, p=config['p'], q=config['q'], sparse=True)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=config['lr'])

        if torch.cuda.is_available():
            device = "cuda:0"
            if gpu_count > 1:
                model = nn.DataParallel(model)
        model.to(device)

        train_data = train_data.to(device)
        validation_data = val_data.to(device)
        test_data = test_data.to(device)

        if torch.cuda.is_available():
            loader = model.module.loader(batch_size=config['batch_size'], shuffle=True, num_workers=cpu_count)
        else:
            loader = model.loader(batch_size=config['batch_size'], shuffle=True, num_workers=cpu_count)

        def _train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    loss = model.module.loss(pos_rw.to(device), neg_rw.to(device))
                else:
                    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        @torch.no_grad()
        def _test():
            model.eval()
            roc_score = link_prediction(model, test_data, validation_data)
            return roc_score

        loss = acc = 0
        for epoch in range(1, 10001):
            loss = _train()
            acc = _test()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
            tune.report(loss=loss, accuracy=acc)
        return model


def node_classification_prediction(model, data):
    model = model()
    clf = LogisticRegression(). \
        fit(
        model[data.train_mask].detach().cpu().numpy(),
        data.y[data.train_mask].detach().cpu().numpy()
    )
    return clf.score(model[data.test_mask].detach().cpu().numpy(),
                     data.y[data.test_mask].detach().cpu().numpy())


def link_prediction(model, train_data, test_data):
    model = model()

    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=1500)
    link_pred_pipeline = Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

    link_features_train_128 = model[train_data.edge_label_index[0]] * model[train_data.edge_label_index[1]]
    link_features_test_128 = model[test_data.edge_label_index[0]] * model[test_data.edge_label_index[1]]

    link_pred_pipeline.fit(link_features_train_128.detach().cpu().numpy(), train_data.edge_label.detach().cpu().numpy())
    final_prediction = link_pred_pipeline.predict_proba(link_features_test_128.detach().cpu().numpy())

    positive_column = list(link_pred_pipeline.classes_).index(1)
    roc_scores = roc_auc_score(test_data.edge_label.detach().cpu().numpy(), final_prediction[:, positive_column])
    return roc_scores


class HyperParameterTuning:
    CONFIG = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([int(math.pow(2, n)) for n in range(1, 10)]),
        "context_size": tune.choice([5, 10]),
        "embedding_dim": tune.choice([64, 128, 256]),
        "walk_length": tune.choice([10, 15, 20]),
        "walks_per_node": tune.choice([10, 20]),
        "p": tune.choice([0.25 * n for n in range(16)]),
        "q": tune.choice([0.25 * n for n in range(16)]),
    }

    RAYTUNE_CONFIG = {
        'num_samples': 10,
        'max_epochs': 150
    }


class DataLoader:
    def load_data(self, dataset: str, path: str, device):
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                              add_negative_train_samples=True, split_labels=False),
            T.ToDevice(device)
        ])
        dataset = Planetoid(path, name=dataset, transform=transform)
        train_data, val_data, test_data = dataset[0]
        return train_data, val_data, test_data


class Tuner:
    def tune(self, data_dir, cpu_count, gpu_count):
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=HyperParameterTuning.RAYTUNE_CONFIG['max_epochs'],
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
        result = tune.run(
            tune.with_parameters(TrainNode2Vec().train_node2vec, cpu_count=cpu_count, data_dir=data_dir),
            resources_per_trial={"cpu": cpu_count, "gpu": gpu_count},
            config=HyperParameterTuning.CONFIG,
            num_samples=HyperParameterTuning.RAYTUNE_CONFIG['num_samples'],
            scheduler=scheduler,
            progress_reporter=reporter,
            log_to_file=True)

        best_trial = result.get_best_trial("accuracy", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final train loss: {}".format(best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        train_data, val_data, test_data = DataLoader().load_data("Cora", data_dir, device)
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)
        best_trained_model = Node2Vec(test_data.edge_index, embedding_dim=best_trial.config['embedding_dim'],
                                      walk_length=best_trial.config['walk_length'],
                                      context_size=best_trial.config['context_size'],
                                      walks_per_node=best_trial.config['walks_per_node'],
                                      num_negative_samples=1, p=best_trial.config['p'], q=best_trial.config['q'],
                                      sparse=True)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)

        test_acc = link_prediction(best_trained_model, train_data, val_data)
        print("Best trial test set accuracy: {}".format(test_acc))
        torch.save(best_trained_model, 'node2vec_best_model.model')
        return best_trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tuning for n2v/dw", )
    parser.add_argument('--gpu_count', help='Set available GPUs to tune on', required=True, type=int)
    parser.add_argument('--cpu_count', help='Set available CPUs to tune on', required=True, type=int)
    args = parser.parse_args()

    dataset = 'Cora'
    path = osp.join('../temp_data', dataset)

    cpu_count = int(multiprocessing.cpu_count() / 2)
    gpu_count = args.gpu_count

    node2vec_model = Tuner().tune(path, cpu_count, gpu_count)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data, test_data, val_data = DataLoader().load_data("Cora", path, 'cpu')
    print(f"Node classification score: {node_classification_prediction(node2vec_model, test_data)}")
    print(f"Link prediction score on train: {link_prediction(node2vec_model, train_data, train_data)}")
    print(f"Link prediction score on test: {link_prediction(node2vec_model, train_data, test_data)}")
    print(f"Link prediction score on val: {link_prediction(node2vec_model, train_data, val_data)}")
