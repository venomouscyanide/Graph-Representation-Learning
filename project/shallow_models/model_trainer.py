# ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
# ray tune: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
"""
    Usage : python3.9 -m project.shallow_models.model_trainer --gpu_count 0
                                                              --cpu_count 4
                                                              --dataset "cora"
                                                              --identifier "cora_run"
                                                              --model_name "node2vec"

"""
import math
import os
import os.path as osp

import argparse
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch_geometric.transforms as T

import warnings

from project.dataset_loader_factory import DatasetLoaderFactory
from project.shallow_models.model_factory import ModelFactory, ModelTrainFactory
from project.shallow_models.utils import link_prediction, node_classification_prediction

warnings.simplefilter(action="ignore")


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
        'max_epochs': 100
    }

    DATASET_SPLIT_CONFIG = {
        "num_val": 0.05,
        "num_test": 0.1
    }
    NORMALIZE_FEATURES = True


class DataLoader:
    def load_data(self, dataset: str, path: str, device: str):
        if HyperParameterTuning.NORMALIZE_FEATURES:
            transform = T.Compose([
                T.NormalizeFeatures(),
                T.RandomLinkSplit(num_val=HyperParameterTuning.DATASET_SPLIT_CONFIG['num_val'],
                                  num_test=HyperParameterTuning.DATASET_SPLIT_CONFIG['num_test'], is_undirected=True,
                                  add_negative_train_samples=True, split_labels=False),
                T.ToDevice(device)
            ])
        else:
            transform = T.Compose([
                T.RandomLinkSplit(num_val=HyperParameterTuning.DATASET_SPLIT_CONFIG['num_val'],
                                  num_test=HyperParameterTuning.DATASET_SPLIT_CONFIG['num_test'], is_undirected=True,
                                  add_negative_train_samples=True, split_labels=False),
                T.ToDevice(device)
            ])
        dataset_klass = DatasetLoaderFactory().get(dataset)
        dataset = dataset_klass(path, name=dataset, transform=transform)
        train_data, val_data, test_data = dataset[0]
        return train_data, val_data, test_data


class Tuner:
    def tune(self, data_dir: str, cpu_count: int, gpu_count: int, dataset: str, identifier: str, model_name: str):
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=HyperParameterTuning.RAYTUNE_CONFIG['max_epochs'],
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
        trainer_function = ModelTrainFactory().get(model_name)
        result = tune.run(
            tune.with_parameters(trainer_function, gpu_count=gpu_count, cpu_count=cpu_count, data_dir=data_dir,
                                 dataset=dataset),
            resources_per_trial={"cpu": cpu_count, "gpu": gpu_count},
            config=HyperParameterTuning.CONFIG,
            num_samples=HyperParameterTuning.RAYTUNE_CONFIG['num_samples'],
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=os.path.join(identifier, "ray_results"),
            log_to_file=True)

        best_trial = result.get_best_trial("accuracy", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final train loss: {}".format(best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        train_data, val_data, test_data = DataLoader().load_data(dataset, data_dir, device)
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)
        model = ModelFactory().get(model_name)
        best_trained_model = model(test_data.edge_index, embedding_dim=best_trial.config['embedding_dim'],
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
        torch.save(best_trained_model, os.path.join(identifier, 'node2vec_best_model.model'))
        return best_trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tuning for n2v/dw/<x> shallow encoders", )
    parser.add_argument('--dataset', help='Choose the dataset to tune on', required=True, type=str)
    parser.add_argument('--gpu_count', help='Set available GPUs to tune on', required=True, type=int)
    parser.add_argument('--cpu_count', help='Set available CPUs to tune on', required=True, type=int)
    parser.add_argument('--model_name', help='Set model name', required=True, type=str)
    parser.add_argument('--identifier', help='Set identifier', required=True, type=str)
    args = parser.parse_args()

    path = osp.join('temp_data', args.dataset)

    cpu_count = args.cpu_count
    gpu_count = args.gpu_count

    node2vec_model = Tuner().tune(path, cpu_count, gpu_count, args.dataset, args.identifier, args.model_name)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data, test_data, val_data = DataLoader().load_data(args.dataset, path, 'cpu')
    print(f"Node classification score: {node_classification_prediction(node2vec_model, test_data)}")
    print(f"Link prediction score on train: {link_prediction(node2vec_model, train_data, train_data)}")
    print(f"Link prediction score on test: {link_prediction(node2vec_model, train_data, test_data)}")
    print(f"Link prediction score on val: {link_prediction(node2vec_model, train_data, val_data)}")
