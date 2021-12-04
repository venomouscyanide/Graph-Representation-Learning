# ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
# ray tune: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
"""
    Usage : python3.9 -m project.shallow_models.model_trainer --gpu_count 0
                                                              --cpu_count 4
                                                              --dataset "cora"
                                                              --identifier "cora_run"
                                                              --model_name "node2vec"

"""
from torch_geometric import seed_everything

seed_everything(42)  # 42 is the answer to all

import math
import os
import os.path as osp

import argparse
import shutil

import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch_geometric.transforms as T

import warnings

from project.dataset_loader_factory import DatasetLoaderFactory
from project.shallow_models.model_factory import ModelFactory, ModelTrainFactory
from project.shallow_models.utils import link_prediction, node_classification_prediction, LinkOperators

warnings.simplefilter(action="ignore")


class HyperParameterTuning:
    CONFIG = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([int(math.pow(2, n)) for n in range(1, 10)]),
        "context_size": tune.choice([5, 10]),
        "embedding_dim": tune.choice([64, 128, 256]),
        "walk_length": tune.choice([10, 15, 20]),
        "walks_per_node": tune.choice([10, 20]),
        "p": tune.choice([1 * n for n in range(1, 5)]),
        "q": tune.choice([1 * n for n in range(1, 5)]),
        "link_prediction_op": tune.choice(
            [LinkOperators.hadamard, LinkOperators.average_u_v, LinkOperators.l1_dist, LinkOperators.l2_distance]
        )
    }

    RAYTUNE_CONFIG = {
        'num_samples': 1,
        'max_epochs': 2
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
    def tune(self, data_dir: str, cpu_count: int, gpu_count: int, dataset: str, identifier: str, model_name: str,
             train_data, val_data, test_data):
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=HyperParameterTuning.RAYTUNE_CONFIG['max_epochs'],
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
        trainer_function = ModelTrainFactory().get(model_name)
        result = tune.run(
            tune.with_parameters(trainer_function, gpu_count=gpu_count, cpu_count=cpu_count, train_data=train_data,
                                 test_data=test_data, val_data=val_data),
            resources_per_trial={"cpu": cpu_count, "gpu": gpu_count},
            config=HyperParameterTuning.CONFIG,
            num_samples=HyperParameterTuning.RAYTUNE_CONFIG['num_samples'],
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=os.path.join(identifier, "ray_results"),
            log_to_file=True,
            resume="AUTO"
        )

        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final train loss: {}".format(best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        device = "cuda:0" if (torch.cuda.is_available() and gpu_count) else "cpu"
        train_data, val_data, test_data = DataLoader().load_data(dataset, data_dir, device)
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)
        model = ModelFactory().get(model_name)
        best_trained_model = model(train_data.edge_index, embedding_dim=best_trial.config['embedding_dim'],
                                   walk_length=best_trial.config['walk_length'],
                                   context_size=best_trial.config['context_size'],
                                   walks_per_node=best_trial.config['walks_per_node'],
                                   num_negative_samples=1, p=best_trial.config['p'], q=best_trial.config['q'],
                                   sparse=True)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))

        if (torch.cuda.is_available() and gpu_count):
            model_state['embedding.weight'] = model_state.pop('module.embedding.weight')

        best_trained_model.load_state_dict(model_state)

        best_op = best_trial.config['link_prediction_op']
        validation_acc = link_prediction(best_trained_model, train_data, val_data, best_op)
        print("Best trial val set accuracy: {}".format(validation_acc))
        torch.save(best_trained_model, os.path.join(identifier, f'{identifier}_best_model.model'))
        return best_trained_model, best_op


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tuning for n2v/dw/<x> shallow encoders", )
    parser.add_argument('--dataset', help='Choose the dataset to tune on', required=True, type=str)
    parser.add_argument('--gpu_count', help='Set available GPUs to tune on', required=True, type=int)
    parser.add_argument('--cpu_count', help='Set available CPUs to tune on', required=True, type=int)
    parser.add_argument('--model_name', help='Set model name', required=True, type=str)
    parser.add_argument('--norm_features', help='Norm node features', action='store_true')
    parser.add_argument('--identifier', help='Set identifier', required=True, type=str)
    args = parser.parse_args()

    path = osp.join('temp_data', args.dataset)

    # if os.path.exists(args.identifier) and os.path.isdir(args.identifier):
    #     shutil.rmtree(args.identifier)

    cpu_count = args.cpu_count
    gpu_count = args.gpu_count

    HyperParameterTuning.NORMALIZE_FEATURES = args.norm_features

    device = "cuda:0" if (torch.cuda.is_available() and gpu_count) else "cpu"
    train_data, test_data, val_data = DataLoader().load_data(args.dataset, path, device)
    node2vec_model, best_op = Tuner().tune(path, cpu_count, gpu_count, args.dataset, args.identifier, args.model_name,
                                           train_data, val_data, test_data)

    print(f"Node classification score on test split: {node_classification_prediction(node2vec_model, test_data)}")
    print(f"Link prediction score on train: {link_prediction(node2vec_model, train_data, train_data, best_op)}")
    print(f"Link prediction score on test: {link_prediction(node2vec_model, train_data, test_data, best_op)}")
    print(f"Link prediction score on val: {link_prediction(node2vec_model, train_data, val_data, best_op)}")
