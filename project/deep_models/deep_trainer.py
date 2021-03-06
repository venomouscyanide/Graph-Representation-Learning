# Ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
"""
Usage:
    python3.9 -m project.deep_models.deep_trainer --dataset "karate"
                                                  --gpu_count 0
                                                  --cpu_count 8
                                                  --model_name "gcn"
                                                  --identifier "karate_gcn"
                                                  --degree_information
                                                  --use_norm
"""
import copy
import os
import warnings

warnings.filterwarnings(action="ignore")

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch import nn
from torch_geometric import seed_everything

from project.deep_models.link_pred_model import LinkPredModel
from project.tune_stopper import TimeStopper
from project.utils import CustomDataLoader

seed_everything(42)  # 42 is the answer to all

import argparse
import torch
from ray import tune
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling


class HyperParameterTuning:
    CONFIG = {
        "p": tune.choice([0.4, 0.5, 0.6, 0.7]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "activation_function": tune.choice([torch.nn.ReLU, torch.nn.ELU, torch.nn.Tanh]),
        "in_out_channel_tuple": tune.choice([256, 128, 64]),
        "use_norm": False
    }

    RAYTUNE_CONFIG = {
        'num_samples': 50,
        'max_epochs': 100
    }

    DATASET_SPLIT_CONFIG = {
        "num_val": 0.05,
        "num_test": 0.1
    }

    FEATURE_NORM = True
    AUGMENT_DEGREE_INFORMATION = False


class TrainDeepNets:
    @staticmethod
    def train(model, optimizer, train_data, criterion, gpu_count):
        model.train()
        optimizer.zero_grad()
        device = "cuda" if (torch.cuda.is_available() and gpu_count) else "cpu"

        if torch.cuda.is_available() and gpu_count:
            z = model.module.encode(train_data.x, train_data.edge_index, move_to_cuda=True)
        else:
            z = model.encode(train_data.x, train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='dense')
        neg_edge_index.to(device)

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label_index.to(device)

        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        edge_label.to(device)

        if torch.cuda.is_available() and gpu_count:
            out = model.module.decode(z, edge_label_index).view(-1)
        else:
            out = model.decode(z, edge_label_index).view(-1)

        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss

    @staticmethod
    @torch.no_grad()
    def test(data, model, gpu_count):
        model.eval()
        if torch.cuda.is_available() and gpu_count:
            z = model.module.encode(data.x, data.edge_index, move_to_cuda=True)
        else:
            z = model.encode(data.x, data.edge_index)
        if torch.cuda.is_available() and gpu_count:
            out = model.module.decode(z, data.edge_label_index).view(-1).sigmoid()
        else:
            out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    @staticmethod
    def train_helper(config, gpu_count, train_data, val_data, test_data, model_name, checkpoint_dir=None,
                     verbose=False):
        hidden_param = config['in_out_channel_tuple']
        out_param = int(hidden_param / 2)
        device = "cuda:0" if (torch.cuda.is_available() and gpu_count) else "cpu"

        model = LinkPredModel(train_data.num_features, hidden_param, out_param, model_name,
                              act_func=config['activation_function'],
                              norm=config['use_norm'], p=config['p']).to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'])
        criterion = torch.nn.BCEWithLogitsLoss()

        if torch.cuda.is_available() and gpu_count:
            model = nn.DataParallel(model)
        model.to(device)

        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        best_val_auc = final_test_auc = 0
        for epoch in range(1, 10001):
            loss = TrainDeepNets.train(model=model, optimizer=optimizer, criterion=criterion, train_data=train_data,
                                       gpu_count=gpu_count)
            val_auc = TrainDeepNets.test(val_data, model, gpu_count)
            test_auc = TrainDeepNets.test(test_data, model, gpu_count)
            if val_auc > best_val_auc:
                best_val = val_auc
                final_test_auc = test_auc

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=loss.item(), accuracy=val_auc, test_acc=test_auc)
            if verbose:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')

        return best_val, final_test_auc


class TuneHelper:
    def tune(self, dataset, gpu_count, cpu_count, model_name, identifier):
        path = f'temp/{identifier}'

        device = "cuda" if (torch.cuda.is_available() and gpu_count) else "cpu"
        train_data, val_data, test_data = CustomDataLoader().load_data(dataset, path, device,
                                                                       HyperParameterTuning.FEATURE_NORM,
                                                                       HyperParameterTuning.AUGMENT_DEGREE_INFORMATION,
                                                                       num_val=HyperParameterTuning.DATASET_SPLIT_CONFIG[
                                                                     'num_val'],
                                                                       num_test=HyperParameterTuning.DATASET_SPLIT_CONFIG[
                                                                     'num_test'],
                                                                       add_negative_during_load=False)

        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=HyperParameterTuning.RAYTUNE_CONFIG['max_epochs'],
            grace_period=1,
            reduction_factor=2)

        # Use me for testing
        # TrainDeepNets.train_helper(
        #     {"in_out_channel_tuple": 256, "activation_function": nn.ELU, "use_norm": False, "p": 0.5, "lr": 0.001},
        #     gpu_count, train_data, val_data, test_data, "id_gin", )
        # comment above line if not needed

        reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
        trainer_function = TrainDeepNets.train_helper
        result = tune.run(
            tune.with_parameters(trainer_function, gpu_count=gpu_count, train_data=train_data,
                                 test_data=test_data, val_data=val_data, model_name=model_name),
            resources_per_trial={"cpu": cpu_count, "gpu": gpu_count},
            config=HyperParameterTuning.CONFIG,
            num_samples=HyperParameterTuning.RAYTUNE_CONFIG['num_samples'],
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=os.path.join(identifier, "ray_results"),
            log_to_file=True,
            stop=TimeStopper(),
            resume="AUTO"
        )

        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final train loss: {}".format(best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        device = 'cpu'
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)

        hidden_param = best_trial.config['in_out_channel_tuple']
        out_param = int(hidden_param / 2)
        best_trained_model = LinkPredModel(train_data.num_features, hidden_param, out_param, model_name,
                                           act_func=best_trial.config['activation_function'],
                                           norm=best_trial.config['use_norm'], p=best_trial.config['p'])
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))

        if torch.cuda.is_available() and gpu_count:
            copy_of_model_state = copy.deepcopy(model_state)
            for key in model_state.keys():
                updated_key = key.split('module.')[-1]
                copy_of_model_state[updated_key] = copy_of_model_state.pop(key)
            model_state = copy_of_model_state

        best_trained_model.load_state_dict(model_state)

        validation_acc = TrainDeepNets.test(val_data, best_trained_model, gpu_count=0)
        test_acc = TrainDeepNets.test(test_data, best_trained_model, gpu_count=0)

        print("Best trial val set accuracy: {}".format(validation_acc))
        print("Best trial test set accuracy: {}".format(test_acc))
        print(f"Identifier info: {identifier}")
        torch.save(best_trained_model, os.path.join(identifier, f'{identifier}_best_model.model'))

        # z = model.encode(test_data.x, test_data.edge_index)
        # final_edge_index = model.decode_all(z)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Tuning for Deep encoders")
    parser.add_argument('--dataset', help='Choose the dataset to tune on', required=True, type=str)
    parser.add_argument('--gpu_count', help='Set available GPUs to tune on', required=True, type=int)
    parser.add_argument('--cpu_count', help='Set available CPUs to tune on', required=True, type=int)
    parser.add_argument('--model_name', help='Set model name', required=True, type=str)
    parser.add_argument('--use_norm', help='Use Norm', action='store_true')
    parser.add_argument('--degree_information', help='Augment node degree information', action='store_true')
    parser.add_argument('--identifier', help='Set identifier', required=True, type=str)

    args = parser.parse_args()

    if args.use_norm:
        HyperParameterTuning.CONFIG['use_norm'] = True

    if args.degree_information:
        HyperParameterTuning.AUGMENT_DEGREE_INFORMATION = True

    TuneHelper().tune(args.dataset, args.gpu_count, args.cpu_count, args.model_name, args.identifier)
