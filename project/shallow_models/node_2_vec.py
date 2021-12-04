import os
from typing import Dict

import torch
from ray import tune
from torch import nn
from torch_geometric.nn import Node2Vec

from project.shallow_models.utils import link_prediction


class TrainNode2Vec:
    @staticmethod
    def train_node2vec(config: Dict, gpu_count: int, cpu_count: int, train_data, val_data, test_data, verbose=False):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = Node2Vec(train_data.edge_index, embedding_dim=config['embedding_dim'],
                         walk_length=config['walk_length'],
                         context_size=config['context_size'], walks_per_node=config['walks_per_node'],
                         num_negative_samples=1, p=config['p'], q=config['q'], sparse=True)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=config['lr'])

        operator = config['link_prediction_op']
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
            roc_score = link_prediction(model, train_data, validation_data, operator)
            return roc_score

        loss = acc = 0
        for epoch in range(1, 10001):
            loss = _train()
            acc = _test()
            if verbose:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=loss, accuracy=acc)
        return model
