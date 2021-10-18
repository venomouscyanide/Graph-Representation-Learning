import copy

import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch.nn import BatchNorm2d, LogSoftmax, BatchNorm1d
from torch_geometric import nn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv

from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import global_add_pool, global_mean_pool
from tqdm import tqdm


def get_num_classes(pyg_dataset):
    return pyg_dataset.num_classes


def get_num_features(pyg_dataset):
    return pyg_dataset.num_features


def get_graph_class(pyg_dataset, idx):
    # TODO: Implement a function that takes a PyG dataset object,
    # an index of a graph within the dataset, and returns the class/label
    # of the graph (as an integer).

    label = -1

    ############# Your code here ############

    label = int(pyg_dataset[idx].y)

    #########################################

    return label


def get_graph_num_edges(pyg_dataset, idx):
    # TODO: Implement a function that takes a PyG dataset object,
    # the index of a graph in the dataset, and returns the number of
    # edges in the graph (as an integer). You should not count an edge
    # twice if the graph is undirected. For example, in an undirected
    # graph G, if two nodes v and u are connected by an edge, this edge
    # should only be counted once.

    num_edges = 0

    ############# Your code here ############
    seen_edges = set()
    nx_graph = to_networkx(pyg_dataset[idx])
    for edge in nx_graph.edges:
        # sort the edge index so that edges like (0, 21) and (21, 0) are only counted once.
        edge = tuple(sorted([*edge]))
        if edge not in seen_edges:
            seen_edges.add(edge)
    num_edges = len(seen_edges)
    assert len(nx_graph.to_undirected().edges) == num_edges
    #########################################

    return num_edges


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        # TODO: Implement a function that initializes self.convs,
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        y = hidden_dim
        for layer in range(num_layers - 1):
            if layer == 0:
                x = input_dim
            else:
                x = y
            self.convs.append(nn.GCNConv(x, y))
            self.bns.append(torch.nn.BatchNorm1d(y))
        self.last_conv = GCNConv(hidden_dim, output_dim)

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        # TODO: Implement a function that takes the feature tensor x and
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.

        out = None

        ############# Your code here ############

        for idx in range(len(self.convs)):
            x = self.convs[idx](x, adj_t)
            x = self.bns[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        if not self.return_embeds:
            x = self.last_conv(x, adj_t)
            x = self.softmax(x)

        out = x
        #########################################

        return out


def train_(model, data, train_idx, optimizer, loss_fn):
    # TODO: Implement a function that trains the model by
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    ############# Your code here ############
    optimizer.zero_grad()
    y = model(data.x, data.adj_t)
    loss = loss_fn(y[train_idx], data.y[train_idx].T[0])
    #########################################

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, save_model_results=False):
    # TODO: Implement a function that tests the model by
    # using the given split_idx and evaluator.
    model.eval()

    # The output of model on all data
    out = None

    ############# Your code here ############
    ## (~1 line of code)
    ## Note:
    ## 1. No index slicing here
    out = model(data.x, data.adj_t)
    #########################################

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    if save_model_results:
        print("Saving Model Predictions")

        data = {}
        data['y_pred'] = y_pred.view(-1).cpu().detach().numpy()

        df = pd.DataFrame(data=data)
        # Save locally as csv
        df.to_csv('ogbn-arxiv_node.csv', sep=',', index=False)

    return train_acc, valid_acc, test_acc


class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Load encoders for Atoms in molecule graphs
        self.node_encoder = AtomEncoder(hidden_dim)

        # Node embedding model
        # Note that the input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(hidden_dim, hidden_dim,
                            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = None

        ############# Your code here ############
        self.pool = global_mean_pool
        #########################################

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, batched_data):
        # TODO: Implement a function that takes as input a
        # mini-batch of graphs (torch_geometric.data.Batch) and
        # returns the predicted graph property for each graph.
        #
        # NOTE: Since we are predicting graph level properties,
        # your output will be a tensor with dimension equaling
        # the number of graphs in the mini-batch

        # Extract important attributes of our mini-batch
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)

        out = None

        ############# Your code here ############
        y = self.gnn_node(embed, edge_index)
        pooled_batch = self.pool(y, batch)
        out = self.linear(pooled_batch)
        #########################################

        return out


def train(model, device, data_loader, optimizer, loss_fn):
    # TODO: Implement a function that trains your model by
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y

            ############# Your code here ############
            optimizer.zero_grad()
            y = model(batch)[is_labeled]
            labels = batch.y[is_labeled].to(torch.float32)
            loss = loss_fn(y, labels)
            #########################################

            loss.backward()
            optimizer.step()

    return loss.item()


if __name__ == '__main__':
    # root = './enzymes'
    # name = 'ENZYMES'
    #
    # # The ENZYMES dataset
    # pyg_dataset = TUDataset(root, name)
    # num_classes = get_num_classes(pyg_dataset)
    # num_features = get_num_features(pyg_dataset)
    # print("{} dataset has {} classes".format(name, num_classes))
    # print("{} dataset has {} features".format(name, num_features))

    # graph_0 = pyg_dataset[0]
    # print(graph_0)
    # idx = 100
    # label = get_graph_class(pyg_dataset, idx)
    # print('Graph with index {} has label {}'.format(idx, label))

    # idx = 200
    # num_edges = get_graph_num_edges(pyg_dataset, idx)
    # print('Graph with index {} has {} edges'.format(idx, num_edges))

    # dataset_name = 'ogbn-arxiv'
    # # Load the dataset and transform it to sparse tensor
    # dataset = PygNodePropPredDataset(name=dataset_name,
    #                                  transform=T.ToSparseTensor())
    # print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))
    #
    # # Extract the graph
    # data = dataset[0]
    # print(data.num_features)

    # dataset_name = 'ogbn-arxiv'
    # dataset = PygNodePropPredDataset(name=dataset_name,
    #                                  transform=T.ToSparseTensor())
    # data = dataset[0]
    #
    # # Make the adjacency matrix to symmetric
    # data.adj_t = data.adj_t.to_symmetric()
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # # If you use GPU, the device should be cuda
    # print('Device: {}'.format(device))
    #
    # data = data.to(device)
    # split_idx = dataset.get_idx_split()
    # train_idx = split_idx['train'].to(device)
    #
    # args = {
    #     'device': device,
    #     'num_layers': 3,
    #     'hidden_dim': 256,
    #     'dropout': 0.5,
    #     'lr': 0.01,
    #     'epochs': 100,
    # }
    # model = GCN(data.num_features, args['hidden_dim'],
    #             dataset.num_classes, args['num_layers'],
    #             args['dropout']).to(device)
    # evaluator = Evaluator(name=dataset_name)
    #
    # # PRIMARY DRIVER!
    # model.reset_parameters()
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    # loss_fn = F.nll_loss
    #
    # best_model = None
    # best_valid_acc = 0
    #
    # for epoch in range(1, 1 + args["epochs"]):
    #     loss = train(model, data, train_idx, optimizer, loss_fn)
    #     result = test(model, data, split_idx, evaluator)
    #     train_acc, valid_acc, test_acc = result
    #     if valid_acc > best_valid_acc:
    #         best_valid_acc = valid_acc
    #         best_model = copy.deepcopy(model)
    #     print(f'Epoch: {epoch:02d}, '
    #           f'Loss: {loss:.4f}, '
    #           f'Train: {100 * train_acc:.2f}%, '
    #           f'Valid: {100 * valid_acc:.2f}% '
    #           f'Test: {100 * test_acc:.2f}%')

    # Load the dataset
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    split_idx = dataset.get_idx_split()

    # Check task type
    print('Task type: {}'.format(dataset.task_type))

    args = {
        'device': device,
        'num_layers': 5,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 30,
    }
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=0)

    model = GCN_Graph(args['hidden_dim'],
                      dataset.num_tasks, args['num_layers'],
                      args['dropout']).to(device)
    evaluator = Evaluator(name='ogbg-molhiv')

    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + args["epochs"]):
        print('Training...')
        model.pool = global_add_pool
        loss = train(model, device, train_loader, optimizer, loss_fn)

        print('Evaluating...')
        train_result = eval(model, device, train_loader, evaluator)
        val_result = eval(model, device, valid_loader, evaluator)
        test_result = eval(model, device, test_loader, evaluator)

        train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], val_result[dataset.eval_metric], \
                                         test_result[dataset.eval_metric]
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')
