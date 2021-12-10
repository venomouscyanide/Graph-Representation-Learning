# Ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
import torch
from torch.nn import BatchNorm1d
from torch_geometric.nn import GIN
import torch.nn.functional as F

from project.deep_models.deep_model_factory import ModelConvLoader

"""
Docs:
    self.conv1 = GINConv(GIN.MLP(in_channels, hidden_channels))
    self.conv2 = GINConv(GIN.MLP(hidden_channels, out_channels))
    
    
    self.conv1 = GINIDConvLayer(GIN.MLP(in_channels, hidden_channels), GIN.MLP(in_channels, hidden_channels))
    self.conv2 = GINIDConvLayer(GIN.MLP(hidden_channels, out_channels), GIN.MLP(hidden_channels, out_channels))
"""


class LinkPredModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_name, act_func, norm, p):
        super().__init__()
        self.id_version = True if model_name[:2] == 'id' else False
        self.act_func = act_func
        self.norm = norm
        self.p = p
        conv_layer = ModelConvLoader().get(model_name)
        if model_name == 'id_gin':
            self.conv1 = conv_layer(GIN.MLP(in_channels, hidden_channels), GIN.MLP(in_channels, hidden_channels))
            self.conv2 = conv_layer(GIN.MLP(hidden_channels, out_channels), GIN.MLP(hidden_channels, out_channels))
        elif model_name == 'gin':
            self.conv1 = conv_layer(GIN.MLP(in_channels, hidden_channels))
            self.conv2 = conv_layer(GIN.MLP(hidden_channels, out_channels))
        else:
            self.conv1 = conv_layer(in_channels, hidden_channels)
            self.conv2 = conv_layer(hidden_channels, out_channels)
        if norm:
            self.batch_norm = BatchNorm1d(hidden_channels)

    def encode(self, x, edge_index, move_to_cuda=False):
        device = 'cpu'
        if move_to_cuda:
            device = "cuda"
        node_id_index = torch.tensor([_ for _ in range(x.size(0))], device=device)

        if self.id_version:
            x = self.conv1(x, edge_index, id=node_id_index)
        else:
            x = self.conv1(x, edge_index).relu()

        if self.norm:
            x = self.batch_norm(x)

        x = self.act_func()(x)
        x = F.dropout(x, p=self.p)

        if self.id_version:
            x = self.conv2(x, edge_index, id=node_id_index)
        else:
            x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
