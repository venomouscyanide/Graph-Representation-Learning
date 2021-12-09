from graphgym.contrib.layer.idconv import GINIDConvLayer, SAGEIDConvLayer, GCNIDConv
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv


class ModelConvLoader:
    # GATIDConvLayer not working. Report this?
    @staticmethod
    def get(model_name):
        if model_name == 'gcn':
            return GCNConv
        if model_name == 'graph_sage':
            return SAGEConv
        if model_name == 'gat':
            return GATConv
        if model_name == 'gin':
            return GINConv
        if model_name == 'id_gcn':
            return GCNIDConv
        if model_name == 'id_sage':
            return SAGEIDConvLayer
        if model_name == 'id_gin':
            return GINIDConvLayer
        else:
            raise NotImplementedError(f' Message Passing Layer for model{model_name} not supported.')
