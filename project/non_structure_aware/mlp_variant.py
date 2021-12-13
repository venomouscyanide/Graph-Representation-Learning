from typing import List

from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Identity
import torch
from torch_geometric.utils import negative_sampling

from project.shallow_models.utils import LinkOperators
from project.statistics.reconstruction_utils import link_prediction_cross_validation, TypeOfModel, get_link_embedding
from project.utils import CustomDataLoader


class MLP(torch.nn.Module):
    # taken from pyg PR#3553: https://github.com/pyg-team/pytorch_geometric/pull/3553
    # merged but not released
    r"""A multi-layer perception (MLP) model.
    Args:
        channel_list (List[int]): List of input, intermediate and output
            channels.
            :obj:`len(channel_list) - 1` denotes the number of layers of the
            MLP.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        relu_first (bool, optional): If set to :obj:`True`, ReLU activation is
            applied before batch normalization. (default: :obj:`False`)
    """

    def __init__(self, channel_list: List[int], dropout: float = 0.,
                 batch_norm: bool = True, relu_first: bool = False):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first

        self.lins = torch.nn.ModuleList()
        for dims in zip(channel_list[:-1], channel_list[1:]):
            self.lins.append(Linear(*dims))

        self.norms = torch.nn.ModuleList()
        for dim in zip(channel_list[1:-1]):
            self.norms.append(BatchNorm1d(dim) if batch_norm else Identity())

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        for lin, norm in zip(self.lins[:-1], self.norms):
            x = lin.forward(x)
            if self.relu_first:
                x = x.relu_()
            x = norm(x)
            if not self.relu_first:
                x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'


def train_mlp(train, test_data, batch_norm, device):
    """
    Train for link prediction downstream task
    """
    mlp = MLP(channel_list=[train.x.size(1), 512, train.x.size(1)], dropout=0.50, batch_norm=batch_norm,
              relu_first=batch_norm).to(device)
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    operator = LinkOperators.hadamard

    mlp.train()
    for epoch in range(101):
        # perform new round of neg sampling per epoch
        neg_edge_index = negative_sampling(
            edge_index=train.edge_index, num_nodes=train.num_nodes,
            num_neg_samples=train.edge_label_index.size(1), method='dense')
        neg_edge_index.to(device)

        edge_label_index = torch.cat(
            [train.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label_index.to(device)

        edge_label = torch.cat([
            train.edge_label,
            train.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        edge_label.to(device)

        train.new_edge_label_index = edge_label_index

        link_embedding = get_link_embedding(TypeOfModel.MLP, operator, mlp, train, arg='new_edge_label_index')
        loss = criterion(link_embedding.sum(dim=-1), edge_label)
        if epoch % 5 == 0:
            print(f"Loss for epoch {epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()

    # forward pass on test data
    mlp.eval()
    train.edge_label_index = edge_label_index
    train.edge_label = edge_label
    link_prediction_cross_validation(mlp, train, test_data, 1, TypeOfModel.MLP, operator)
    return mlp


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, _, test_data = CustomDataLoader().load_data('karate', 'temp_delete_me', 'cpu',
                                                            True,
                                                            False,
                                                            num_val=0.05,
                                                            num_test=0.10,
                                                            add_negative_during_load=False)

    train_mlp(train_data, test_data, True, device)
