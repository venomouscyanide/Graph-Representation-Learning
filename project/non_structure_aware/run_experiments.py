import warnings
warnings.filterwarnings('ignore')
from torch_geometric import seed_everything

seed_everything(49)
import argparse

import torch

from project.non_structure_aware.mlp_variant import train_mlp
from project.utils import CustomDataLoader


def run_experiments(dataset, identifier, degree_information, use_norm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, _, test_data = CustomDataLoader().load_data(dataset, 'temp_delete_me', device,
                                                            True,
                                                            degree_information,
                                                            num_val=0.05,
                                                            num_test=0.10,
                                                            add_negative_during_load=False)

    mlp = train_mlp(train_data, test_data, use_norm, device)
    torch.save(mlp, f'{identifier}_best_model.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Link prediction MLP")
    parser.add_argument('--dataset', help='Choose the dataset to train on', required=True, type=str)
    parser.add_argument('--use_norm', help='Use Batch Norm', action='store_true')
    parser.add_argument('--degree_information', help='Augment node degree information', action='store_true')
    parser.add_argument('--identifier', help='Set identifier', required=True, type=str)

    args = parser.parse_args()

    run_experiments(args.dataset, args.identifier, args.degree_information, args.use_norm)
