import argparse
import os
from collections import defaultdict

import pandas as pd
import torch

from project.dataset_loader_factory import DatasetLoaderFactory
from project.statistics.reconstruction_utils import prep_dataset, TypeOfModel
from project.statistics.stats_capture import GraphReconstructionMMD


class ShallowConfig:
    """
    Ordering is key here
    """
    models = ['mlp', 'node_2_vec', 'dw']
    datasets = ['karate', 'cora', 'citeseer', 'pubmed', 'computers', 'photo']


class GenTable:
    def gen_shallow_table(self, base_folder, output_folder):
        df = self._gen_table()
        files = self._get_files(base_folder)
        file_map = self._create_file_map(files)

        for index, model_name in enumerate(ShallowConfig.models):
            row = [model_name.title()]
            for dataset in ShallowConfig.datasets:
                corres_file = file_map[model_name][dataset]

                data_sans_deg = DatasetLoaderFactory().get(dataset, 'temp_delete_me', None)[0]
                prep_dataset(data_sans_deg)

                print(f"Running on {model_name}_{dataset}")
                model = torch.load(os.path.join(base_folder, corres_file), map_location=torch.device('cpu'))
                model_type = TypeOfModel.MLP if model_name == 'mlp' else TypeOfModel.SHALLOW
                mmd = GraphReconstructionMMD().model_graph_reconstruction(data_sans_deg, model, dataset, model_type,
                                                                          cv=1)

                row.append(mmd)
            df.loc[index] = row

        self._write_to_file(df, output_folder)

    def _gen_table(self):
        header = ['Model'] + [dataset.title() for dataset in ShallowConfig.datasets]
        df = pd.DataFrame(columns=header)
        return df

    def _get_files(self, base_folder):
        # https://stackoverflow.com/a/3207973
        files = [f for f in os.listdir(base_folder) if os.path.isfile(os.path.join(base_folder, f))]
        return files

    def _create_file_map(self, files):
        file_map = defaultdict(dict)

        for file in files:
            cleaned_file_name = file.replace('_no_norm', '')
            split_name = cleaned_file_name.split('_no_degree_information_')
            model = split_name[0]
            dataset = split_name[-1].split('_best_model.model')[0]
            file_map[model][dataset] = file
        return file_map

    def _write_to_file(self, df, output_folder):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        df.to_csv(os.path.join(output_folder, 'shallow_stats.csv'), index=False)
        df.to_latex(os.path.join(output_folder, 'shallow_stats.tex'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run table gen for n2v/dw shallow encoders")
    parser.add_argument('--base_folder', help='Input Folder', required=True, type=str)
    parser.add_argument('--output_folder', help='Output Folder', required=True, type=str)
    args = parser.parse_args()

    GenTable().gen_shallow_table(args.base_folder, args.output_folder)
