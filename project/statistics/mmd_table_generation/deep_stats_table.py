import argparse
import os
from collections import defaultdict

import pandas as pd
import torch
import torch_geometric.transforms as T
from project.dataset_loader_factory import DatasetLoaderFactory
from project.statistics.reconstruction_utils import prep_dataset, TypeOfModel
from project.statistics.stats_capture import GraphReconstructionMMD
from project.utils import MaxDegreeMapping


class DeepConfig:
    """
    Ordering is key here
    """
    models = ['mlp', 'gcn', 'id_gcn', 'graph_sage', 'id_graph_sage', 'gat', 'gin', 'id_gin']
    # 00 10 01 11
    types = ['no_norm_no_degree_information', 'norm_no_degree_information', 'no_norm_degree_information',
             'norm_degree_information']
    datasets = ['karate', 'cora', 'citeseer', 'pubmed', 'computers', 'photo']


class DeepTableGen:
    def generate(self, base_folder, output_folder, mmd_type):
        """
        TODO:: Refactor a lot of reuse of Shallow table gen here
        """
        df = self._gen_table()
        files = self._get_files(base_folder)
        file_map = self._create_file_map(files)

        for index, model_name in enumerate(DeepConfig.models):
            row = [model_name.title()]
            for dataset in DeepConfig.datasets:
                for type_info in DeepConfig.types:
                    prepped_dataset = self._get_dataset(type_info, dataset)
                    print(f"Running on {model_name}_{type_info}_{dataset}")
                    corres_file = file_map[model_name].get(type_info).get(dataset)
                    if not corres_file:
                        mmd = "OOM"
                    else:
                        type_of_model = TypeOfModel.MLP if model_name == 'mlp' else TypeOfModel.DEEP
                        model = torch.load(os.path.join(base_folder, corres_file), map_location=torch.device('cpu'))
                        mmd = GraphReconstructionMMD().model_graph_reconstruction(prepped_dataset, model, dataset,
                                                                                  type_of_model, mmd_type, cv=1)
                    row.append(mmd)
            df.loc[index] = row
        self._write_to_file(df, output_folder)

    def _get_files(self, base_folder):
        # https://stackoverflow.com/a/3207973
        files = [f for f in os.listdir(base_folder) if os.path.isfile(os.path.join(base_folder, f))]
        return files

    def _gen_table(self):
        header = ['Model'] + [f"{dataset.title()}_{type_info}" for dataset in DeepConfig.datasets for type_info in
                              DeepConfig.types]
        df = pd.DataFrame(columns=header)
        return df

    def _create_file_map(self, files):
        """
        parse from file name the type of model, the dataset and the config used.
        """
        file_map = defaultdict(lambda: defaultdict(dict))
        for file in files:
            original_file_name = file
            type_info_norm = 'no_norm' if len(file.split('no_norm')) > 1 else 'norm'
            type_info_deg = 'no_degree_information' if len(
                file.split('no_degree_information')) > 1 else 'degree_information'
            type_info = f"{type_info_norm}_{type_info_deg}"
            model_name = file.split(f'_{type_info_norm}')[0]
            dataset = file.split(f"{type_info}_")[-1].split('_best_model.model')[0]
            if dataset == 'citeceer':
                # Spelt it wrong :'(
                dataset = 'citeseer'
            file_map[model_name][type_info][dataset] = original_file_name
        return file_map

    def _get_dataset(self, type_info, dataset):
        if len(type_info.split("no_degree_information")) > 1:
            data = DatasetLoaderFactory().get(dataset, 'temp_delete_me', None)[0]
        else:
            data = DatasetLoaderFactory().get(dataset, 'delete_me',
                                              T.OneHotDegree(
                                                  max_degree=MaxDegreeMapping.MAPPING[dataset])
                                              )[0]
        prep_dataset(data)
        return data

    def _write_to_file(self, df, output_folder):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        df.to_csv(os.path.join(output_folder, 'shallow_stats.csv'), index=False)
        df.to_latex(os.path.join(output_folder, 'shallow_stats.tex'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run table gen for n2v/dw shallow encoders")
    parser.add_argument('--base_folder', help='Input Folder', required=True, type=str)
    parser.add_argument('--output_folder', help='Output Folder', required=True, type=str)
    parser.add_argument('--mmd_type', help='MMD type', required=True, type=str)
    args = parser.parse_args()

    DeepTableGen().generate(args.base_folder, args.output_folder, args.mmd_type)
