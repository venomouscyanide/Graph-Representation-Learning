from torch_geometric.datasets import Planetoid


class DatasetLoaderFactory:
    @staticmethod
    def get(dataset: str):
        dataset = dataset.lower()
        if dataset.lower() in ['cora', 'citeseer', 'pubmed']:
            return Planetoid
        else:
            raise NotImplementedError(f'Module not configured for dataset:{dataset}')
