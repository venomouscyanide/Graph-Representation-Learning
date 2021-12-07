from torch_geometric.datasets import Planetoid, KarateClub, Amazon, Coauthor, SNAPDataset


class DatasetLoaderFactory:
    @staticmethod
    def get(dataset: str, path: str, transform):
        dataset = dataset.lower()
        if dataset in ['cora', 'citeseer', 'pubmed']:
            # Planetoid
            return Planetoid(path, name=dataset, transform=transform)
        if dataset == 'karate':
            # classic toy example
            return KarateClub(transform=transform)
        if dataset in ['computers', 'photo']:
            # pitfalls paper
            return Amazon(path, name=dataset, transform=transform)
        if dataset in ['cs', 'physics']:
            # pitfalls paper
            return Coauthor(path, name=dataset, transform=transform)
        if dataset == 'ego-facebook':
            # snap ego-facebook dataset
            return SNAPDataset(path, name=dataset, transform=transform)
        else:
            raise NotImplementedError(f'Module not configured for dataset:{dataset}')
