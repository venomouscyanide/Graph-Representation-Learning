from torch_geometric.nn import Node2Vec


class ModelTrainFactory:
    @staticmethod
    def get(model_name: str):
        if model_name in ['node2vec', 'deepwalk']:
            from project.shallow_models.node_2_vec import TrainNode2Vec  # avoid cyclic import
            return TrainNode2Vec().train_node2vec
        else:
            raise NotImplementedError(f'Training on shallow model: {model_name} not supported.')


class ModelFactory:
    @staticmethod
    def get(model_name: str):
        if model_name in ['node2vec', 'deepwalk']:
            return Node2Vec
        else:
            raise NotImplementedError(f'Shallow model: {model_name} not supported.')
