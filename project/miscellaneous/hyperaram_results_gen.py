import json
from ast import literal_eval

import pandas as pd


def gen_shallow_params():
    # 0 is DeepWalk
    # 1 is node2vec

    df = pd.DataFrame(
        columns=['Model', 'Learning Rate', 'Batch Size', 'Context Size', 'Embedding Dimension', 'Walk Length',
                 'Walks Per Node', 'p', 'q', "Link Embedding Operator"])
    rows = pd.read_csv('Link Prediction GRL project - hyperparam shallow.csv')
    models = ["DeepWalk", "node2vec"]
    counter = 0
    for index, row in rows.iterrows():
        model = models[index]
        for col in row:
            new_row_value = list(literal_eval(col[18:].strip().split(', \'link_prediction_op\'')[0] + '}').values())
            new_row_value.insert(0, model)
            new_row_value.append("Hadamard")
            df.loc[counter] = new_row_value
            counter += 1
    df.to_csv('shallow_config.csv', index=False)


def deep_embedding():
    df = pd.DataFrame(
        columns=['Model', 'Dropout', 'Learning Rate', 'Activation Function', 'Hidden Layer 1 size',
                 'Hidden Layer 2 size', "Link Embedding Operator", "Use Batch Norm"])
    rows = pd.read_csv('deep.csv')
    models = ['GCN', 'ID-GCN', 'GraphSAGE', 'ID-GraphSAGE', 'GAT', 'GIN', 'ID-GIN']
    counter = 0
    for index, row in rows.iterrows():
        model = models[index]
        for col in row:
            if col == 'EMPTY':
                counter += 1
                oom_row = [f"OOM" for _ in range(7)]
                oom_row.insert(0, model)
                df.loc[counter] = oom_row
                continue

            first = col[18:].strip().split('\'activation_function\': <class \'torch.nn.modules.activation.')[0]

            activation = \
                col[18:].strip().split('\'activation_function\': <class \'torch.nn.modules.activation.')[1].split(
                    '\'>')[0]

            second = \
                col[18:].strip().split('\'activation_function\': <class \'torch.nn.modules.activation.')[1].split(
                    '\'>')[1]

            final = first + second[1:]
            new_row = list(literal_eval(final).values())
            new_row.insert(0, model)
            new_row.insert(3, activation)
            new_row.insert(5, new_row[4] // 2)
            new_row.insert(-1, "Hadamard")
            df.loc[counter] = new_row
            counter += 1
    df.to_csv('deep_config.csv', index=False)


if __name__ == '__main__':
    # gen_shallow_params()
    deep_embedding()
