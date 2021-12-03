from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def link_prediction(model, train_data, test_data):
    model = model()

    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=1500)
    link_pred_pipeline = Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

    link_features_train_128 = model[train_data.edge_label_index[0]] * model[train_data.edge_label_index[1]]
    link_features_test_128 = model[test_data.edge_label_index[0]] * model[test_data.edge_label_index[1]]

    link_pred_pipeline.fit(link_features_train_128.detach().cpu().numpy(), train_data.edge_label.detach().cpu().numpy())
    final_prediction = link_pred_pipeline.predict_proba(link_features_test_128.detach().cpu().numpy())

    positive_column = list(link_pred_pipeline.classes_).index(1)
    roc_scores = roc_auc_score(test_data.edge_label.detach().cpu().numpy(), final_prediction[:, positive_column])
    return roc_scores


def node_classification_prediction(model, data):
    model = model()
    clf = LogisticRegression(). \
        fit(
        model[data.train_mask].detach().cpu().numpy(),
        data.y[data.train_mask].detach().cpu().numpy()
    )
    return clf.score(model[data.test_mask].detach().cpu().numpy(),
                     data.y[data.test_mask].detach().cpu().numpy())