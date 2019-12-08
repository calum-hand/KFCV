__version__ = '1.0.0'
__author__ = 'Calum Hand'

import numpy as np
from sklearn.cluster import KMeans


class ClusterFoldValidation(object):
    """
    DocString
    """
    def __init__(self, X, y, test_size=0.2, model=KMeans):
        self.X = X
        self.y = y
        self.model = model
        self.k_clusters = int(1/test_size)
        assert 0 < self.k_clusters < len(X), 'test size specified does not allow for suitable clustering'
        self.k_assignments, self.k_labels = self._cluster_data(self.X, self.k_clusters)

        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)

    def _cluster_data(self, X, k):
        """
        DocString

        :param X: np.ndarray(),
        :param k: int,
        :return:
        """
        mdl = self.model(n_clusters=k)
        mdl.fit(X)
        k_assignments = mdl.labels_
        k_labels = np.unique(k_assignments)
        return k_assignments, k_labels

    def train_test_split(self, test_label=0):
        """
        DocString

        :param test_label: int,
        :return:
        """
        assert test_label in self.k_labels, 'test label specified not present in cluster labels'
        k_train = self.k_labels[self.k_labels != test_label]
        train, test = self.k_assignments == k_train, self.k_assignments != k_train
        tt_split_data = self.X[train], self.X[test], self.y[train], self.y[test]
        self.X_train, self.X_test, self.y_train, self.y_test = tt_split_data
        return tt_split_data

    @staticmethod
    def _cross_val_tt_split(X, y, label, cv_assignments, cv_labels):
        """
        DocString

        :param X: np.ndarray(),
        :param y: np.ndarray(),
        :param label: int
        :param cv_assignments: np.ndarray(),
        :param cv_labels: np.ndarray(),
        :return:
        """
        cv_train = cv_labels[cv_labels != label]
        train, test = cv_assignments == cv_train, cv_assignments != cv_train
        return X[train], X[test], y[train], y[test]

    def cross_cluster_validate(self, cv, estimator, metric):
        """
        DocString

        :param cv: int,
        :param estimator: object,
        :param metric: function,
        :return:
        """
        cross_val_scores = {'train': [], 'cv': []}
        cv_assig, cv_labels = self._cluster_data(self.X_train, cv)
        for label in cv_labels:
            X_tr, X_cv, y_tr, y_cv = self._cross_val_tt_split(self.X_train, self.y_train, label, cv_assig, cv_labels)
            estimator.fit(X_tr, y_tr)
            y_pred_tr, y_pred_cv = estimator.predict(X_tr), estimator.predict(X_cv)
            tr_score = metric(y_pred_tr, y_tr)
            cv_score = metric(y_pred_cv, y_cv)
            cross_val_scores['train'].append(tr_score)
            cross_val_scores['cv'].append(cv_score)
        return cross_val_scores

    def estimator_test(self, estimator, metric):
        """
        DocString

        :param estimator: object,
        :param metric: function,
        :return:
        """
        estimator.fit(self.X_train, self.y_train)
        y_pred = estimator.predict(self.X_test)
        model_score = metric(y_pred, self.y_test)
        return model_score
