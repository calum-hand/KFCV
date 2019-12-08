__version__ = '1.0.0'
__author__ = 'Calum Hand'

import numpy as np
from sklearn.cluster import KMeans


class ClusterFoldValidation(object):
    """
    DocString
    """
    def __init__(self, X, y, test_div=5, model=KMeans):
        self.X = X
        self.y = y
        self.k_clusters = test_div
        self.model = model
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

    @staticmethod
    def _assess_model(X_train, X_test, y_train, y_test, estimator, metric):
        """

        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :param estimator:
        :param metric:
        :return:
        """
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = metric(y_pred, y_test)
        return score

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
            tr_score = self._assess_model(X_tr, X_tr, y_tr, y_tr, estimator, metric)
            cv_score = self._assess_model(X_tr, X_cv, y_tr, y_cv, estimator, metric)
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
        model_score = self._assess_model(self.X_train, self.X_test, self.y_train, self.y_test, estimator, metric)
        return model_score
