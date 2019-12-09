__version__ = '2.1.1'
__author__ = 'Calum Hand'

import numpy as np
from sklearn.cluster import KMeans


class KlusterFoldCrossValidation(object):
    """
    Allows for higher harsher assessment of supervised learning methods by designating train and test sets through
    clustering of the original feature matrix.

    The initial dataset clustering is performed on initialising the object where the data is segmented into `k` clusters
    where `k` is the denominator to the typical `test_size` fraction.
    Once segmented, the training data is used in model evaluation by performing `n` fold cross validation where `n` is
    the number of folds / clusters to split the training data into.
    Each iteration of cross validation removes one data cluster from the training set rather than a uniformly sampled
    random subset of data as with traditional cross validation.
    """
    def __init__(self, X, y, test_div=5, model=KMeans):
        """
        :param X: np.ndarray(), Feature matrix with `m` entries (rows) and `n` features (columns)
        :param y: np.ndarray(), Target array with `m` entries
        :param test_div: int, Fraction denominator used for splitting train and test data (5 ~ test_size=0.2)
        :param model: object, Clustering model to segment data, must have `fit()` method and `.labels_` attribute to
        determine and export the integer labels of the clustered feature data `X`
        """
        self.X = X
        self.y = y
        self.k_clusters = test_div
        self.model = model
        self.k_assignments, self.k_labels = self._cluster_data(self.X, self.k_clusters)
        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)

    def _cluster_data(self, X, k):
        """
        Clusters the passed feature matrix wih the clustering method specified at class initialisation.
        The cluster assignments for all passed data points in `X` are returned along with an array of cluster labels.
        The use of `k_labels` here allows for quicker isolation of training and test data sets as only parsing a small
        array rather than the entire length of feature matrix `X`

        :param X: np.ndarray(), Feature matrix with `m` entries (rows) and `n` features (columns)
        :param k: int, The number of clusters to segment the passed data into

        :return: (k_assignments, k_labels): (np.ndarray(), np.ndarray()), (Cluster labels of all entries, Label array)
        """
        mdl = self.model(n_clusters=k)
        mdl.fit(X)
        k_assignments = mdl.labels_
        k_labels = np.unique(k_assignments)
        return k_assignments, k_labels

    @staticmethod
    def _tt_split(X, y, label, data_assignments, data_labels):
        """
        Performs the splitting of the data into training and testing sections based on the clustering conducted when
        the object is initialised.
        A single label corresponding to a cluster is selected to act as the test set and the data is indexed based on
        these labels to resolve the training and test sets for `X` and `y` as needed.

        :param X: np.ndarray(), Feature matrix with `m` entries (rows) and `n` features (columns)
        :param y: np.ndarray(), Target array with `m` entries
        :param label: int, Label of the cluster to use as the test set
        :param data_assignments: np.ndarray(), Cluster labels of all `m` entries
        :param data_labels: np.ndarray(), Listing of unique cluster labels

        :return: (X_train, X_test, y_train, y_test), segmented train and test data feature matrices and target arrays
        """
        data_train = data_labels[data_labels != label]
        train, test = np.isin(data_assignments, data_train), np.isin(data_assignments, data_train, invert=True)
        return X[train], X[test], y[train], y[test]

    def train_test_split(self, test_label=0):
        """
        Separates the `X` and `y` data used to initialise the class into a training and test set based on the cluster
        labelling of the feature matrix `X`.
        The cluster labels are integer values from [0, `test_div`) and any one can be passed by the user if a specific
        test set is desired, else the first cluster of data is selected by default.
        The internal object attributes are updated before the values are returned.

        :param test_label: int, The label of the cluster to be set as the test data from the passed data sets

        :return: (X_train, X_test, y_train, y_test), Segmented train and test data feature matrices and target arrays
        """
        assert test_label in self.k_labels, 'test label specified not present in cluster labels'
        tt_split_data = self._tt_split(self.X, self.y, test_label, self.k_assignments, self.k_labels)
        self.X_train, self.X_test, self.y_train, self.y_test = tt_split_data
        return tt_split_data


    @staticmethod
    def _assess_model(X_train, X_test, y_train, y_test, estimator, metric):
        """
        Using the scikit-learn convention, fits a passed model on the training data; generates a prediction from the
        test feature data and then compares it against the test target values using the passed metric.

        :param X_train: np.ndarray(), Feature matrix used for training the passed estimator
        :param X_test: np.ndarray(), Feature matrix used to test the passed estimator
        :param y_train: np.ndarray(), Target array used for training the passed estimator
        :param y_test: np.ndarray(), Target array that the estimator predicted values are compared against
        :param estimator: object, The ML estimator to be assessed, must have `.fit()` and `.predict()` methods
        :param metric: function, Assesses passed estimator, must only compare predicted and test target values directly

        :return: score: float, The numerical score of the estimator as per the passed `metric` function
        """
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = metric(y_pred, y_test)
        return score

    def cross_cluster_validate(self, cv, estimator, metric):
        """
        Performs a cluster fold validation of the training data.
        The number of folds, `cv`, is taken as the number of clusters to segment the training data into.
        For each cluster of data in the segmented data set, a model is trained using the remaining train data minus the
        withheld cluster.
        The prediction score of the model against the training data and cross validation data is then returned.

        :param cv: int,  The number of folds / clusters to segment the passed data into
        :param estimator: object, The ML estimator to be assessed, must have `.fit()` and `.predict()` methods
        :param metric: function, Assesses passed estimator, must only compare predicted and test target values directly

        :return: dict, Model train and cross validation scores stored as lists in key value pairs with score name
        """
        cross_val_scores = {'train': [], 'cv': []}
        cv_assig, cv_labels = self._cluster_data(self.X_train, cv)
        for label in cv_labels:
            X_tr, X_cv, y_tr, y_cv = self._tt_split(self.X_train, self.y_train, label, cv_assig, cv_labels)
            tr_score = self._assess_model(X_tr, X_tr, y_tr, y_tr, estimator, metric)
            cv_score = self._assess_model(X_tr, X_cv, y_tr, y_cv, estimator, metric)
            cross_val_scores['train'].append(tr_score)
            cross_val_scores['cv'].append(cv_score)
        return cross_val_scores

    def estimator_test(self, estimator, metric):
        """
        Allows for the passed estimator to be trained on the full training data set and then tested against the test set
        originally segmented by the clustering process.

        :param estimator: object, The ML estimator to be assessed, must have `.fit()` and `.predict()` methods
        :param metric: function, Assesses passed estimator, must only compare predicted and test target values directly

        :return: model_score, float, The numerical score of the estimator as per the passed `metric` function
        """
        model_score = self._assess_model(self.X_train, self.X_test, self.y_train, self.y_test, estimator, metric)
        return model_score
