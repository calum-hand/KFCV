import numpy as np


class KMeans(object):
    """
    Implementation of KMeans clustering algorithm using the Euclidean distance between the centroid points and feature
    vectors present in the training data set. The initial centroid points are the same as `K` randomly members of the
    training set to help reduce poor clustering.
    """
    def __init__(self, num_clusters, max_iterations):
        self.K = num_clusters
        self.max_iters = max_iterations
        self.iters_ran = 0
        self.centroids = None
        self.losses = []
        self.prediction_labels = []

    def _cluster_assignment(self, X, cluster_labels):
        """
        Assigns each entry in X (row) to one of the passed centroid coordinates based on the euclidean distance between
        that point and the centroid.

        :param X: np.array(float64), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        :param cluster_labels: np.array(int), an `m` sized array
        :return: cluster_labels: np.array(int), an array of size `m`
        """
        for i in range(len(cluster_labels)):
            xi = X[i]
            distances = np.linalg.norm(self.centroids - xi, axis=1)  # calc distances
            cluster_labels[i] = distances.argmin()  # get index of smallest distance
        return cluster_labels

    def _move_centroids(self, X, cluster_labels):
        """
        Calculates the positions of the centroids based on the average of all data points belonging to the cluster.
        As potentially no data points can belong to a centroid due to initialisation, a centroid could be removed and so
        only `k-z` centroids would be in use, this alerts the user who may be expecting `k` centroids.

        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        :param cluster_labels: np.array(int), an array of size `m`
        """
        k_labels = np.unique(cluster_labels)  # sorted np array of all cluster labels still active
        centroid_count = len(k_labels)

        if centroid_count != self.K:  # if we have k-z clusters then update number of centroids accordingly
            self.centroids = np.empty(centroid_count)
            print(F'Centroid removed, proceeding with {centroid_count} centroids')

        for i, k in enumerate(k_labels):  # find indices of points in point k
            cluster_constituents = X[cluster_labels == k]
            constituent_average = cluster_constituents.mean(axis=0)
            self.centroids[i] = constituent_average

    @staticmethod
    def _loss_function(X, cluster_labels, centroids):
        """
        Calculates the mean squared error for each data point x_i and its centroid to monitor loss

        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        :param cluster_labels: np.array(int), an array of size `m`
        :param centroids: np.array(float), an array
        :return: mean_squared_distance: float, the mean squared distance of all the coordinates from their centroids
        """
        k_labels = np.unique(cluster_labels)
        distances = np.empty(len(k_labels))

        for i, k in enumerate(k_labels):
            xi = X[cluster_labels == k]  # features with centroid location of cluster `k`
            centroid_location = centroids[i]
            k_dist = np.linalg.norm(xi - centroid_location, axis=1)  # Vect euclid distance between points and centroid
            distances = np.concatenate((distances, k_dist))

        mean_squared_distance = np.power(distances, 2).mean()
        return mean_squared_distance

    @staticmethod
    def _initial_kmean_params(X):
        """
        Calculates the number of entries in the passed feature matrix `X` and creates the initial cluster labels
        array which will be updated by the class.

        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        :return: m: int, number of entries in passed matrix `X`
        :return: cluster_labels: np.array(), empty array of size `m`
        """
        m = X.shape[0]
        cluster_labels = np.empty(m).astype(int)
        return m, cluster_labels

    def fit(self, X):
        """
        Divides the given feature matrix into `K` clusters based on random initialisation and subsequent refinement

        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        """
        m, cluster_labels = self._initial_kmean_params(X)
        k_indices = np.random.choice(m, size=self.K, replace=False)  # sample indices without replacement
        self.centroids = X[k_indices]

        while self.iters_ran < self.max_iters:
            cluster_labels = self._cluster_assignment(X, cluster_labels)
            self._move_centroids(X, cluster_labels)
            iteration_loss = self._loss_function(X, cluster_labels, self.centroids)

            self.losses.append(iteration_loss)
            self.iters_ran += 1
        self.prediction_labels = cluster_labels  # used for testing on 2D / 3D examples for plotting

    def predict(self, X):
        """
        Following the fitting / determination of the clusters using the `fit` method, this allows users to input new
        data points of shape `m` by `n` and predict which clusters they belong to

        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        :return: np.array(int), an array of size `m`
        """
        m, cluster_labels = self._initial_kmean_params(X)
        cluster_predictions = self._cluster_assignment(X, cluster_labels)
        return cluster_predictions
