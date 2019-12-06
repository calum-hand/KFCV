import numpy as np
from sklearn.cluster import KMeans


class ClusterFoldValidation(object):

    def __init__(self, k_clusters, model):
        self.k_clusters = k_clusters
        self.model = model

        self.k_assignments = None
        self.k_labels = None


    def _initial_cluster_assignment(self, X):
        mdl = self.model(self.k_clusters)
        mdl.fit(X)
        self.k_assignments = mdl.labels_
        self.k_labels = np.unique(self.k_assignments)

    def _choose_test_set(self):
        return self

    def train_test_split(self, X, y):
        return self
