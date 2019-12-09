import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

from models.ClusterFold import KlusterFoldCrossValidation


def plotter(A, b):
    plt.scatter(A[:, 0], A[:, 1], c=b)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()


n = 1000
X = np.random.randn(n, 2)
y = np.ones(n)
y[X[:, 0] + X[:, 1] < 0] = 0

kfcv = KlusterFoldCrossValidation(X, y, test_div=5)
X_train, X_test, y_train, y_test = kfcv.train_test_split(test_label=0)


plotter(X, y)
plotter(X, kfcv.k_assignments)
plotter(X_train, y_train)
plotter(X_test, y_test)


knr = KNeighborsRegressor()
cv_results = kfcv.cross_cluster_validate(estimator=knr, metric=mean_absolute_error, cv=10)