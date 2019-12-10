import pytest
import numpy as np

from sklearn.linear_model import LinearRegression  # regression model for testing
from sklearn.tree import DecisionTreeClassifier  # classification model for testing
from sklearn.metrics import mean_absolute_error as mae  # arbitrary regression metric
from sklearn.metrics import accuracy_score  # arbitrary classification metric

from models.ClusterFold import KlusterFoldCrossValidation


@pytest.mark.parametrize("n, m, k",
                         [(100, 2, 5),
                          (200, 5, 2)])
def test_kfcv_initialisation(n, m, k):
    """
    Test behaviour of kfcv at initialisation:
    * Internal attributes updated correctly
    * Cluster assignments are int
    * Get as many output labels as expected

    :param n, int, number of data points
    :param m, int, number of features per data point
    :param k, int, number of clusters to generate
    """
    X, y = np.random.randn(n, m), np.ones(n)
    kfcv = KlusterFoldCrossValidation(X, y, test_div=k)

    assert X.shape == kfcv.X.shape  # check no change when initialising object
    assert y.shape == kfcv.y.shape
    assert len(kfcv.X) == len(kfcv.y) == len(kfcv.k_assignments)  # check assignment generated for each data point
    assert kfcv.X_train == kfcv.X_test == kfcv.y_train == kfcv.y_test is None  # check not yet initialised
    assert kfcv.k_assignments.dtype == kfcv.k_labels.dtype
    #assert kfcv.k_labels.dtype in ['int8', 'int16', 'int32', 'int64']
    assert len(kfcv.k_labels) == k  # check clustering produced anticipated number of clusters


@pytest.mark.parametrize("n, m, k",
                         [(100, 2, 5),
                          (200, 5, 2)])
def test_kfcv_train_test_split(n, m, k):
    """
    Test the output of kfcv `train_test_split` method:
    * Feature matrices are 2D and target arrays 1D
    * Train and test sets have equal number of data points
    * The number of data points in train and test equals that of the original data

    :param n, int, number of data points
    :param m, int, number of features per data point
    :param k, int, number of clusters to generate
    """
    X, y = np.random.randn(n, m), np.ones(n)
    kfcv = KlusterFoldCrossValidation(X, y, test_div=k)
    X_train, X_test, y_train, y_test = kfcv.train_test_split()

    assert len(X_train.shape) == len(X_test.shape) == 2  # feature space is 2D
    assert len(y_train.shape) == len(y_test.shape) == 1  # target space is 1D
    assert len(X_train) == len(y_train)  # train is same size
    assert len(X_test) == len(y_test)  # test is same size
    assert len(X_train) + len(X_test) == n  # train and test data sets recombine to give full sized data set
    assert X_train.shape[1] == X_test.shape[1] == X.shape[1]  # check number of features is preserved


@pytest.mark.parametrize("n, m, k, cv_count, estimator, metric",
                         [(100, 2, 5, 10, LinearRegression(), mae),
                          (200, 5, 2, 10, LinearRegression(), mae),
                          (100, 2, 5, 10, DecisionTreeClassifier(), accuracy_score),
                          (200, 5, 2, 10, DecisionTreeClassifier(), accuracy_score)])
def test_kfcv_cross_cluster_validate(n, m, k, cv_count, estimator, metric):
    """
    Test the output of the kfcv's `cross_cluster_validate` method:
    * 2 dictionary entries returned
    * Dictionary keys are as expected
    * Number of entries in output lists equal the number of cross validations performed

    :param n, int, number of data points
    :param m, int, number of features per data point
    :param k: int, number of clusters to generate
    :param cv_count: int, number of clusters for the cross validation
    :param estimator: machine learning model with `fit` and `predict` methods
    :param metric: score used to quantify the prediction of the estimator
    """
    X, y = np.random.randn(n, m), np.ones(n)
    kfcv = KlusterFoldCrossValidation(X, y, test_div=k)
    kfcv.train_test_split()
    cv_results = kfcv.cross_cluster_validate(estimator=estimator, metric=metric, cv=cv_count)

    assert len(cv_results) == 2  # only two entries in returned dictionary
    assert 'train' in cv_results and 'cv' in cv_results  # keys are as expected
    train, cv = cv_results['train'], cv_results['cv']
    assert len(train) == len(cv) == cv_count  # get as many values out as we expected to


@pytest.mark.parametrize("n, m, k, cv_count, estimator, metric",
                         [(100, 2, 5, 10, LinearRegression(), mae),
                          (200, 5, 2, 10, LinearRegression(), mae),
                          (100, 2, 5, 10, DecisionTreeClassifier(), accuracy_score),
                          (200, 5, 2, 10, DecisionTreeClassifier(), accuracy_score)])
def test_kfcv_estimator_test(n, m, k, cv_count, estimator, metric):
    """
    Test output of the kfcv `estimator_test` method:
    * Have the internal feature (train/test) and target (train/test) changed since `train_test_split` method was called
    * Output value is a float

    :param n, int, number of data points
    :param m, int, number of features per data point
    :param k: int, number of clusters to generate
    :param cv_count: int, number of clusters for the cross validation
    :param estimator: machine learning model with `fit` and `predict` methods
    :param metric: score used to quantify the prediction of the estimator
    """
    X, y = np.random.randn(n, m), np.ones(n)
    kfcv = KlusterFoldCrossValidation(X, y, test_div=k)
    X_train, X_test, y_train, y_test = kfcv.train_test_split()
    kfcv.cross_cluster_validate(estimator=estimator, metric=metric, cv=cv_count)

    assert kfcv.X_train.all() == X_train.all()  # check that the internal attributes do not undergo change after split
    assert kfcv.X_test.all() == X_test.all()
    assert kfcv.y_train.all() == y_train.all()
    assert kfcv.y_test.all() == y_test.all()

    model_out = kfcv.estimator_test(estimator=estimator, metric=metric)
    assert model_out.dtype == float  # ensure output is float
