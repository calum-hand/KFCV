import numpy as np
from sklearn.linear_model import LinearRegression  # chosen as very simple model, therefore allows quick testing
from sklearn.metrics import mean_absolute_error as mae  # arbitrary choice

from models.ClusterFold import KlusterFoldCrossValidation


def test_kfcv_initialisation():
    """
    Test behaviour of kfcv at initialisation:
    * Internal attributes updated correctly
    * Cluster assignments are int
    * Get as many output labels as expected
    """
    n = 100
    X, y = np.random.randn(n, 2), np.ones(n)
    y[X[:, 0] + X[:, 1] < 0] = 0
    k = 5
    kfcv = KlusterFoldCrossValidation(X, y, test_div=k)

    assert X.shape == kfcv.X.shape  # check no change when initialising object
    assert y.shape == kfcv.y.shape
    assert len(kfcv.X) == len(kfcv.y) == len(kfcv.k_assignments)  # check assignment generated for each data point
    assert kfcv.X_train == kfcv.X_test == kfcv.y_train == kfcv.y_test is None  # check not yet initialised
    assert kfcv.k_assignments.dtype == kfcv.k_labels.dtype == 'int'  # check labels are integer values
    assert len(kfcv.k_labels) == k  # check clustering produced anticipated number of clusters


def test_kfcv_train_test_split():
    """
    Test the output of kfcv `train_test_split` method:
    * Feature matrices are 2D and target arrays 1D
    * Train and test sets have equal number of data points
    * The number of data points in train and test equals that of the original data
    """
    n = 100
    X, y = np.random.randn(n, 2), np.ones(n)
    y[X[:, 0] + X[:, 1] < 0] = 0
    k = 5
    kfcv = KlusterFoldCrossValidation(X, y, test_div=k)
    X_train, X_test, y_train, y_test = kfcv.train_test_split()

    assert len(X_train.shape) == len(X_test.shape) == 2  # feature space is 2D
    assert len(y_train.shape) == len(y_test.shape) == 1  # target space is 1D
    assert len(X_train) == len(y_train)  # train is same size
    assert len(X_test) == len(y_test)  # test is same size
    assert len(X_train) + len(X_test) == n  # train and test data sets recombine to give full sized data set
    assert X_train.shape[1] == X_test.shape[1] == X.shape[1]  # check number of features is preserved


def test_kfcv_cross_cluster_validate():
    """
    Test the output of the kfcv's `cross_cluster_validate` method:
    * 2 dictionary entries returned
    * Dictionary keys are as expected
    * Number of entries in output lists equal the number of cross validations performed
    """
    n = 100
    X, y = np.random.randn(n, 2), np.ones(n)
    y[X[:, 0] + X[:, 1] < 0] = 0
    k = 5
    kfcv = KlusterFoldCrossValidation(X, y, test_div=k)
    kfcv.train_test_split()
    lr = LinearRegression()
    cv_count = 5
    cv_results = kfcv.cross_cluster_validate(estimator=lr, metric=mae, cv=cv_count)

    assert len(cv_results) == 2  # only two entries in returned dictionary
    assert 'train' in cv_results and 'cv' in cv_results  # keys are as expected
    train, cv = cv_results['train'], cv_results['cv']
    assert len(train) == len(cv) == cv_count  # get as many values out as we expected to


def test_kfcv_estimator_test():
    """
    Test output of the kfcv `estimator_test` method:
    * Have the internal feature (train/test) and target (train/test) changed since `train_test_split` method was called
    * Output value is a float
    """
    n = 100
    X, y = np.random.randn(n, 2), np.ones(n)
    y[X[:, 0] + X[:, 1] < 0] = 0
    k = 5
    kfcv = KlusterFoldCrossValidation(X, y, test_div=k)
    X_train, X_test, y_train, y_test = kfcv.train_test_split()
    lr = LinearRegression()
    cv_count = 5
    kfcv.cross_cluster_validate(estimator=lr, metric=mae, cv=cv_count)

    assert kfcv.X_train.all() == X_train.all()  # check that the internal attributes do not undergo change after split
    assert kfcv.X_test.all() == X_test.all()
    assert kfcv.y_train.all() == y_train.all()
    assert kfcv.y_test.all() == y_test.all()

    model_out = kfcv.estimator_test(estimator=lr, metric=mae)
    assert model_out.dtype == float  # ensure output is float
