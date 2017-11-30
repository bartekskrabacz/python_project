from sklearn.utils import  indexable, check_random_state, safe_indexing
from sklearn.utils.validation import _num_samples, column_or_1d
import warnings
from math import ceil, floor
import numpy as np
from itertools import chain
from Spliter import Spliter


class Categorizer:

    def __init__ (self, lfw_people):
        self.lfw_people=lfw_people
        self.n_samples, self.h, self.w = self.lfw_people.images.shape

    def return_shape_data(self):
        return self.n_samples, self.h, self.w

    def categorize(self):

        # introspect the images arrays to find the shapes (for plotting)
        self.n_samples, self.h, self.w
        # for machine learning we use the 2 data directly (as relative pixel
        # positions info is ignored by this model)
        X = self.lfw_people.data
        n_features = X.shape[1]

        # the label to predict is the id of the person
        y = self.lfw_people.target
        target_names = self.lfw_people.target_names
        n_classes = target_names.shape[0]

        print("Total dataset size:")
        print("n_samples: %d" % self.n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)


        # #############################################################################
        # Split into a training set and a test set using a stratified k fold

        # split into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        return  X_train, X_test, y_train, y_test


def train_test_split(*arrays, **options):
    """Split arrays or matrices into random train and test subsets

    Quick utility that wraps input validation and
    ``next(ShuffleSplit().split(X, y))`` and application to input data
    into a single call for splitting (and optionally subsampling) data in a
    oneliner.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.25.
        The default will change in version 0.21. It will remain 0.25 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.

    train_size : float, int, or None, default None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like or None (default is None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    ...
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> y_train
    [2, 0, 3]
    >>> X_test
    array([[2, 3],
           [8, 9]])
    >>> y_test
    [1, 4]

    >>> train_test_split(y, shuffle=False)
    [[0, 1, 2], [3, 4]]

    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', 'default')
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    stratify = options.pop('stratify', None)
    shuffle = options.pop('shuffle', True)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    if test_size == 'default':
        test_size = None
        if train_size is not None:
            warnings.warn("From version 0.21, test_size will always "
                          "complement train_size unless both "
                          "are specified.",
                          FutureWarning)

    if test_size is None and train_size is None:
        test_size = 0.25

    arrays = indexable(*arrays)

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for "
                "shuffle=False")

        n_samples = _num_samples(arrays[0])
        n_train, n_test = _validate_shuffle_split(n_samples, test_size,
                                                  train_size)

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        spliter = Spliter()
        if stratify is not None:
            CVClass = spliter.StratifiedShuffleSplit
        else:
            CVClass = spliter.ShuffleSplit

        cv = CVClass(test_size=test_size,
                     train_size=train_size,
                     random_state=random_state)

        train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(chain.from_iterable((safe_indexing(a, train),
                                     safe_indexing(a, test)) for a in arrays))


def _validate_shuffle_split(n_samples, test_size, train_size):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    """
    if (test_size is not None and
            np.asarray(test_size).dtype.kind == 'i' and
            test_size >= n_samples):
        raise ValueError('test_size=%d should be smaller than the number of '
                         'samples %d' % (test_size, n_samples))

    if (train_size is not None and
            np.asarray(train_size).dtype.kind == 'i' and
            train_size >= n_samples):
        raise ValueError("train_size=%d should be smaller than the number of"
                         " samples %d" % (train_size, n_samples))

    if test_size == "default":
        test_size = 0.1

    if np.asarray(test_size).dtype.kind == 'f':
        n_test = ceil(test_size * n_samples)
    elif np.asarray(test_size).dtype.kind == 'i':
        n_test = float(test_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif np.asarray(train_size).dtype.kind == 'f':
        n_train = floor(train_size * n_samples)
    else:
        n_train = float(train_size)

    if test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n_samples))

    return int(n_train), int(n_test)
