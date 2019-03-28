"""
This module provides utility functions for reading and writing objects to a file using pickle.
"""

import gzip
import pickle

import numpy as np
import theano
import theano.tensor as T


def casting(x):
    return np.array(x).astype(theano.config.floatX)


def LogSumExp(x, axis=None):
    """
    Compute the LogSumExp.

    :param x: a matrix of dimension [n_samples, batch_size, n_features].
    :param axis: axis along which to compute the LogSumExp.
    :return LogSumExp
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max


def save_object(obj, filename):

    """
    Function that saves an object to a file using pickle
    """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest:
        dest.write(result)
    dest.close()


def load_object(filename):

    """
    Function that loads an object from a file using pickle
    """

    with gzip.GzipFile(filename, 'rb') as source:
        result = source.read()
    ret = pickle.loads(result)
    source.close()

    return ret


def load_data(X_tr_bran, y_tr_con, X_te_bran, y_te_con):
    """
    Loads the dataset

    X_tr_bran and y_tr_con are hard-coded above.

    """

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #whose rows correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that has the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    train_set = (X_tr_bran, y_tr_con)
    test_set = (X_te_bran, y_te_con)

    train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval, train_set[0].shape[0], train_set[0].shape[1], np.max(train_set[1]) + 1


def initialise_regression_data(inputs_path, scores_path, train_test_split):
    """
    Reads regression data from a file and divides it into train and test sets.

    :param inputs_path: path of the file to read the inputs (x values) from.
    :param scores_path: path of the file to read the scores (y values) from.
    :param train_test_split: float which describes the train/test split
                             e.g. train_test_split = 0.9 means the data
                             is 90% train and 10% test.
    :return X_train, X_test,, y_train, y_test: training and test sets.
    """

    X = np.loadtxt(inputs_path)
    y = -np.loadtxt(scores_path)  # negative because we're minimising.
    y = y.reshape((-1, 1))

    n = X.shape[0]  # number of training examples
    permutation = np.random.choice(n, n, replace=False)  # for shuffling data

    X_train = X[permutation, :][0: np.int(np.round(train_test_split * n)), :]
    X_test = X[permutation, :][np.int(np.round(train_test_split * n)):, :]
    y_train = y[permutation][0: np.int(np.round(train_test_split * n))]
    y_test = y[permutation][np.int(np.round(train_test_split * n)):]

    return X_train, X_test, y_train, y_test
