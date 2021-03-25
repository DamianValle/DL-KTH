import numpy as np
import os
import sys
import pickle
from ann import ANN
import random

def main():
    path = 'cifar-10-batches-py'
    train_fpath = os.path.join('../', path, 'data_batch_1')
    val_fpath = os.path.join('../', path, 'data_batch_2')
    test_fpath = os.path.join('../', path, 'test_batch')

    x_train, y_train = load_batch(train_fpath)
    x_val, y_val = load_batch(val_fpath)
    x_test, y_test = load_batch(test_fpath)

    print('x_train shape:\t', x_train.shape, '\t y_train shape:\t', y_train.shape)
    print('x_val shape:\t', x_val.shape, '\t y_val shape:\t', y_val.shape)
    print('x_test shape:\t', x_test.shape, '\t y_test shape:\t', y_test.shape)

    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    print('mean shape:\t', mean.shape, '\nstd shape:\t', std.shape)

    x_train = ( x_train - mean ) / std
    x_val = ( x_val- mean ) / std
    x_test = ( x_test - mean ) / std

    y_train = one_hot(y_train)

    x_train = x_train.T
    y_train = y_train.T

    print("x_train shape ", x_train.shape)

    np.random.seed(400)
    random.seed(400)


    ann = ANN(x_train, y_train)
    ann.evaluate_classifier(x_train[:, :15])
    
    ann.minibatch_gd(x_train[:, :], y_train[:, :], x_train[:, :], y_train[:, :])

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3072)
    return np.array(data), np.array(labels)

def one_hot(Y):
    shape = (Y.size, Y.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(Y.size)
    one_hot[rows, Y] = 1
    
    return one_hot


if __name__ == "__main__":
    main()
