import numpy as np
import os
import sys
import pickle
from ann import ANN
import random

def main():
    path = 'cifar-10-batches-py'

    x_train, y_train, x_val, y_val, x_test, y_test = load_all(path)

    mean, std = x_train.mean(axis=0), x_train.std(axis=0)

    x_train = ( x_train - mean ) / std
    x_val = ( x_val - mean ) / std
    x_test = ( x_test - mean ) / std
    
    x_train = x_train.T
    y_train = y_train.T
    x_val = x_val.T
    y_val = y_val.T
    x_test = x_test.T
    y_test = y_test.T

    ann = ANN()
    ann.train(x_train, y_train, x_val, y_val, x_test, y_test)

def load_subset(path):

    train_fpath = os.path.join('../', path, 'data_batch_1')
    val_fpath = os.path.join('../', path, 'data_batch_2')
    test_fpath = os.path.join('../', path, 'test_batch')

    x_train, y_train = load_batch(train_fpath)
    x_val, y_val = load_batch(val_fpath)
    x_test, y_test = load_batch(test_fpath)

    return x_train, y_train, x_val, y_val, x_test, y_test

def load_all(path):

    fpath = os.path.join('../', path, 'data_batch_' + str(1))
    x_train, y_train = load_batch(fpath)
    
    for i in range(2, 6):
        fpath = os.path.join('../', path, 'data_batch_' + str(i))
        x, y = load_batch(fpath)
        x_train = np.append(x_train, x, axis=0)
        y_train = np.append(y_train, y, axis=0)

    x_val = x_train[-1000:,:]
    y_val = y_train[-1000:,:]
    x_train = x_train[:-1000,:]
    y_train = y_train[:-1000,:]

    test_fpath = os.path.join('../', path, 'test_batch')
    x_test, y_test = load_batch(test_fpath)

    return x_train, y_train, x_val, y_val, x_test, y_test

def load_batch(fpath, label_key='labels'):
    """
    Internal utility for parsing CIFAR data.
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
    return np.array(data), one_hot(np.array(labels))

def one_hot(Y):
    shape = (Y.size, Y.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(Y.size)
    one_hot[rows, Y] = 1
    
    return one_hot

if __name__ == "__main__":
    main()