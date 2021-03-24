import numpy as np
import os
import sys
import pickle

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

    data = data.reshape(data.shape[0], 3, 32, 32)
    return np.array(data), np.array(labels)

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def evaluate_classifier(X, W, b):
    P = np.zeros((X.shape[0], K))
    
    for i in range(X.shape[0]):
        P[i] = np.dot(W, X[i]) + b
        
    return np.array([softmax(x) for x in P])

def compute_cost(X, Y, W, b, lamb):
    
    cost = 0

    P = evaluate_classifier(X, W, b)
    
    for i in range(X.shape[0]):
        cost -= np.log2(P[i][Y[i]] + sys.float_info.epsilon)
    
    cost /=  X.shape[0]
    
    cost += lamb * np.sum(W**2)
    
    return cost

if __name__ == "__main__":

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

    print(x_train.mean(), x_train.std())
    print(x_val.mean(), x_val.std())
    print(x_test.mean(), x_test.std())

    K = 10  # number of classes

    W = np.random.normal(0, 0.01, (K, 3072))
    b = np.random.normal(0, 0.01, K)