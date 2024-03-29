"""
Author: el puto Doppler co
"""

import numpy as np
import matplotlib.pyplot as plt

class ANN:

    def __init__(self):
        self.k = 10
        self.gauss_mean = 0
        self.gauss_std = 0.01
        self.d = 3072
        self.num_hidden = 512

        self.lamda = 6 * 1e-4
        self.batch_size = 100
        self.epochs = 300
        self.lr = 1e-5
        self.xavier = True
        self.jitter = True
        self.dropout = 0.25

        self.lr_min = 1e-5
        self.lr_max = 2e-2
        self.ns = -1
        self.num_cycles = 6

        self.plot = True

        self.init_weights_and_biases()

    def init_weights_and_biases(self):
        if self.xavier:
            self.w1 = np.random.normal(self.gauss_mean, 1 / np.sqrt(self.d), (self.num_hidden, self.d))
            self.w2 = np.random.normal(self.gauss_mean, 1 / np.sqrt(self.num_hidden), (self.k, self.num_hidden))
        else:
            self.w1 = np.random.normal(self.gauss_mean, self.gauss_std, (self.num_hidden, self.d))
            self.w2 = np.random.normal(self.gauss_mean, self.gauss_std, (self.k, self.num_hidden))

        self.b1 = np.zeros((self.num_hidden, 1))
        self.b2 = np.zeros((self.k, 1))

    def train(self, x_train, y_train, x_val, y_val, x_test, y_test):
        num_batches = int(x_train.shape[1] / self.batch_size)
        self.ns = 10 * num_batches

        train_cost_hist = []
        train_acc_hist = []
        val_cost_hist = []
        val_acc_hist = []
        lr_vs_acc = []
        lr = []

        t = 0
        done = False
        for i in range(self.epochs):
            rand = np.random.permutation(num_batches)

            for j in rand:
                j_start = j * self.batch_size
                j_end = j_start + self.batch_size

                x_batch = x_train[:, j_start:j_end]
                y_batch = y_train[:, j_start:j_end]

                if self.jitter:
                    x_batch += 0.01 * np.random.randn(*x_batch.shape)

                grad_w1, grad_b1, grad_w2, grad_b2 = self.compute_gradients(x_batch, y_batch)

                self.w1 -= self.lr * grad_w1
                self.b1 -= self.lr * grad_b1
                self.w2 -= self.lr * grad_w2
                self.b2 -= self.lr * grad_b2

                t += 1
                if t == self.ns * self.num_cycles + 1:
                    done = True
                    break

            if done:
                break

            self.lr = self.cyclical_lr(t)

            train_cost = self.compute_cost(x_train, y_train)
            train_acc = self.compute_accuracy(x_train, y_train)
            val_cost = self.compute_cost(x_val, y_val)
            val_acc = self.compute_accuracy(x_val, y_val)

            train_cost_hist.append(train_cost)
            train_acc_hist.append(train_acc)
            val_cost_hist.append(val_cost)
            val_acc_hist.append(val_acc)

            print("Epoch ", str(i+1), " val acc: ", str(self.compute_accuracy(x_val, y_val)))

        print("TRAINING FINISHED")
        print("Test accuracy: ", self.compute_accuracy(x_test, y_test))

        if self.plot:
            self.plot_graphs(train_acc_hist, train_cost_hist, val_acc_hist, val_cost_hist)

    def cyclical_lr(self, t):
        l = int(t / (2*self.ns))

        if t < (2*l + 1) * self.ns:
            lr = self.lr_min + (self.lr_max - self.lr_min) * (( t - 2*l * self.ns) / self.ns)
        else:
            lr = self.lr_max - (self.lr_max - self.lr_min) * (( t - (2*l + 1) * self.ns ) / self.ns)

        return lr

    def linear_lr(self, epoch):
        return (epoch+1) * 0.005

    def plot_graphs(self, train_acc_hist, train_cost_hist, val_acc_hist, val_cost_hist):
        plt.title('accuracy evolution')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.plot(train_acc_hist, label='train')
        plt.plot(val_acc_hist, label='val')
        plt.legend()
        plt.show()

        plt.title('cost evolution')
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.plot(train_cost_hist, label='train')
        plt.plot(val_cost_hist, label='val')
        plt.legend()
        plt.show()

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def evaluate_classifier(self, X):
        """
        w: k x d
        X: d x n
        b: k x 1

        output: n x k
        """
        s1 = np.dot(self.w1, X) + self.b1
        h = np.maximum(0, s1)

        if self.dropout:
           h = np.where(np.random.random(h.shape) < self.dropout, 0, h)
           h /= (1-self.dropout)

        s2 = np.dot(self.w2, h) + self.b2

        if self.dropout:
            s2 = np.where(np.random.random(s2.shape) < self.dropout, 0, s2)
            s2 /= (1-self.dropout)

        p = self.softmax(s2)

        return p, h

    def compute_cost(self, X, y_true):
        y_pred, _ = self.evaluate_classifier(X)

        return self.cross_entropy(y_true, y_pred) / X.shape[1] + self.lamda * (np.sum(self.w1 ** 2) + np.sum(self.w2 ** 2))

    def cross_entropy(self, y_true, y_pred):
        conf = np.sum(y_true * y_pred, axis=0)
        c_entropy = np.sum(-np.log(conf), axis=0)

        return c_entropy

    def compute_accuracy(self, X, y_true):
        """
        Computes the accuracy of the classifier for a given set of samples and their ground truth labels.

        X: d x n
        y_true, y_pred: k x n
        """

        y_pred, _ = self.evaluate_classifier(X)
        match = len(np.where(np.argmax(y_true, axis=0) == np.argmax(y_pred, axis=0))[0])

        return match/y_true.shape[1]

    def compute_gradients(self, X, y_true):
        """
        X:      d x batch_size
        y_true: k x batch_size
        """
        size = y_true.shape[1]

        y_pred, h = self.evaluate_classifier(X)
        g_batch = y_pred - y_true

        grad_w2 = np.dot(g_batch, h.T) / size + 2 * self.lamda * self.w2
        grad_b2 = np.sum(g_batch, axis=1).reshape(-1,1) / size

        g_batch = np.dot(self.w2.T, g_batch)
        g_batch = np.where(h > 0, g_batch, 0)

        grad_w1 = np.dot(g_batch, X.T) / size + 2 * self.lamda * self.w1
        grad_b1 = np.sum(g_batch, axis=1).reshape(-1, 1) / size

        return grad_w1, grad_b1, grad_w2, grad_b2

    def check_gradients(self, X, y_true):
        grad_w1, _, _, _ = self.compute_gradients(X, y_true)
        grad_w1_num, _, _, _ = self.ComputeGradsNum(X, y_true, self.w1, self.w2, self.b1, self.b2, self.lamda, 1e-6)

        mean_re = np.mean(abs(grad_w1 - grad_w1_num) / np.maximum(abs(grad_w1) + abs(grad_w1_num), np.finfo(float).eps))

        print("Mean Squared Error:\t{:.2e}".format(np.mean((grad_w1 - grad_w1_num) ** 2)))
        print("Mean Relative Error:\t{:.2e}".format(mean_re)) 

    def EvaluateClassifier(self, X, W1, W2, b1, b2):
        s1 = np.dot(W1, X) + b1
        h = np.maximum(0, s1)
        s2 = np.dot(W2, h) + b2
        p = self.softmax(s2)

        return p

    def ComputeCost(self, X, Y, W1, W2, b1, b2, lam):
        y_pred = self.EvaluateClassifier(X, W1, W2, b1, b2)

        return self.cross_entropy(Y, y_pred) / X.shape[1] + lam * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    def ComputeGradsNum(self, X, Y, W1, W2, b1, b2, lam, h):
        grad_W1 = np.zeros(W1.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_W2 = np.zeros(W2.shape)
        grad_b2 = np.zeros(b2.shape)

        c = self.ComputeCost(X, Y, W1, W2, b1, b2, lam)
        
        for i in range(len(b1)):
            b1_try = np.array(b1)
            b1_try[i] += h
            c2 = self.ComputeCost(X, Y, W1, W2, b1_try, b2, lam)
            grad_b1[i] = (c2 - c) / h

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                W1_try = np.array(W1)
                W1_try[i,j] += h
                c2 = self.ComputeCost(X, Y, W1_try, W2, b1, b2, lam)
                grad_W1[i,j] = (c2 - c) / h
                
        for i in range(len(b2)):
            b2_try = np.array(b2)
            b2_try[i] += h
            c2 = self.ComputeCost(X, Y, W1, W2, b1, b2_try, lam)
            grad_b2[i] = (c2 - c) / h

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                W2_try = np.array(W2)
                W2_try[i,j] += h
                c2 = self.ComputeCost(X, Y, W1, W2_try, b1, b2, lam)
                grad_W2[i,j] = (c2 - c) / h

        return grad_W1, grad_b1, grad_W2, grad_b2

    def ComputeGradsNumSlow(self, X, Y, W1, W2, b1, b2, lam, h):
        grad_W1 = np.zeros(W1.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_W2 = np.zeros(W2.shape)
        grad_b2 = np.zeros(b2.shape)
        
        for i in range(len(b1)):
            b1_try = np.array(b1)
            b1_try[i] -= h
            c1 = self.ComputeCost(X, Y, W1, W2, b1_try, b2, lam)

            b1_try = np.array(b1)
            b1_try[i] += h
            c2 = self.ComputeCost(X, Y, W1, W2, b1_try, b2, lam)

            grad_b1[i] = (c2 - c1) / (2 * h)

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                W1_try = np.array(W1)
                W1_try[i,j] -= h
                c1 = self.ComputeCost(X, Y, W1_try, W2, b1, b2, lam)

                W1_try = np.array(W1)
                W1_try[i,j] += h
                c2 = self.ComputeCost(X, Y, W1_try, W2, b1, b2, lam)

                grad_W1[i,j] = (c2 - c1) / (2 * h)
                
        for i in range(len(b2)):
            b2_try = np.array(b2)
            b2_try[i] -= h
            c1 = self.ComputeCost(X, Y, W1, W2, b1, b2_try, lam)

            b2_try = np.array(b2)
            b2_try[i] += h
            c2 = self.ComputeCost(X, Y, W1, W2, b1, b2_try, lam)

            grad_b2[i] = (c2 - c1) / (2 * h)

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                W2_try = np.array(W2)
                W2_try[i,j] -= h
                c1 = self.ComputeCost(X, Y, W1, W2_try, b1, b2, lam)

                W2_try = np.array(W2)
                W2_try[i,j] += h
                c2 = self.ComputeCost(X, Y, W1, W2_try, b1, b2, lam)

                grad_W2[i,j] = (c2 - c1) / (2 * h)

        return [grad_W1, grad_W2, grad_b1, grad_b2]