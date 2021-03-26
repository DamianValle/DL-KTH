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

        self.lamda = 1
        self.batch_size = 100
        self.epochs = 40
        self.lr = 0.001

        self.w = np.random.normal(self.gauss_mean, self.gauss_std, (self.k, self.d))
        self.b = np.random.normal(self.gauss_mean, self.gauss_std, (self.k, 1))

    def train(self, x_train, y_train, x_val, y_val, x_test, y_test):

        num_batches = int(x_train.shape[1] / self.batch_size)

        train_cost_hist = []
        train_acc_hist = []
        val_cost_hist = []
        val_acc_hist = []

        for i in range(self.epochs):
            rand = np.random.permutation(num_batches)

            for j in rand:
                j_start = j * num_batches
                j_end = j_start + self.batch_size

                x_batch = x_train[:, j_start:j_end]
                y_batch = y_train[:, j_start:j_end]

                y_batch_pred = self.evaluate_classifier(x_batch)

                grad_w, grad_b = self.compute_gradients(x_batch, y_batch, y_batch_pred)

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

            train_cost = self.compute_cost(x_train, y_train)
            train_acc = self.compute_accuracy(x_train, y_train)
            val_cost = self.compute_cost(x_val, y_val)
            val_acc = self.compute_accuracy(x_val, y_val)

            train_cost_hist.append(train_cost)
            train_acc_hist.append(train_acc)
            val_cost_hist.append(val_cost)
            val_acc_hist.append(val_acc)

            print("Epoch ", str(i+1), " training loss: ", str(self.compute_cost(x_train, y_train)))

        print("TRAINING FINISHED")
        print("Test accuracy: ", self.compute_accuracy(x_test, y_test))

        self.plot_graphs(train_acc_hist, train_cost_hist, val_acc_hist, val_cost_hist)

    def plot_graphs(self, train_acc_hist, train_cost_hist, val_acc_hist, val_cost_hist):

        plt.title('accuracy evolution')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.plot(train_acc_hist, label='train')
        plt.plot(val_acc_hist, label='val')
        plt.show()

        plt.title('cost evolution')
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.plot(train_cost_hist, label='train')
        plt.plot(val_cost_hist, label='val')
        plt.show()

        self.montage()

    def montage(self):
        """ Display the image for each label in W """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,5)
        for i in range(2):
            for j in range(5):
                im  = self.w[i*5+j,:].reshape(32,32,3, order='F')
                sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
                sim = sim.transpose(1,0,2)
                ax[i][j].imshow(sim, interpolation='nearest')
                ax[i][j].set_title("y="+str(5*i+j))
                ax[i][j].axis('off')
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

        return self.softmax(np.dot(self.w, X) + self.b)

    def compute_cost(self, X, y_true):

        y_pred = self.evaluate_classifier(X)

        return self.cross_entropy(y_true, y_pred) / X.shape[1] + self.lamda * np.sum( self.w ** 2 )

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

        y_pred = self.evaluate_classifier(X)
        match = len(np.where(np.argmax(y_true, axis=0) == np.argmax(y_pred, axis=0))[0])

        return match/y_true.shape[1]

    def compute_gradients(self, X, y_true, y_pred):

        size = y_true.shape[1]

        g_batch = y_pred - y_true

        grad_w = 1/size * np.dot(g_batch, X.T) + 2 * self.lamda * self.w
        grad_b = 1/size * np.sum(g_batch, axis=1).reshape(-1,1)

        return grad_w, grad_b

    def check_gradients(self, X, y_true):

        y_pred = self.evaluate_classifier(X)

        grad_w, grad_b = self.compute_gradients(X, y_true, y_pred)
        grad_w_num, grad_b_num = self.ComputeGradsNum(X, self.w, self.b, self.lamda, y_true, y_pred, 1e-6)

        print("grad_w shape: ", grad_w.shape)
        print("grad_w_num shape: ", grad_w_num.shape)
        print(grad_w)
        print(grad_w_num)
        print(grad_w - grad_w_num)

        print("Squared Error: ", np.sum(grad_w - grad_w_num) ** 2)

    def ComputeGradsNum(self, X, W, b, lamda, Y, P, h):

        no 	= 	W.shape[0]
        d 	= 	X.shape[0]

        grad_W = np.zeros(W.shape)
        grad_b = np.zeros((no, 1))

        c = self.ComputeCost(X, Y, W, b, lamda)
        
        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] += h
            c2 = self.ComputeCost(X, Y, W, b_try, lamda)
            grad_b[i] = (c2-c) / h

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i,j] += h
                c2 = self.ComputeCost(X, Y, W_try, b, lamda)
                grad_W[i,j] = (c2-c) / h

        return [grad_W, grad_b]


    def EvaluateClassifier(self, X, W, b):
        """
        w: k x d
        X: d x n
        b: k x 1

        output: n x k
        """

        return self.softmax(np.dot(W, X) + b)

    def ComputeCost(self, X, y_true, W, b, lamda):

        y_pred = self.EvaluateClassifier(X, W, b)

        return self.cross_entropy(y_true, y_pred) / X.shape[1] + lamda * np.sum( W ** 2 )