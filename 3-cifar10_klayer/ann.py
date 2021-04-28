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

        self.lamda = 0.005
        self.batch_size = 100
        self.epochs = 300
        self.lr = 1e-5
        self.jitter = False
        self.batch_norm = True

        self.lr_min = 1e-5
        self.lr_max = 1e-2
        self.ns = -1
        self.num_cycles = 2
        self.alpha = 0.7

        self.h_nodes = [50, 30, 20, 20, 10, 10, 10, 10]
        self.num_layers = len(self.h_nodes)

        self.plot = True

        self.init_weights_and_biases()
        self.init_gamma_and_beta()

    def init_gamma_and_beta(self):
        self.gamma = [None] * (self.num_layers)
        self.gamma = [np.random.normal(self.gauss_mean, np.sqrt(2/self.h_nodes[i]), (self.h_nodes[i], 1)) for i in range(self.num_layers)]

        self.beta = [None] * (self.num_layers + 1)
        for i in range(self.num_layers):
            self.beta[i] = np.zeros((self.h_nodes[i], 1))
        self.beta[self.num_layers] = np.zeros((self.k, 1))

        self.mu_avg = None
        self.var_avg = None

    def init_weights_and_biases(self):
        """
        self.w matrix shape structure: [(d, h_1), (h_1, h_2), ..., (h_n-1, h_n), (h_n, k)]
        """

        sigma_weights_w = np.zeros(self.num_layers + 1)
        sigma_weights_w[0] = 1 / np.sqrt(self.d)
        for i in range(1, self.num_layers + 1):
            sigma_weights_w[i] = 1 / np.sqrt(self.h_nodes[i-1])

        #meter aqui None??
        #self.w = [np.zeros((2, 2))] * (self.num_layers + 1)
        self.w = [None] * (self.num_layers + 1)
        self.b = [None] * (self.num_layers + 1)

        self.w[0] = np.random.normal(self.gauss_mean, sigma_weights_w[0], (self.h_nodes[0], self.d))
        for i in range(1, self.num_layers):
            self.w[i] = np.random.normal(self.gauss_mean, sigma_weights_w[i], (self.h_nodes[i], self.h_nodes[i-1]))
        self.w[self.num_layers] = np.random.normal(self.gauss_mean, sigma_weights_w[self.num_layers], (self.k, self.h_nodes[self.num_layers-1]))

        for i in range(self.num_layers):
            self.b[i] = np.zeros((self.h_nodes[i], 1))
        self.b[self.num_layers] = np.zeros((self.k, 1))

    def train(self, x_train, y_train, x_val, y_val, x_test, y_test):
        num_batches = int(x_train.shape[1] / self.batch_size)
        self.ns = 75 * 45000 / num_batches

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

                #if self.jitter:
                #    x_batch += 0.01 * np.random.randn(*x_batch.shape)

                grad_w, grad_b, grad_beta, grad_gamma = self.compute_gradients(x_batch, y_batch)
                self.parameter_step(grad_w, grad_b, grad_beta, grad_gamma)

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
        print("Test accuracy: ", self.compute_accuracy(x_test, y_test, 'test'))

        if self.plot:
            self.plot_graphs(train_acc_hist, train_cost_hist, val_acc_hist, val_cost_hist)

    def parameter_step(self, grad_w, grad_b, grad_beta, grad_gamma):

        if self.batch_norm:
            for i in range(self.num_layers):
                self.beta[i] = self.beta[i] - self.lr * grad_beta[i]
                self.gamma[i] = self.gamma[i] - self.lr * grad_gamma[i]

        for i in range(self.num_layers+1):
            self.w[i] -= self.lr * grad_w[i]
            self.b[i] -= self.lr * grad_b[i]


    def update_mu_var_avg(self, mu, var):
        self.mu_avg = mu if self.mu_avg == None else [self.alpha * self.mu_avg[l] + (1 - self.alpha) * mu[l] for l in range(len(mu))]
        self.var_avg = var if self.var_avg == None else [self.alpha * self.var_avg[l] + (1 - self.alpha) * var[l] for l in range(len(var))]

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
        act_h =     [None] * (self.num_layers)
        s =         [None] * (self.num_layers + 1)
        norm_s =  [None] * (self.num_layers)
        final_s =   [None] * (self.num_layers)
        mu =        [None] * (self.num_layers)
        var =       [None] * (self.num_layers)

        # first layer
        s[0] = np.dot(self.w[0], X) + self.b[0]

        if self.batch_norm:
            #mu[0] = np.sum(s[0], axis = 1) / s[0].shape[1]
            mu[0] = np.mean(s[0], axis = 1)
            #var[0] = 1/ s[0].shape[1] * np.sum(((s[0].T - mu[0]).T)**2, axis = 1)
            var[0] = np.var(s[0], axis = 1)
            norm_s[0] = self.batch_normalize(s[0], mu[0], var[0])
            final_s[0] = self.gamma[0] * norm_s[0] + self.beta[0]

            act_h[0] = np.maximum(0, final_s[0])
        else:
            act_h[0] = np.maximum(0, s[0])

        # hidden layers
        for i in range(1, self.num_layers):
            s[i] = np.dot(self.w[i], act_h[i-1]) + self.b[i]

            if self.batch_norm:
                #mu[i] = 1/s[i].shape[1] * np.sum(s[i], axis = 1)
                mu[i] = np.mean(s[i], axis = 1)
                #var[i] = 1/ s[i].shape[1] * np.sum(((s[i].T - mu[i]).T)**2, axis = 1)
                var[i] = np.var(s[i], axis = 1)
                norm_s[i] = self.batch_normalize(s[i], mu[i], var[i])
                final_s[i] = self.gamma[i] * norm_s[i] + self.beta[i]

                act_h[i] = np.maximum(0, final_s[i])
            else:
                act_h[i] = np.maximum(0, s[i])

        # last layer
        s[self.num_layers] = np.dot(self.w[self.num_layers], act_h[self.num_layers-1]) + self.b[self.num_layers]

        p = self.softmax(s[self.num_layers])

        if self.batch_norm:
            self.update_mu_var_avg(mu, var)

        return p, act_h, s, norm_s, mu, var

    def evaluate_classifier_test(self, X):
        act_h =     [None] * (self.num_layers)
        s =         [None] * (self.num_layers + 1)
        norm_s =  [None] * (self.num_layers)
        final_s =   [None] * (self.num_layers)
        mu =        [None] * (self.num_layers)
        var =       [None] * (self.num_layers)

        # first layer
        s[0] = np.dot(self.w[0], X) + self.b[0]

        if self.batch_norm:

            norm_s[0] = self.batch_normalize(s[0], self.mu_avg[0], self.var_avg[0])
            final_s[0] = self.gamma[0] * norm_s[0] + self.beta[0]

            act_h[0] = np.maximum(0, final_s[0])
        else:
            act_h[0] = np.maximum(0, s[0])

        # hidden layers
        for i in range(1, self.num_layers):
            s[i] = np.dot(self.w[i], act_h[i-1]) + self.b[i]

            if self.batch_norm:
                norm_s[i] = self.batch_normalize(s[i], self.mu_avg[i], self.var_avg[i])
                final_s[i] = self.gamma[i] * norm_s[i] + self.beta[i]

                act_h[i] = np.maximum(0, final_s[i])
            else:
                act_h[i] = np.maximum(0, s[i])

        # last layer
        s[self.num_layers] = np.dot(self.w[self.num_layers], act_h[self.num_layers-1]) + self.b[self.num_layers]
        p = self.softmax(s[self.num_layers])

        return p

    def batch_normalize(self, s, mu, var):
        part1 = np.diag(np.power((var + 1e-6),(-1/2)))
        part2 = (s.T - mu).T
        
        return np.dot(part1, part2)

    def compute_cost(self, X, y_true):
        y_pred, _, _, _, _, _ = self.evaluate_classifier(X)

        reg_term = sum(list(map(lambda x: np.sum(x**2), self.w)))

        return self.cross_entropy(y_true, y_pred) / X.shape[1] + self.lamda * reg_term

    def cross_entropy(self, y_true, y_pred):
        conf = np.sum(y_true * y_pred, axis=0)
        c_entropy = np.sum(-np.log(conf), axis=0)

        return c_entropy

    def compute_accuracy(self, X, y_true, split='train'):
        """
        Computes the accuracy of the classifier for a given set of samples and their ground truth labels.

        X: d x n
        y_true, y_pred: k x n
        """
        if split == 'train':
            y_pred, _, _, _, _, _ = self.evaluate_classifier(X)
        elif split == 'test':
            y_pred = self.evaluate_classifier_test(X)

        match = len(np.where(np.argmax(y_true, axis=0) == np.argmax(y_pred, axis=0))[0])

        return match/y_true.shape[1]

    def compute_gradients(self, X, y_true):
        """
        X:      d x batch_size
        y_true: k x batch_size
        """
        size = y_true.shape[1]

        grad_b = [None] * (self.num_layers+1)
        grad_w = [None] * (self.num_layers+1)
        grad_gamma = [None] * (self.num_layers)
        grad_beta = [None] * (self.num_layers)

        y_pred, act_h, s, norm_s, mu, var = self.evaluate_classifier(X)

        g_batch = y_pred - y_true

        # gradient of W and b for the last layer
        grad_w[-1] = np.dot(g_batch, act_h[self.num_layers - 1].T) / size + 2 * self.lamda * self.w[self.num_layers]
        grad_b[-1] = np.dot(g_batch, np.ones((self.batch_size,1))) / size
        #grad_b[-1] = np.sum(g_batch, axis=1).reshape(-1,1) / size

        #propagate the gradient to previous layers
        g_batch = np.dot(self.w[-1].T, g_batch)

        # this is gbatch * ind(X_batch[-2] > 0)
        layers_input = act_h[self.num_layers-1]
        h_act_ind = np.zeros(layers_input.shape)
        for k in range(layers_input.shape[0]):
            for j in range(layers_input.shape[1]):
                if layers_input[k,j] > 0:
                    h_act_ind[k, j] = 1
        g_batch = g_batch * h_act_ind

        for l in reversed(range(self.num_layers)):

            if self.batch_norm:
                grad_gamma[l] = np.dot((g_batch * norm_s[l]), np.ones(size)).reshape(-1,1) / size
                grad_beta[l] = np.dot(g_batch, np.ones(size)).reshape(-1,1) / size

                g_batch *= np.dot(self.gamma[l], np.ones((size,1)).T)
                g_batch = self.batchnorm_backward(g_batch, s[l], mu[l], var[l], size)

            if l == 0:
                grad_w[l] = np.dot(g_batch, X.T) / size + 2 * self.lamda*self.w[l]
            else:
                grad_w[l] = np.dot(g_batch, act_h[l-1].T) / size + 2 * self.lamda * self.w[l]

            #grad_b[l] = np.sum(g_batch, axis=1).reshape(-1, 1) / size
            grad_b[l] = np.dot(g_batch, np.ones((size,1))) / size

            if l > 0:
                g_batch = np.dot(self.w[l].T, g_batch)
                # layers_input is the input to layer i (which is the activation of the previous layer)
                layers_input = act_h[l-1]
                h_act_ind = np.zeros(layers_input.shape)

                for k in range(layers_input.shape[0]):
                    for j in range(layers_input.shape[1]):
                        if layers_input[k,j] > 0:
                            h_act_ind[k, j] = 1
                g_batch = g_batch * h_act_ind

        return grad_w, grad_b, grad_beta, grad_gamma

    def batchnorm_backward(self, g_batch, s_batch, mu, var, size):
        sigma_1 = ((var + 1e-6)**(-0.5)).T
        sigma_2 = ((var + 1e-6)**(-1.5)).T
        bigG1 = g_batch * np.outer(sigma_1, np.ones((size,1)))
        bigG2 = g_batch * np.outer(sigma_2, np.ones((size,1)))
        bigD = s_batch - np.outer(mu, np.ones((size,1)))
        c = np.dot((bigG2 * bigD), np.ones((size,1)))
        g_batch = bigG1 - 1/size * np.dot(bigG1, np.ones((size,1)))
        g_batch -= 1 / size * (bigD * np.outer(c, np.ones((size))))

        return g_batch

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