"""
Author: el puto Doppler co
"""

import numpy as np
import matplotlib.pyplot as plt

class ANN:

    def __init__(self):
        self.k = 10
        self.gauss_mean = 0
        self.gauss_std = 1e-1
        self.d = 3072

        self.lamda = 0.005
        self.batch_size = 100
        self.epochs = 300
        self.lr = 1e-5
        self.jitter = False
        self.batch_norm = False
        self.he_init = True

        self.lr_min = 1e-5
        self.lr_max = 1e-1
        self.ns = -1
        self.num_cycles = 2
        self.alpha = 0.7

        self.h_nodes = [50, 30, 20, 20, 10, 10, 10, 10]
        self.num_layers = len(self.h_nodes)

        self.plot = True

        self.init_weights_and_biases()
        self.init_gamma_and_beta()

    def init_weights_and_biases(self):
        """
        self.w matrix shape structure: [(d, h_1), (h_1, h_2), ..., (h_n-1, h_n), (h_n, k)]
        """

        self.w = [None] * (self.num_layers + 1)
        self.b = [None] * (self.num_layers + 1)

        if self.he_init:
            sigma_weights_w = np.zeros(self.num_layers + 1)
            sigma_weights_w[0] = np.sqrt(2 / self.d)
            for i in range(1, self.num_layers + 1):
                sigma_weights_w[i] = np.sqrt(2 / self.h_nodes[i-1])

            self.w[0] = np.random.normal(self.gauss_mean, sigma_weights_w[0], (self.h_nodes[0], self.d))
            for i in range(1, self.num_layers):
                self.w[i] = np.random.normal(self.gauss_mean, sigma_weights_w[i], (self.h_nodes[i], self.h_nodes[i-1]))
            self.w[self.num_layers] = np.random.normal(self.gauss_mean, sigma_weights_w[self.num_layers], (self.k, self.h_nodes[self.num_layers-1]))
        
        else:
            self.w[0] = np.random.normal(self.gauss_mean, self.gauss_std, (self.h_nodes[0], self.d))
            for i in range(1, self.num_layers):
                self.w[i] = np.random.normal(self.gauss_mean, self.gauss_std, (self.h_nodes[i], self.h_nodes[i-1]))
            self.w[self.num_layers] = np.random.normal(self.gauss_mean, self.gauss_std, (self.k, self.h_nodes[self.num_layers-1]))

        for i in range(self.num_layers):
            self.b[i] = np.zeros((self.h_nodes[i], 1))
        self.b[self.num_layers] = np.zeros((self.k, 1))

    def init_gamma_and_beta(self):
        self.gamma = [None] * (self.num_layers)
        self.gamma = [np.random.normal(self.gauss_mean, np.sqrt(2/self.h_nodes[i]), (self.h_nodes[i], 1)) for i in range(self.num_layers)]

        self.beta = [None] * (self.num_layers + 1)
        for i in range(self.num_layers):
            self.beta[i] = np.zeros((self.h_nodes[i], 1))
        self.beta[self.num_layers] = np.zeros((self.k, 1))

        self.mu_avg = None
        self.var_avg = None

    def train(self, x_train, y_train, x_val, y_val, x_test, y_test):
        num_batches = int(x_train.shape[1] / self.batch_size)

        self.ns = 5 * 45000 / self.batch_size

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
                    x_batch += 0.15 * np.random.randn(*x_batch.shape)

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

        s[0] = np.dot(self.w[0], X) + self.b[0]

        if self.batch_norm:
            mu[0] = np.mean(s[0], axis = 1)
            var[0] = np.var(s[0], axis = 1)
            norm_s[0] = self.batch_normalize(s[0], mu[0], var[0])
            final_s[0] = self.gamma[0] * norm_s[0] + self.beta[0]

            act_h[0] = np.maximum(0, final_s[0])
        else:
            act_h[0] = np.maximum(0, s[0])

        for i in range(1, self.num_layers):
            s[i] = np.dot(self.w[i], act_h[i-1]) + self.b[i]

            if self.batch_norm:
                mu[i] = np.mean(s[i], axis = 1)
                var[i] = np.var(s[i], axis = 1)
                norm_s[i] = self.batch_normalize(s[i], mu[i], var[i])
                final_s[i] = self.gamma[i] * norm_s[i] + self.beta[i]

                act_h[i] = np.maximum(0, final_s[i])
            else:
                act_h[i] = np.maximum(0, s[i])

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

        s[0] = np.dot(self.w[0], X) + self.b[0]

        if self.batch_norm:
            norm_s[0] = self.batch_normalize(s[0], self.mu_avg[0], self.var_avg[0])
            final_s[0] = self.gamma[0] * norm_s[0] + self.beta[0]

            act_h[0] = np.maximum(0, final_s[0])
        else:
            act_h[0] = np.maximum(0, s[0])

        for i in range(1, self.num_layers):
            s[i] = np.dot(self.w[i], act_h[i-1]) + self.b[i]

            if self.batch_norm:
                norm_s[i] = self.batch_normalize(s[i], self.mu_avg[i], self.var_avg[i])
                final_s[i] = self.gamma[i] * norm_s[i] + self.beta[i]

                act_h[i] = np.maximum(0, final_s[i])
            else:
                act_h[i] = np.maximum(0, s[i])

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

        grad_w[-1] = np.dot(g_batch, act_h[self.num_layers - 1].T) / size + 2 * self.lamda * self.w[self.num_layers]
        grad_b[-1] = np.dot(g_batch, np.ones((self.batch_size,1))) / size

        g_batch = np.dot(self.w[-1].T, g_batch)

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

            grad_b[l] = np.dot(g_batch, np.ones((size,1))) / size

            if l > 0:
                g_batch = np.dot(self.w[l].T, g_batch)
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
        g1 = g_batch * np.outer(sigma_1, np.ones((size,1)))
        g2 = g_batch * np.outer(sigma_2, np.ones((size,1)))
        D = s_batch - np.outer(mu, np.ones((size,1)))
        c = np.dot((g2 * D), np.ones((size,1)))
        g_batch = g1 - 1/size * np.dot(g1, np.ones((size,1)))
        g_batch -= 1 / size * (D * np.outer(c, np.ones((size))))

        return g_batch

    def check_gradients(self, X, y_true):
        grad_w, _, _, _ = self.compute_gradients(X, y_true)
        grad_w_num, _, _, _ = self.compute_numerical_gradients(X, y_true, h=1e-6)

        grad_w = grad_w[0]
        grad_w_num = grad_w_num[0]

        mean_re = np.mean(abs(grad_w - grad_w_num) / np.maximum(abs(grad_w) + abs(grad_w_num), np.finfo(float).eps))

        print("Mean Squared Error:\t{:.2e}".format(np.mean((grad_w - grad_w_num) ** 2)))
        print("Mean Relative Error:\t{:.2e}".format(mean_re))

    def compute_numerical_gradients(self, x, y, h):
        grad_w = []
        grad_b = []
        grad_gamma = []
        grad_beta = []

        for i in range(self.num_layers):
            print('Computing numerical gradients... Layer ', str(i + 1))
            grad_w.append(np.zeros_like(self.w[i]))
            grad_b.append(np.zeros_like(self.b[i]))

            for j in range(len(self.b[i])):
                self.b[i][j] -= h
                c1 = self.compute_cost(x, y)
                self.b[i][j] += 2 * h
                c2 = self.compute_cost(x, y)

                grad_b[i][j] = (c2 - c1) / (2 * h)
                self.b[i][j] -= h

            for j in range(self.w[i].shape[0]):
                for l in range(self.w[i].shape[1]):
                    self.w[i][j, l] -= h
                    c1 = self.compute_cost(x, y)
                    self.w[i][j, l] += 2 * h
                    c2 = self.compute_cost(x, y)

                    grad_w[i][j, l] = (c2 - c1) / (2 * h)
                    self.w[i][j, l] -= h

        for i in range(self.num_layers - 1):
            grad_gamma.append(np.zeros_like(self.gamma[i]))
            grad_beta.append(np.zeros_like(self.beta[i]))

            for j in range(len(self.gamma[i])):
                self.gamma[i][j] -= h
                c1 = self.compute_cost(x, y)
                self.gamma[i][j] += 2 * h
                c2 = self.compute_cost(x, y)

                grad_gamma[i][j] = (c2 - c1) / (2 * h)
                self.gamma[i][j] -= h

            for j in range(len(self.beta[i])):
                self.beta[i][j] -= h
                c1 = self.compute_cost(x, y)
                self.beta[i][j] += 2 * h
                c2 = self.compute_cost(x, y)

                grad_beta[i][j] = (c2 - c1) / (2 * h)
                self.beta[i][j] -= h

        return grad_w, grad_b, grad_gamma, grad_beta