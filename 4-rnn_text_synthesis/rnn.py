"""
Author: el puto Doppler co
"""

import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader

class RNN:

    def __init__(self):
        self.m = 100            # hidden size
        self.eta = 0.1          # learning rate
        self.seq_length = 25    # length of the input sequences
        self.k = 80             # number of unique characters
        self.sig = 0.1          # weight initial std
        self.h_num = 1e-4       # step size for numerical gradient computations
        self.e = 0              # book pointer
        self.its = 350000       # number of iterations
        self.best_loss = 500    # used to save best model

        self.grads = {}
        self.data = DataLoader('goblet_book.txt')

        self.init_weights()

    def init_weights(self):
        self.weights = {}
    
        self.weights['b'] = np.zeros((self.m, 1))
        self.weights['c'] = np.zeros((self.k, 1))

        self.weights['u'] = np.random.rand(self.m, self.k) * self.sig
        self.weights['v'] = np.random.rand(self.k, self.m) * self.sig
        self.weights['w'] = np.random.rand(self.m, self.m) * self.sig

        self.sum_grad2 = {}
        for key in self.weights:
            self.sum_grad2[key] = np.zeros_like(self.weights[key])

    def train(self):
        smooth_loss = None
        self.h_prev = np.zeros((self.m, 1))

        for i in range(self.its):

            X, Y = self.data.get_batch(self.e, self.seq_length)

            if i % 10000 == 0:
                self.generate(200, X[:, 1])

            self.update_e()
            p, a, h = self.forward(X)
            loss = self.loss(p, Y)
            smooth_loss = self.smooth_loss(smooth_loss, loss)

            if smooth_loss < self.best_loss:
                self.best_loss = smooth_loss
                self.best_weights = self.weights

            self.backward(X, Y, p, a, h)
            self.step()

            if i % 1000 == 0:
                print('# iterations:{}\t smooth_loss={}'.format(i, round(smooth_loss, 3)))
                print('pointer at {}% of the book\n'.format(int(100 * self.e / self.data.book_length)))

        self.weights = self.best_weights
        self.generate(1000, X[:, 1])

    def forward(self, X):
        p = np.zeros((X.shape[1], self.k))
        a = np.zeros((X.shape[1], self.m))
        h = np.zeros((X.shape[1], self.m))

        for t in range(X.shape[1]):
            x_t = X[:,t].reshape((self.k, 1))
            a_t = np.dot(self.weights['w'], self.h_prev) + np.dot(self.weights['u'], x_t) + self.weights['b']
            h_t = np.tanh(a_t)
            o_t = np.dot(self.weights['v'], h_t) + self.weights['c']
            p_t = self.softmax(o_t)

            a[t] = a_t.reshape(self.m)
            h[t] = h_t.reshape(self.m)
            p[t] = p_t.reshape(self.k)

            self.h_prev = h_t

        return p, a, h

    def backward(self, X, Y, p, a, h):
        grad_o = np.zeros((self.seq_length, self.k))
        for t in range(self.seq_length):
            y_t = Y[:, t].reshape(self.k, 1)
            p_t = p[t].reshape(self.k, 1)
            grad_o[t] = (p_t - y_t).reshape(self.k)

        grad_v = np.zeros((self.k, self.m))
        for t in range(self.seq_length):
            grad_v += np.dot(grad_o[t].reshape(self.k, 1), h[t].reshape(1, self.m))

        grad_h = np.zeros((self.seq_length, self.m))
        grad_a = np.zeros((self.seq_length, self.m))

        grad_h[-1] = np.dot(grad_o[-1], self.weights['v'])
        diag = np.diag(1 - np.tanh(a[-1])**2)
        grad_a[-1] = np.dot(grad_h[-1], diag)

        for t in reversed(range(self.seq_length - 1)):
            grad_h[t] = np.dot(grad_o[t], self.weights['v']) + np.dot(grad_a[t+1], self.weights['w'])
            diag = np.diag(1 - np.tanh(a[t])**2)
            grad_a[t] = np.dot(grad_h[t], diag)

        grad_c = grad_o.sum(axis = 0).reshape(self.k, 1)
        grad_b = grad_a.sum(axis = 0).reshape(self.m, 1)

        grad_w = np.zeros((self.m, self.m))
        for t in range(1, self.seq_length):
            grad_w += np.outer(grad_a[t].reshape(self.m,1), h[t-1])

        grad_u = np.zeros((self.m, self.k))
        for t in range(self.seq_length):
            xt = X[:,t].reshape(self.k, 1)
            grad_u += np.dot(grad_a[t].reshape(self.m,1), xt.T)

        self.grads['b'] = np.clip(grad_b, -5, 5)
        self.grads['c'] = np.clip(grad_c, -5, 5)
        self.grads['u'] = np.clip(grad_u, -5, 5)
        self.grads['v'] = np.clip(grad_v, -5, 5)
        self.grads['w'] = np.clip(grad_w, -5, 5)

    def loss(self, p, y):
        loss = 0
        for t in range(p.shape[0]):
            loss -= np.log(np.dot(y[:, t].T, p[t]))

        return loss

    def smooth_loss(self, smooth_loss, new_loss):
        if smooth_loss == None:
            smooth_loss = new_loss
        else:
            smooth_loss = 0.999 * smooth_loss + 0.001 * new_loss

        return smooth_loss

    def step(self):
        for key in self.weights:
            self.sum_grad2[key] += np.square(self.grads[key])
            self.weights[key] -= self.grads[key] * self.eta / np.sqrt(self.sum_grad2[key] + np.finfo(float).eps)

    def generate(self, n, x0):
        p, a, h = self.synthesize(n, x0)
        onehot_seq = np.array([self.sample_char(pt) for pt in p])

        print('\n================= generated text =================')
        print(self.data.onehot2string(onehot_seq))
        print('==================================================\n')

    def synthesize(self, n, x0):
        p = np.zeros((n, self.k))
        a = np.zeros((n, self.m))
        h = np.zeros((n, self.m))

        self.x_prev = x0.reshape((self.k, 1))
        h_t = self.h_prev

        for t in range(n):
            x_t = self.x_prev
            a_t = np.dot(self.weights['w'], h_t)
            a_t += np.dot(self.weights['u'], x_t)
            a_t += self.weights['b']
            h_t = np.tanh(a_t)
            o_t = np.dot(self.weights['v'], h_t) + self.weights['c']
            p_t = self.softmax(o_t)

            a[t] = a_t.reshape(self.m)
            h[t] = h_t.reshape(self.m)
            p[t] = p_t.reshape(self.k)

            self.x_prev = self.sample_char(p[t]).reshape((self.k, 1))

        return p, a, h

    def sample_char(self, p):
        idx_sample = np.random.choice(self.k, 1, p=p)

        onehot_sample = np.zeros(self.k)
        onehot_sample[idx_sample] = 1

        return onehot_sample

    def update_e(self):
        if self.e + self.seq_length + 25 >= self.data.book_length:
            print('reseting book pointer')
            self.e = 0
            self.h_prev = np.zeros((self.m, 1))
        else:
            self.e += self.seq_length

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def mean_relative_error(self, x, y):
        return np.mean(abs(x - y) / np.maximum(abs(x) + abs(y), np.finfo(float).eps))

    def check_gradients(self):
        X, Y = self.data.get_batch(0, self.seq_length)
        p, a, h = self.forward(X)
        self.backward(X, Y, p, a, h)

        num_grads = {}
        for key in self.grads:
            num_grads[key] = self.num_backward(key, X, Y)
            mre = self.mean_relative_error(self.grads[key], num_grads[key])

            print("Mean Relative Error for grad_" + key + ": " + str(mre))

    def num_backward(self, key, X, Y):
      num_grad = np.zeros(self.grads[key].shape)
      if key == 'b' or key == 'c':
        for i in range(self.weights[key].shape[0]):
          self.weights[key][i] -= self.h_num
          p1, _, _ = self.forward(X)
          l1 = self.loss(p1, Y)
          self.weights[key][i] += 2*self.h_num
          p2, _, _ = self.forward(X)
          l2 = self.loss(p2, Y)
          num_grad[i] = (l2-l1) / (2*self.h_num)
          self.weights[key][i] -= self.h_num
      else:
        for i in range(self.weights[key].shape[0]):  
          for j in range(self.weights[key].shape[1]):
            self.weights[key][i][j] -= self.h_num
            p1, _, _ = self.forward(X)
            l1 = self.loss(p1, Y)
            self.weights[key][i][j] += 2*self.h_num
            p2, _, _ = self.forward(X)
            l2 = self.loss(p2, Y)
            num_grad[i][j] = (l2-l1) / (2*self.h_num)   
            self.weights[key][i][j] -= self.h_num
      return num_grad

if __name__ == '__main__':
    rnn = RNN()
    rnn.train()