
import numpy as np

class DataLoader():

    def __init__(self, file_path):
        self.book_string = self.read_book(file_path)
        self.book_chars = list(self.book_string)

        self.idx2char = self.read_unique_chars()
        self.char2idx = self.create_dict()
        self.num_unique = len(self.idx2char)

    def read_book(self, txtfile):
        return open(txtfile, 'r').read()

    def read_unique_chars(self):
        return list(set(self.book_string))

    def create_dict(self):
        char2idx = {}
        for idx, char in enumerate(self.idx2char):
            char2idx[char] = idx

        return char2idx

    def idx2char_list(self, idx_list):
        return [self.idx2char[idx] for idx in idx_list]

    def char2idx_list(self, char_list):
        return [self.char2idx[char] for char in char_list]

    def onehot2string(self, onehot_seq):
        idx_list = [np.where(x==1)[0][0] for x in onehot_seq]
        char_list = self.idx2char_list(idx_list)

        return ''.join(char_list)

    def chars2onehot(self, char_list):
        idx_list = self.char2idx_list(char_list)
        onehot = np.zeros((self.num_unique, len(char_list)))
        for i, idx in enumerate(idx_list):
            onehot[idx, i] = 1

        return onehot

    def get_batch(self, i, seq_length):
        # improve performance by taking from i to i+seq+1 and then getting rid of ony start and end
        X_chars = self.book_chars[i: i+seq_length]
        Y_chars = self.book_chars[i+1: i+seq_length+1]

        X = self.chars2onehot(X_chars)
        Y = self.chars2onehot(Y_chars)

        return X, Y

if __name__ == '__main__':
    d = DataLoader('goblet_book.txt')