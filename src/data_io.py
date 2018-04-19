import numpy as np


def load_dataset(name='seeds'):
    if name == 'seeds':
        with open('../datasets/seeds_dataset.txt') as ds:
            lines = ds.readlines()
            X = []
            Y = []
            for line in lines:
                x = line.split()
                y = x[-1]
                x = x[:-1]
                X.append(x)
                Y.append(y)
            X = np.array(X, dtype=np.float)
            Y = np.array(Y, dtype=np.uint8)
            return X, Y
    if name == 'wine':
        with open('../datasets/wine_dataset.txt') as ds:
            lines = ds.readlines()
            X = []
            Y = []
            for line in lines:
                x = line.strip().split(',')
                y = x[0]
                x = x[1:]
                X.append(x)
                Y.append(y)
            X = np.array(X, dtype=np.float)
            Y = np.array(Y, dtype=np.uint8)
            return X, Y
    if name == 'soy':
        with open('../datasets/soybean_dataset.txt') as ds:
            lines = ds.readlines()
            X = []
            Y = []
            for line in lines:
                x = line.strip().split(',')
                y = x[-1]
                x = x[:-1]
                X.append(x)
                Y.append(int(y[-1]) - 1)
            X = np.array(X, dtype=np.float)
            Y = np.array(Y, dtype=np.uint8)
            return X, Y
    if name == 'hand':
        with open('../datasets/handjob_dataset.txt') as ds:
            lines = ds.readlines()
            X = []
            Y = []
            for line in lines:
                x = line.strip().split(',')
                y = x[0]
                x = x[1:]
                X.append(x)
                Y.append(y)
            X = np.array(X, dtype=np.float)
            Y = np.array(Y, dtype=np.uint8)
            Y = Y if np.min(Y) == 0 else Y - 1
            return X, Y
    if name == 'olive':
        with open('../datasets/olive_dataset.txt') as ds:
            lines = ds.readlines()
            X = []
            Y = []
            for line in lines:
                x = line.strip().split(',')
                y = x[0]
                x = x[1:]
                X.append(x)
                Y.append(y)
            X = np.array(X, dtype=np.float)
            Y = np.array(Y, dtype=np.uint8)
            Y = Y if np.min(Y) == 0 else Y - 1
            return X, Y
    if name == 'symbol':
        with open('../datasets/symbol_dataset.txt') as ds:
            lines = ds.readlines()
            X = []
            Y = []
            for line in lines:
                x = line.strip().split(',')
                y = x[0]
                x = x[1:]
                X.append(x)
                Y.append(y)
            X = np.array(X, dtype=np.float)
            Y = np.array(Y, dtype=np.uint8)
            Y = Y if np.min(Y) == 0 else Y - 1
            return X, Y
    if name == 'plane':
        with open('../datasets/plane_dataset.txt') as ds:
            lines = ds.readlines()
            X = []
            Y = []
            for line in lines:
                x = line.strip().split(',')
                y = x[0]
                x = x[1:]
                X.append(x)
                Y.append(y)
            X = np.array(X, dtype=np.float)
            Y = np.array(Y, dtype=np.uint8)
            Y = Y if np.min(Y) == 0 else Y - 1
            return X, Y
    if name == 'mnist':
        with open('../datasets/mnist_dataset.txt') as ds:
            lines = ds.readlines()
            X = []
            Y = []
            for line in lines:
                x = line.strip().split(',')
                y = x[0]
                x = x[1:]
                X.append(x)
                Y.append(int(float(y)))
            X = np.array(X, dtype=np.float)
            Y = np.array(Y, dtype=np.uint8)
            Y = Y if np.min(Y) == 0 else Y - 1
            return X, Y


if __name__ == '__main__':
    load_dataset()
