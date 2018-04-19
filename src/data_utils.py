import numpy as np


def normalize_dataset(X):
    n, d = X.shape
    for i in range(d):
        mu = np.mean(X[:, i])
        std = np.std(X[:, i]) + 1e-10
        X[:, i] = (X[:, i] - mu) / std
    return X


if __name__ == '__main__':
    normalize_dataset(np.random.uniform(-10, 10, (1000, 20)))
