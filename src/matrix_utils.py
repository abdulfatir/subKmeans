from __future__ import print_function
import numpy as np
import numpy.linalg as LA


def projection_matrix(d, m):
    Pc = np.zeros([d, m], dtype=np.uint8)
    Pc[:m, :m] = np.eye(m, dtype=np.uint8)
    return Pc


def random_V(d):
    V = np.random.uniform(0, 1, [d, d])
    V = LA.qr(V, mode='complete')[0]
    return V


def sorted_eig(X, ascending=True):
    e_vals, e_vecs = LA.eig(X)
    idx = np.argsort(e_vals)
    if not ascending:
        idx = idx[::-1]
    e_vecs = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs


if __name__ == '__main__':
    print(projection_matrix(5, 3))
    V = random_V(5)
    print(np.allclose(np.matmul(V, V.T), np.eye(5)))
