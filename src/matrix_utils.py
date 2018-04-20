from __future__ import print_function
import numpy as np


def projection_matrix(d, m, mode='cpu'):
    Pc = np.zeros([d, m], dtype=np.uint8)
    Pc[:m, :m] = np.eye(m, dtype=np.uint8)
    if mode == 'cpu':
        return Pc
    elif mode == 'gpu':
        import pycuda.gpuarray as gpuarray
        Pc = Pc.astype(np.float32)
        Pc_gpu = gpuarray.to_gpu(Pc)
        return Pc_gpu


def random_V(d, mode='cpu'):
    V = np.random.uniform(0, 1, [d, d])
    if mode == 'cpu':
        Q = np.linalg.qr(V, mode='complete')[0]
        return Q
    elif mode == 'gpu':
        import skcuda.linalg as LA
        import pycuda.gpuarray as gpuarray
        V = V.astype(np.float32)
        V_gpu = gpuarray.to_gpu(V)
        Q_gpu, R_gpu = LA.qr(V_gpu, mode='reduced', lib='cusolver')
        return Q_gpu


def sorted_eig(X, ascending=True, mode='cpu'):
    if mode == 'cpu':
        e_vals, e_vecs = np.linalg.eig(X)
        idx = np.argsort(e_vals)
        if not ascending:
            idx = idx[::-1]
        e_vecs = e_vecs[:, idx]
        e_vals = e_vals[idx]
        return e_vals, e_vecs
    elif mode == 'gpu':
        import skcuda.linalg as LA
        import pycuda.gpuarray as gpuarray
        e_vecs_gpu, e_vals_gpu = LA.eig(X, 'N', 'V', lib='cusolver')
        e_vals = e_vals_gpu.get()
        idx = np.argsort(e_vals)
        V_gpu = gpuarray.empty((X.shape[0], X.shape[1]), np.float32)
        d = X.shape[0]
        for i in range(d):
            V_gpu[i] = e_vecs_gpu[idx[i]]
        V_gpu = LA.transpose(V_gpu)
        return e_vals, V_gpu


if __name__ == '__main__':
    print(projection_matrix(5, 3))
    V = random_V(5)
    print(np.allclose(np.matmul(V, V.T), np.eye(5)))
