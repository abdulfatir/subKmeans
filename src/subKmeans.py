from __future__ import print_function
import time
import numpy as np
from numpy import matmul as MM
from matrix_utils import projection_matrix, random_V, sorted_eig
from display import log_time


def sub_kmeans(X, k):
    start_time = time.time()
    n, d = X.shape
    V = random_V(d)
    m = d / 2
    mu_D = np.mean(X, axis=0, keepdims=True)
    S_D = MM((X - mu_D).T, (X - mu_D))
    mu_is = X[np.random.choice(n, k)]
    MAX_ITER = 100
    itr = 0
    assignment_unchanged = 0
    t1, t2, t3, t4 = [], [], [], []
    while itr < MAX_ITER:

        Pc = projection_matrix(d, m)
        PcV = MM(Pc.T, V.T)[None, :, :]  # 1,m,d
        PcVmu_is = MM(PcV, mu_is[:, :, None])  # k,m,1
        start = time.time()  # n,d,1
        X_trans = MM(PcV.squeeze(0), X.T).T  # n, m
        Mus_trans = PcVmu_is.squeeze(-1)  # k, m
        sq_diff = np.square(X_trans[:, None, :] -
                            Mus_trans[None, :, :])  # n, k, m
        sq_diff = np.sum(sq_diff, axis=-1)
        end = time.time()
        t1.append(end - start)
        start = time.time()
        if itr == 0:
            C = np.argmin(sq_diff, axis=1)
        else:
            Cnew = np.argmin(sq_diff, axis=1)
            points_changed = np.sum(1 - np.equal(C, Cnew).astype(np.uint8))
            if points_changed == 0:
                assignment_unchanged += 1
            if assignment_unchanged >= 5:
                break
            print('[i] Itr %d: %d points changed' % (itr, points_changed))
            C = Cnew
        counts = {i: 0 for i in range(k)}
        mu_is = np.zeros([k, d])
        S_is = np.zeros([k, d, d])
        for i, x in enumerate(X):
            c_id = C[i]
            mu_is[c_id] += x
            counts[c_id] += 1
        end = time.time()
        t2.append(end - start)
        start = time.time()
        mu_is = np.array([mu_is[i] / counts[i] for i in range(k)])
        C_matrics = {i: [] for i in range(k)}
        for i, x in enumerate(X):
            c_id = C[i]
            x_minus_mu_isi = (x - mu_is[c_id]).T[:, None]
            C_matrics[c_id].append(x_minus_mu_isi)
        for ki in C_matrics:
            CX_m = np.array(C_matrics[ki]).squeeze()
            S_is[ki] = MM(CX_m.T, CX_m)
        end = time.time()
        t3.append(end - start)
        start = time.time()
        Evals, Evecs = sorted_eig(np.sum(S_is, axis=0) - S_D)
        end = time.time()
        t4.append(end - start)
        V = Evecs
        maxVal = min(Evals)
        m = np.sum([1 for i in Evals if i / maxVal > 1e-8])
        m = max(1, m)

        itr += 1
    end_time = time.time()
    print('[i] Completed!')
    print('[t] Average Time for Operations')
    log_time('Find assignments', np.mean(t1))
    log_time('Find Means', np.mean(t2))
    log_time('k,d,d Matrices Computation', np.mean(t3))
    log_time('Eigen Decomposition', np.mean(t4))
    log_time('Total Time', end_time - start_time, tabs=0)
    return C, V, m
