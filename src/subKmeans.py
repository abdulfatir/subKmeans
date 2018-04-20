from __future__ import print_function
import time
import numpy as np
from numpy import matmul as MM
from matrix_utils import projection_matrix, random_V, sorted_eig
from display import log_time


def sub_kmeans(X, k, mode='cpu'):
    if mode == 'cpu':
        return _sub_kmeans_cpu(X, k)
    elif mode == 'gpu':
        return _sub_kmeans_gpu(X, k)
    elif mode == 'gpu_custom':
        return _sub_kmeans_gpu_custom(X, k)


def _sub_kmeans_gpu_custom(X, k):
    import skcuda
    import skcuda.linalg as LA
    import pycuda.gpuarray as gpuarray
    import custom_kernels as CC
    LA.init()

    n, d = X.shape
    V_gpu, QR_time = random_V(d, mode='gpu')

    m = d / 2

    X_gpu = gpuarray.to_gpu(X)
    mu_D_gpu = CC.column_mean(X_gpu)
    sub_gpu = skcuda.misc.subtract(X_gpu, mu_D_gpu)
    sub_gpu_T = LA.transpose(sub_gpu)
    S_D_gpu = CC.matmul(sub_gpu_T, sub_gpu)
    mu_is_gpu = gpuarray.to_gpu(X[np.random.choice(n, k)])
    itr = 1
    assignment_unchanged = 0
    C_gpu = None

    while True:
        Pc_gpu = projection_matrix(d, m, mode='gpu')
        PcV_gpu = LA.dot(Pc_gpu, V_gpu, transa='T', transb='T')
        PcVmu_is_gpu = gpuarray.empty((k, m), dtype=np.float32)

        for i in range(k):
            PcVmu_is_gpu[i] = LA.dot(PcV_gpu, mu_is_gpu[i][:, None]).ravel()

        global_temp = LA.dot(X_gpu, PcV_gpu, transb='T')
        if itr % 2 == 0:
            C_old = C_gpu.get()
        C_gpu = CC.argmin_mu_diff(global_temp, PcVmu_is_gpu)
        if itr % 2 == 0:
            Cnew = C_gpu.get()
            points_changed = np.sum(1 - np.equal(C_old, Cnew).astype(np.uint8))
            if points_changed == 0:
                assignment_unchanged += 1
            if assignment_unchanged >= 2:
                break
            print('[i] Itr %d: %d points changed' % (itr, points_changed))

        C = C_gpu.get()
        counts = {i: 0 for i in range(k)}

        for i in xrange(n):
            C_id = np.int(C[i])
            counts[C_id] += 1
        maxv = np.max(counts.values())
        storage = np.zeros((k, np.int(maxv), d)).astype(np.float32)

        counter = np.zeros(k, dtype=np.uint32)  # k
        for i in range(n):
            C_id = np.int(C[i])
            storage[C_id, np.int(counter[C_id]), :] = X[i].ravel()
            counter[C_id] += 1

        storage_gpu = gpuarray.to_gpu(storage)

        mu_is_gpu = CC.sum_axis2(storage_gpu)
        counter_gpu = gpuarray.to_gpu(counter)[:, None]

        mu_is_gpu = skcuda.misc.divide(
            mu_is_gpu, counter_gpu.astype(np.float32))
        S_is_gpu = gpuarray.zeros((k, d, d), dtype=np.float32)  # k,d,d

        for i in range(k):
            storage_gpu[i] = skcuda.misc.subtract(storage_gpu[i], mu_is_gpu[i])
            curr_cluster_points = storage_gpu[i,
                                              :np.int(counter[i]), :]  # |k|,d
            S_is_gpu[i] = LA.dot(curr_cluster_points,
                                 curr_cluster_points, transa='T')

        S_is_sum_gpu = S_is_gpu.reshape((k, d * d))
        S_is_sum_gpu = skcuda.misc.sum(S_is_sum_gpu, axis=0, keepdims=True)
        S_is_sum_gpu = S_is_sum_gpu.reshape((d, d))

        S_is_diff_gpu = skcuda.misc.subtract(S_is_sum_gpu, S_D_gpu)

        w, V_gpu = sorted_eig(S_is_diff_gpu, mode='gpu')

        maxVal = min(w)
        m = np.sum([1 for i in w if i / maxVal > 1e-3])
        m = max(1, m)

        itr += 1


def _sub_kmeans_gpu(X, k):
    import skcuda
    import skcuda.linalg as LA
    import pycuda.gpuarray as gpuarray
    LA.init()

    n, d = X.shape
    V_gpu = random_V(d, mode='gpu')
    m = d / 2
    X_gpu = gpuarray.to_gpu(X)
    mu_D_gpu = skcuda.misc.mean(X_gpu, axis=0, keepdims=True)
    sub_gpu = skcuda.misc.subtract(X_gpu, mu_D_gpu)
    S_D_gpu = LA.dot(sub_gpu, sub_gpu, transa='T')
    mu_is_gpu = gpuarray.to_gpu(X[np.random.choice(n, k)])
    itr = 1
    assignment_unchanged = 0
    C_gpu = None
    while True:
        Pc_gpu = projection_matrix(d, m, mode='gpu')
        PcV_gpu = LA.dot(Pc_gpu, V_gpu, transa='T', transb='T')
        PcVmu_is_gpu = gpuarray.empty((k, m), dtype=np.float32)

        for i in range(k):
            PcVmu_is_gpu[i] = LA.dot(PcV_gpu, mu_is_gpu[i][:, None]).ravel()

        global_temp = LA.dot(X_gpu, PcV_gpu, transb='T')
        if itr % 2 == 0:
            C_old = C_gpu.get()
        X_transformed_gpu = gpuarray.empty(
            (n, k, m), dtype=np.float32)
        for i in xrange(n):
            temp = global_temp[i]
            X_transformed_gpu[i] = skcuda.misc.subtract(
                PcVmu_is_gpu, temp)

        X_transformed_squared_gpu = LA.multiply(
            X_transformed_gpu, X_transformed_gpu)
        X_transformed_squared_gpu = X_transformed_squared_gpu.reshape(
            (n * k, m))
        X_transformed_sum_gpu = skcuda.misc.sum(
            X_transformed_squared_gpu, axis=-1, keepdims=True)
        X_transformed_sum_gpu = X_transformed_sum_gpu.reshape((n, k))
        C_gpu = skcuda.misc.argmin(
            X_transformed_sum_gpu, axis=1)
        if itr % 2 == 0:
            Cnew = C_gpu.get()
            points_changed = np.sum(1 - np.equal(C_old, Cnew).astype(np.uint8))
            if points_changed == 0:
                assignment_unchanged += 1
            if assignment_unchanged >= 2:
                break
            print('[i] Itr %d: %d points changed' % (itr, points_changed))
        C = C_gpu.get()
        counts = {i: 0 for i in range(k)}
        mu_is = np.zeros((k, d)).astype(np.float32)
        for i in range(n):
            C_id = np.int(C[i])
            mu_is[C_id] += X[i]
            counts[C_id] += 1

        mu_is = np.array([mu_is[i] / counts[i] for i in range(k)])
        mu_is_gpu = gpuarray.to_gpu(mu_is)
        S_is_gpu = gpuarray.zeros((k, d, d), dtype=np.float32)

        maxv = np.max(counts.values())
        storage = np.empty((k, np.int(maxv), d)).astype(np.float32)
        counter = np.zeros(k, dtype=np.uint32)

        for i in range(n):
            C_id = np.int(C[i])
            X_minus_mu_isi = (X[i] - mu_is[C_id])[:, None]
            storage[C_id, np.int(counter[C_id]), :] = X_minus_mu_isi.ravel()
            counter[C_id] += 1

        storage_gpu = gpuarray.to_gpu(storage)
        for i in range(k):
            curr_cluster_points = storage_gpu[i,
                                              :np.int(counter[i]), :]
            S_is_gpu[i] = LA.dot(curr_cluster_points,
                                 curr_cluster_points, transa='T')

        S_is_sum_gpu = S_is_gpu.reshape((k, d * d))
        S_is_sum_gpu = skcuda.misc.sum(S_is_sum_gpu, axis=0, keepdims=True)
        S_is_sum_gpu = S_is_sum_gpu.reshape((d, d))

        S_is_diff_gpu = skcuda.misc.subtract(S_is_sum_gpu, S_D_gpu)

        w, V_gpu = sorted_eig(S_is_diff_gpu, mode='gpu')

        maxVal = min(w)
        m = np.sum([1 for i in w if i / maxVal > 1e-3])
        m = max(1, m)

        itr += 1


def _sub_kmeans_cpu(X, k):
    start_time = time.time()
    n, d = X.shape
    V = random_V(d)
    m = d // 2
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
            if assignment_unchanged >= 2:
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
