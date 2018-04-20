from __future__ import print_function
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


mod = None


def init():
    global mod
    with open('custom_kernels.cu', 'r') as c:
        cuda_source = c.read()
    mod = SourceModule(cuda_source)


def matmul(a_gpu, b_gpu, c_gpu):
    SHAPE_A = np.array(a_gpu.shape).astype(np.uint32)
    SHAPE_B = np.array(b_gpu.shape).astype(np.uint32)
    SHAPE_A_gpu = cuda.mem_alloc(SHAPE_A.nbytes)
    cuda.memcpy_htod(SHAPE_A_gpu, SHAPE_A)
    SHAPE_B_gpu = cuda.mem_alloc(SHAPE_B.nbytes)
    cuda.memcpy_htod(SHAPE_B_gpu, SHAPE_B)
    BLOCK_DIMX = 32
    BLOCK_DIMY = 32
    GRID_DIMX = int(np.ceil(SHAPE_B[1] / float(BLOCK_DIMX)))
    GRID_DIMY = int(np.ceil(SHAPE_A[0] / float(BLOCK_DIMY)))
    func = mod.get_function("matmul")
    func(a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, SHAPE_A_gpu, SHAPE_B_gpu, block=(
        BLOCK_DIMX, BLOCK_DIMY, 1), grid=(GRID_DIMX, GRID_DIMY, 1))
    # return c_gpu


def column_mean(a_gpu):
    ss = time.time()
    SHAPE_A = np.array(a_gpu.shape).astype(np.uint32)
    SHAPE_A_gpu = cuda.mem_alloc(SHAPE_A.nbytes)
    cuda.memcpy_htod(SHAPE_A_gpu, SHAPE_A)
    mean_a_gpu = gpuarray.empty((1, SHAPE_A[1]), np.float32)
    func = mod.get_function("column_mean")
    BLOCK_DIMX = 1024
    GRID_DIMX = int(np.ceil(a_gpu.shape[1] / float(BLOCK_DIMX)))
    func(a_gpu.gpudata, mean_a_gpu.gpudata, SHAPE_A_gpu, block=(
        BLOCK_DIMX, 1, 1), grid=(GRID_DIMX, 1, 1))
    print('Func:', time.time() - ss)
    return mean_a_gpu


def argmin_mu_diff(a_gpu, mu_gpu):
    ss = time.time()
    SHAPE_A_gpu = gpuarray.to_gpu(np.asarray(
        [a_gpu.shape[0], a_gpu.shape[1]]).astype(np.uint32))
    SHAPE_MU_gpu = gpuarray.to_gpu(np.asarray(
        [mu_gpu.shape[0], mu_gpu.shape[1]]).astype(np.uint32))
    C_gpu = gpuarray.empty((a_gpu.shape[0]), dtype=np.uint32)
    func = mod.get_function("argmin_mu_diff")
    BLOCK_DIMX = 1024
    GRID_DIMX = int(np.ceil(a_gpu.shape[0] / float(BLOCK_DIMX)))
    func(a_gpu.gpudata, mu_gpu.gpudata, SHAPE_A_gpu, SHAPE_MU_gpu, C_gpu,
         block=(BLOCK_DIMX, 1, 1), grid=(GRID_DIMX, 1, 1))
    print('Func:', time.time() - ss)
    return C_gpu


def sum_axis2(a_gpu):
    ss = time.time()
    BLOCK_DIMX = 32
    BLOCK_DIMY = 32
    GRID_DIMX = int(np.ceil(a_gpu.shape[2] / float(BLOCK_DIMX)))
    GRID_DIMY = int(np.ceil(a_gpu.shape[0] / float(BLOCK_DIMY)))
    func = mod.get_function("sum_axis2")
    SHAPE_A_gpu = gpuarray.to_gpu(np.asarray(
        [a_gpu.shape[0], a_gpu.shape[1], a_gpu.shape[2]]).astype(np.uint32))
    sum_gpu = gpuarray.empty(
        (a_gpu.shape[0], a_gpu.shape[2]), dtype=np.float32)
    func(a_gpu.gpudata, SHAPE_A_gpu, sum_gpu.gpudata, block=(BLOCK_DIMX,
                                                             BLOCK_DIMY, 1), grid=(GRID_DIMX, GRID_DIMY, 1))
    print('Func:', time.time() - ss)
    return sum_gpu


def _random_unif_ndarray(*shape):
    return np.random.uniform(-2, 2, shape).astype(np.float32)


def _test_matmul():
    print('[*] Testing matmul')
    a = _random_unif_ndarray(125000, 1980)
    b = _random_unif_ndarray(1980, 10500)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_cuda = gpuarray.empty((a_gpu.shape[0], b_gpu.shape[1]), np.float32)
    t1 = time.time()
    matmul(a_gpu, b_gpu, c_cuda)
    t2 = time.time()
    c_np = np.matmul(a, b)
    t3 = time.time()
    status = 'PASSED' if np.allclose(
        c_cuda.get(), c_np, atol=1e-2) else 'FAILED'
    print('Test: %s, CUDA: %.4fs, Numpy: %.4fs' % (status, t2 - t1, t3 - t2))


def _test_column_mean():
    print('[*] Testing column_mean')
    a = _random_unif_ndarray(10500, 19800)
    a_gpu = gpuarray.to_gpu(a)
    t1 = time.time()
    c_cuda = column_mean(a_gpu)
    t2 = time.time()
    c_np = np.mean(a, axis=0)
    t3 = time.time()
    status = 'PASSED' if np.allclose(
        c_cuda.get(), c_np, atol=1e-2) else 'FAILED'
    print('Test: %s, CUDA: %.4fs, Numpy: %.4fs' % (status, t2 - t1, t3 - t2))


def _test_argmin_mu_diff():
    print('[*] Testing argmin_mu_diff')
    a = _random_unif_ndarray(19800, 105)
    b = _random_unif_ndarray(100, 105)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    t1 = time.time()
    c_cuda = argmin_mu_diff(a_gpu, b_gpu)
    t2 = time.time()
    c_np = np.argmin(
        np.sum(np.square(a[:, None, :] - b[None, :, :]), axis=-1), axis=-1)
    t3 = time.time()
    status = 'PASSED' if np.allclose(
        c_cuda.get(), c_np, atol=1e-2) else 'FAILED'
    print('Test: %s, CUDA: %.4fs, Numpy: %.4fs' % (status, t2 - t1, t3 - t2))


def _test_sum_axis2():
    print('[*] Testing sum_axis2')
    a = _random_unif_ndarray(10, 19800, 198)
    a_gpu = gpuarray.to_gpu(a)
    t1 = time.time()
    c_cuda = sum_axis2(a_gpu)
    t2 = time.time()
    c_np = np.sum(a, axis=1)
    t3 = time.time()
    status = 'PASSED' if np.allclose(
        c_cuda.get(), c_np, atol=1e-2) else 'FAILED'
    print('Test: %s, CUDA: %.4fs, Numpy: %.4fs' % (status, t2 - t1, t3 - t2))


if __name__ == '__main__':
    init()
    start = cuda.Event()
    end = cuda.Event()
    # Test matmul
    _test_matmul()
    # Test column_mean
    _test_column_mean()
    # Test cluster assignment step
    _test_argmin_mu_diff()
    # Test sum over axis 2 (1-indexed)
    _test_sum_axis2()
