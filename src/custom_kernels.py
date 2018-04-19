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