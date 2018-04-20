# Fast Subspace Clustering
## A GPU Implementation of *subKMeans*<sup>[1]</sup>

This is a CPU and GPU implementation of the KDD 2017 paper **Towards an Optimal Subspace for K-Means**<sup>[1]</sup>. 
The CPU implementation (`mode=cpu`) is written in Numpy. 
There are 2 GPU implementations:    
1. Using only PyCUDA and scikit-cuda API (`mode=gpu`)
2. Using PyCUDA with custom kernels optimized for this algorithm (`mode=gpu_custom`)

### Dependencies

* Numpy
* PyCUDA: `pip install pycuda`
* scikit-cuda: Install from source as described [here](https://github.com/lebedov/scikit-cuda/blob/master/docs/source/install.rst).
We tested using [commit #249538c](https://github.com/lebedov/scikit-cuda/commit/249538c95e68d891e2477d93eeff57941c99eb93).
* Matplotlib (for plots)
* scikit-learn (for computing NMI score)

Note: We tested this implementation only on Python 2. There are some issues with the GPU version on Python 3.

### Usage

Go to `src/`
```bash
python main.py -d=<dataset_name> -k=<number_of_clusters> -mode=<mode>
```
For help: `python main.py -h`
3 available modes: `cpu`, `gpu`, `gpu_custom`

#### Example Usage
`python main.py -d=wine -k=3 -mode=cpu`
##### Sample Output
```
[i] Itr 1: 24 points changed
[i] Itr 2: 7 points changed
[i] Itr 3: 7 points changed
[i] Itr 4: 2 points changed
[i] Itr 5: 1 points changed
[i] Itr 6: 0 points changed

[i] Results
[*] m: 2
[*] NMI: 0.87590
```



### References
[1] *Mautz et. al.* Towards an Optimal Subspace for K-Means