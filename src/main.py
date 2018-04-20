import numpy as np
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_io import load_dataset
from data_utils import normalize_dataset
from subKmeans import sub_kmeans
from matrix_utils import projection_matrix
from sklearn.metrics import normalized_mutual_info_score as nmi
from display import assign_markers, assign_colors


def run(dataset_name='seeds', n_clusters=3, mode='cpu'):
    X, Y = load_dataset(dataset_name)
    X = normalize_dataset(X)
    for i in range(1):
        C, V, m = sub_kmeans(X, n_clusters, mode)
        trans = V.T.real
        X_rotated = np.matmul(
            trans[None, :, :], np.transpose(X[:, None, :], [0, 2, 1]))
        X_rotated = X_rotated.squeeze(-1).T
        acc = nmi(Y, C)
        M = assign_markers(C)
        K = assign_colors(Y)
        print('')
        print('[i] Results')
        print('[*] m: %d' % m)
        print('[*] NMI: %.5f' % acc)
        data_points = zip(X_rotated[0], X_rotated[1], K, M)
        for x_, y_, c_, m_ in data_points:
            plt.scatter(x_, y_, c=c_, marker=m_, s=3)
        plt.title('Seeds, m={:d}, NMI={:.3f}'.format(m, acc))
        plt.savefig('{}.png'.format(dataset_name), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str,
                        choices=['seeds', 'wine', 'soy', 'hand', 'olive',
                                 'symbol', 'plane', 'mnist', 'mnist_large', 'fmnist',
                                 'fmnist_large'], help="name of the dataset to use", default='seeds')
    parser.add_argument("-k", type=int, default=3, help="number of classes/clusters")
    parser.add_argument("-mode", type=str, default="cpu", choices=["cpu", "gpu", "gpu_custom"], help="computation mode")
    args = parser.parse_args()
    run(args.d, args.k, args.mode)
