import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_io import load_dataset
from data_utils import normalize_dataset
from subKmeans import sub_kmeans
from matrix_utils import projection_matrix
from sklearn.metrics import normalized_mutual_info_score as nmi
from display import assign_markers, assign_colors


dataset_name = 'fmnist_large'
n_clusters = 10
X, Y = load_dataset(dataset_name)
X = normalize_dataset(X)
for i in range(1):
    C, V, m = sub_kmeans(X, n_clusters)
    Pc = projection_matrix(X.shape[1], m)
    trans = V.T.real
    X_rotated = np.matmul(
        trans[None, :, :], np.transpose(X[:, None, :], [0, 2, 1]))
    X_rotated = X_rotated.squeeze(-1).T
    acc = nmi(Y, C)
    M = assign_markers(C)
    K = assign_colors(Y)
    print('[i] Results')
    print('[*] m: %d' % m)
    print('[*] NMI: %.5f' % acc)
    data_points = zip(X_rotated[0], X_rotated[1], K, M)
    for x_, y_, c_, m_ in data_points:
        plt.scatter(x_, y_, c=c_, marker=m_, s=3)
    plt.title('Fashion MNIST (Large), m={:d}, NMI={:.3f}'.format(m, acc))
    plt.savefig('{}.png'.format(dataset_name), dpi=300)
