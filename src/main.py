import numpy as np
import matplotlib.pyplot as plt
from data_io import load_dataset
from data_utils import normalize_dataset
from subKmeans import sub_kmeans
from matrix_utils import projection_matrix
from sklearn.metrics import normalized_mutual_info_score as nmi


X, Y = load_dataset('seeds')
X = normalize_dataset(X)
for i in range(1):
    C, V, m = sub_kmeans(X, 3)
    Pc = projection_matrix(X.shape[1], m)
    trans = V.T.real
    X_rotated = np.matmul(
        trans[None, :, :], np.transpose(X[:, None, :], [0, 2, 1]))
    X_rotated = X_rotated.squeeze(-1).T
    acc = nmi(Y, C)
    print('[i] Results')
    print('[*] m: %d' % m)
    print('[*] NMI: %.5f' % acc)
    plt.scatter(X_rotated[0], X_rotated[1], c=C)
    plt.show()
