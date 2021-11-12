import numpy as np

def rms(xx, yy):
    dim = len(xx)
    return np.sqrt(np.sum((xx - yy) ** 2) / dim)

def shifting_euclidean_distance(xx, yy, offset=2):
    dist = rms(xx, yy)
    for i in range(1, offset + 1):
        mis_dist = rms(xx[i:], yy[:-i]) + rms(yy[i:], xx[:-i])
        dist += mis_dist
    return dist / (offset * 2 + 1)

def distance_matrix(X):
    """
    Build distance matrix proceeding to spectral clustering.
    :param X: input 2D tensor
    :return: N x N distance matrix
    """
    (N, D) = X.shape
    affinity_matrix = np.zeros(shape=(N, N))

    for i in range(N):
        for j in range(i + 1, N):
            s_ij = shifting_euclidean_distance(X[i], X[j])
            affinity_matrix[i, j] = s_ij
            affinity_matrix[j, i] = s_ij
        print('{}/{}'.format(i + 1, N))

    return affinity_matrix
