import numpy as np

def nearest_neighbors(splited_image:np.array)->np.array:
    m, n, i, j, k = splited_image.shape
    arr_flat = splited_image.reshape(m*n, i*j, k)
    dists = np.sqrt(((arr_flat[:, None] - arr_flat) ** 2).sum(axis=2))
    return dists.reshape(m,n)

def simga_dist(dist_mat:np.array, k=20)->np.array:
    nn_indices = np.argpartition(dist_mat, kth=k, axis=1)[:, :k]
    sigma = np.take_along_axis(dist_mat, nn_indices, axis=1)
    return sigma

def weight(x:np.array, y:np.array, *args)->np.array:
    if x.shape != y.shape:
        raise Exception("miss math input imags shapes")
    
    [sigma, norm_scale] = args
    # norm image
    if np.max(x) != 0:
        x = x/np.max(x)
    if np.max(y) != 0:
        y = y/np.max(y)
    # get weight
    norm_part = np.linalg.norm(x - y)**2 / norm_scale
    return np.exp(-norm_part / sigma**2)

# loss funcito in DL