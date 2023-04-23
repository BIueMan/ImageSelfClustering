import numpy as np

def nearest_neighbors(splited_image:np.array)->np.array:
    m, n, i, j, k = splited_image.shape
    arr_flat = splited_image.reshape(m*n, i*j*3)
    dists = np.sqrt(((arr_flat[:, None] - arr_flat) ** 2).sum(axis=2))
    return dists

def sigma_dist(dist_mat:np.array, splited_image_shape, k=5)->np.array:
    m, n, _, _, _ = splited_image_shape
    sigma = np.zeros(m*n)
    for ii in range(m*n):
        closest_points = np.argsort(dist_mat[ii])[:k]
        sigma[ii] = dist_mat[ii, closest_points[k-1]]
    return sigma.reshape(m,n)/np.max(sigma)

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
    norm_part = np.sum(np.mean(x - y))**2 / norm_scale
    return np.exp(-norm_part / sigma**2)