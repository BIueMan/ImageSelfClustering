import numpy as np
import cv2
from scipy.sparse.csgraph import laplacian
import itertools

def weight_idex(img1:np.array, img2:np.array, *args)->np.array:
    if img1.shape != img2.shape:
        raise Exception("input imags shapes are not math")
    
    [sigma] = args
    abs_2 = np.sum(np.abs(img1**2 - img2**2))
    return np.exp(-abs_2/(2*sigma**2)) # todo: the returning value is to small. (exp(-inf) = 0)

def weight(splited, *args):
    [sigma] = args
    N, M = splited.shape[0:2]

    W = np.zeros([N*M, N*M])
    pairs = list(itertools.product(range(N), range(M), range(N), range(M)))
    for p in pairs:
        i,j,k,l = p
        W[i*M+j, k*M+l] = weight_idex(splited[i,j,:,:,:],
                                      splited[k,l,:,:,:], sigma)
    
    return W

def laplacian_graph(weight:np.array):
    return laplacian(weight)