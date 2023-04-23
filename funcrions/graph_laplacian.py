import numpy as np
import cv2
from scipy.sparse.csgraph import laplacian
import itertools
from tqdm import tqdm

def weight_idex(img1:np.array, img2:np.array, *args)->np.array:
    if img1.shape != img2.shape:
        raise Exception("input imags shapes are not math")
    
    [sigma] = args
    abs_2 = np.sum(np.abs(img1**2 - img2**2))
    return np.exp(-abs_2/(2*sigma**2))

def weight(splited, *args):
    N, M = splited.shape[0:2]

    W = np.zeros([N*M, N*M])
    pairs = list(itertools.product(range(N), range(M), range(N), range(M)))
    for p in tqdm(pairs):
        i,j,k,l = p
        if i*M+j > k*M+l: # W is simetric
            continue
        W[i*M+j, k*M+l] = weight_idex(splited[i,j,:,:,:],
                                      splited[k,l,:,:,:], *args)
        W[k*M+l, i*M+j] = W[i*M+j, k*M+l]
    
    return W

def laplacian_graph(weight:np.array):
    return np.diagflat(np.sum(weight, 0)) - weight

def weight_ssl(weight, label):
    N, M = label.shape
    alpha = N*M/np.sum(label!=0) - 1

    W_label = np.zeros([N*M, N*M])
    pairs = list(itertools.product(range(N), range(M), range(N), range(M)))
    for p in tqdm(pairs):
        i,j,k,l = p
        if i*M+j > k*M+l: # W is simetric
            continue

        def w_label_cal(label_1, label_2, weight):
            if label_1 == 0 and label_2 == 0:
                return 0
            elif (label_1 != 0 and label_2 == 0) or (label_1 == 0 and label_2 != 0):
                return weight[i*M+j, k*M+l]
            elif label_1 == label_2:
                return np.max(weight)
            else:
                return -(2/alpha)*weight[i*M+j, k*M+l]

        W_label[i*M+j, k*M+l] = w_label_cal(label[i, j], label[k,l], weight)
        W_label[k*M+l, i*M+j] = W_label[i*M+j, k*M+l]
    
    return 2*weight + alpha*W_label