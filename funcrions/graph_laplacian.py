import numpy as np
import cv2
from scipy.sparse.csgraph import laplacian
import itertools
from tqdm import tqdm

def weight_idex(img1:np.array, img2:np.array, *args)->np.array:
    if img1.shape != img2.shape:
        raise Exception("input imags shapes are not math")
    
    [sigma, norm_scale] = args
    # norm image
    if np.max(img1) != 0:
        img1 = img1/np.max(img1)
    if np.max(img2) != 0:
        img2 = img2/np.max(img2)
    abs_2 = np.sum(np.abs(img1**2 - img2**2)) / (norm_scale*img1.size)
    return np.exp(-abs_2/(2*sigma**2)) # todo: the returning value is to small. (exp(-inf) = 0)

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

def weight_ssl(splited, weight, label):
    N, M = splited.shape[0:2]
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