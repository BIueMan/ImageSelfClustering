import numpy as np
import matplotlib.pyplot as plt
import time
from munkres import Munkres
import itertools

# pytorch imports
import torch
import torch.nn as nn

# scikit-learn imports
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import sklearn.metrics
from sklearn.metrics.cluster import normalized_mutual_info_score


def createAffinitySSL(data, labels, ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2):
    '''
    Computes SSL Affinity matrix 
    inputs:
    data:                   array of data featrues
    labels:                 array of data labels
    ms:                     neighbors number per node
    ms_normal:              neighbor for the kernel std. 
    sigmaFlag:              flag for the kernel variance calculation
    labeled_index:          labeled set index array
    classNum:               number of classes
    mu1:                    unsupervised affinity parameter 
    mu2:                    labeled affinity parameter 
    
    returns:    
    y:                      the affinity matrix                
    '''    
    class_labeles = np.arange(0, classNum, 1)
    n, m = data.shape
    W = createAffinity(data, ms, ms_normal, sigmaFlag)
    W = np.array(W)
    W_max = np.max(W)

    W_for_labeld = np.zeros((n, n))
    W_for_labeld[:, labeled_index] = W[:, labeled_index]
    W_for_labeld[labeled_index, :] = 0
    W_for_labeld_T = W_for_labeld.T
    W_for_labeld = 0.5 * (W_for_labeld_T + W_for_labeld)

    lables_labled = np.ones(n) * (-1)
    vals = labels[list(labeled_index)]
    lables_labled[list(labeled_index)] = list(vals)
    for i in range(classNum):
        curr_group = class_labeles[i]
        group_indx = np.where(lables_labled == curr_group)
        group_indx = group_indx[0]
        pairs = list(itertools.product(group_indx, repeat=2))
        pairs_num = len(pairs)
        for j in range(pairs_num):
            W_for_labeld[pairs[j]] = W_max

    W_all = mu1 * W + mu2 * W_for_labeld
    for i in range(classNum - 1):
        first_group = class_labeles[i]
        first_group_indx = np.where(lables_labled == first_group)
        first_group_indx = first_group_indx[0]

        for j in range(i + 1, classNum):
            sec_group = class_labeles[j]
            sec_group_indx = np.where(lables_labled == sec_group)
            sec_group_indx = sec_group_indx[0]
            pairs = list(itertools.product(first_group_indx, sec_group_indx, repeat=1))
            pairs_num = len(pairs)
            for k in range(pairs_num):
                pair_k_flip = np.flip(pairs[k])
                W_all[pairs[k]] = 0
                W_all[pair_k_flip[0], pair_k_flip[1]] = 0
    W_all = (W_all + W_all.T) / 2
    diag_indx = np.arange(0, n, 1)
    W_all[diag_indx, diag_indx] = np.max(W_all)
    W_all = torch.Tensor(W_all)
    return W_all



def createAffinityWNLL(data, ms, ms_normal, sigmaFlag, labeled_index):
    '''
    Computes WNLL Affinity matrix
    inputs:
    data:                   array of data featrues
    ms:                     neighbors number per node
    ms_normal:              neighbor for the kernel std. 
    sigmaFlag:              flag for the kernel variance calculation
    labeled_index:          labeled set index array
    
    returns:    
    y:                      the affinity matrix                
    '''    
    n, m = data.shape
    S_size = len(labeled_index)
    mu1 = 2
    if S_size == 0:
        mu2 = 0
    else:
        mu2 = (n / S_size)-1

    W = createAffinity(data, ms, ms_normal, sigmaFlag)
    W = np.array(W)

    W_for_labeld = np.zeros((n, n))

    W_for_labeld[:, labeled_index] = W[:, labeled_index]
    W_for_labeld[labeled_index, :] = 0
    W_for_labeld_T = W_for_labeld.T
    W_for_labeld = 0.5 * (W_for_labeld_T + W_for_labeld)

    W_all = mu1 * W + mu2 * W_for_labeld
    W_all = torch.Tensor(W_all)
    return W_all


def createAffinityMaxOnly(data, labels, ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2):
    '''
    Computes contrastive affinity (positive) matrix 
    inputs:
    data:                   array of data featrues
    labels:                 array of data labels
    ms:                     neighbors number per node
    ms_normal:              neighbor for the kernel std. 
    sigmaFlag:              flag for the kernel variance calculation
    labeled_index:          labeled set index array
    classNum:               number of classes
    mu1:                    unsupervised affinity parameter 
    mu2:                    labeled affinity parameter 
    
    returns:    
    y:                      the affinity matrix                
    '''    
    class_labeles = np.arange(0, classNum, 1)
    n, m = data.shape
    W = createAffinity(data, ms, ms_normal, sigmaFlag)
    W = np.array(W)
    W_max = np.max(W)

    W_for_labeld = np.zeros((n, n))
    lables_labled = np.ones(n) * (-1)
    vals = labels[list(labeled_index)]
    lables_labled[list(labeled_index)] = list(vals)
    for i in range(classNum):
        curr_group = class_labeles[i]
        group_indx = np.where(lables_labled == curr_group)
        group_indx = group_indx[0]
        pairs = list(itertools.product(group_indx, repeat=2))
        pairs_num = len(pairs)
        for j in range(pairs_num):
            W_for_labeld[pairs[j]] = W_max

    W_all = mu1 * W + mu2 * W_for_labeld
    W_all = (W_all + W_all.T) / 2
    diag_indx = np.arange(0, n, 1)
    W_all[diag_indx, diag_indx] = np.max(W_all)
    W_all = torch.Tensor(W_all)
    return W_all


def createAffinityDisconnectOnly(data, labels, ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2):
    '''
    Computes contrastive affinity (negative) matrix 
    inputs:
    data:                   array of data featrues
    labels:                 array of data labels
    ms:                     neighbors number per node
    ms_normal:              neighbor for the kernel std. 
    sigmaFlag:              flag for the kernel variance calculation
    labeled_index:          labeled set index array
    classNum:               number of classes
    mu1:                    unsupervised affinity parameter 
    mu2:                    labeled affinity parameter 
    
    returns:    
    y:                      the affinity matrix                
    '''    
    class_labeles = np.arange(0, classNum, 1)
    n, m = data.shape
    W = createAffinity(data, ms, ms_normal, sigmaFlag)
    W_all = np.array(W)

    lables_labled = np.ones(n) * (-1)
    vals = labels[list(labeled_index)]
    lables_labled[list(labeled_index)] = list(vals)

    for i in range(classNum - 1):
        first_group = class_labeles[i]
        first_group_indx = np.where(lables_labled == first_group)
        first_group_indx = first_group_indx[0]

        for j in range(i + 1, classNum):
            sec_group = class_labeles[j]
            sec_group_indx = np.where(lables_labled == sec_group)
            sec_group_indx = sec_group_indx[0]
            pairs = list(itertools.product(first_group_indx, sec_group_indx, repeat=1))
            pairs_num = len(pairs)
            for k in range(pairs_num):
                pair_k_flip = np.flip(pairs[k])
                W_all[pairs[k]] = 0.0
                W_all[pair_k_flip[0], pair_k_flip[1]] = 0.0

    W_all = (W_all + W_all.T) / 2
    diag_indx = np.arange(0, n, 1)
    W_all[diag_indx, diag_indx] = np.max(W_all)
    W_all = torch.Tensor(W_all)
    return W_all



def createAffinity(data, ms, ms_normal, sigmaFlag):
    '''
    Computes unsupervised affinity matrix 
    inputs:
    data:                   array of data featrues
    ms:                     neighbors number per node
    ms_normal:              neighbor for the kernel std. 
    sigmaFlag:              flag for the kernel variance calculation
    
    returns:    
    y:                      the affinity matrix                
    '''    
    n = data.shape[0]
    nbrs = NearestNeighbors(n_neighbors=ms, algorithm='kd_tree').fit(data)
    dist, idx = nbrs.kneighbors(data)
    graph_median = np.median(dist)
    dist = torch.Tensor(dist.T)
    idx = torch.Tensor(idx.T)
    id_row = torch.Tensor([range(0, n)])
    id_row = id_row.repeat(ms, 1)
    id_row = id_row.numpy()
    id_col = idx.numpy()

    if sigmaFlag == 0:
        sigma = torch.diag(1. / dist[ms_normal, :])
        W = torch.exp(-(dist @ sigma) ** 2)

    if sigmaFlag == 1:
        sigma = torch.median(dist[ms_normal, :])
        W = torch.exp(-dist ** 2 / (sigma ** 2))

    if sigmaFlag == 2:
        W = torch.exp(-dist ** 2 / (2 * graph_median ** 2))

    if sigmaFlag == 3:
        sigma = 1
        W = torch.exp(-dist ** 2 / sigma)

    y = torch.sparse_coo_tensor([id_row.flatten(), id_col.flatten()], W.flatten(), (n, n))
    y = y.to_dense()
    y = (y + y.T) / 2
    return y



def ev_calculation_L(W, classNum):
    '''
    Computes the graph Laplacian eigenvectors
    inputs:
    W:                      affinity matrix
    classNum:               number of classes
   
    returns:    
    RCut_EV:                the first K eigenvectors of L               
    '''

    s0 = torch.sum(W, axis=0)
    # L
    D = torch.diag(s0)
    L = D - W
    S_L, U_L = torch.linalg.eig(L)
    S_L = torch.real(S_L)
    U_L = torch.real(U_L)
    S_L, indices = torch.sort(S_L, dim=0, descending=False, out=None)
    U_L = U_L[:, indices]
    RCut_EV = U_L[:, 1:classNum]
    # RCut_EV = U_L[:, 0:classNum]
    return RCut_EV


def ev_calculation_LN(W, classNum,):
    '''
    Computes the normalized graph Laplacian eigenvectors
    inputs:
    W:                      affinity matrix
    classNum:               number of classes
   
    returns:    
    RCut_EV:                the first K eigenvectors of L_N               
    '''

    n = W.size(0)
    s0 = torch.sum(W, axis=0)

    # L_N
    D_sqrt = torch.diag(1. / torch.sqrt(s0))
    I = torch.eye(n)
    N = I - D_sqrt @ W @ D_sqrt
    S_N, U_N = torch.linalg.eig(N) 
    S_N = torch.real(S_N)
    U_N = torch.real(U_N)
    S_N, indices = torch.sort(S_N, dim=0, descending=False, out=None)
    U_N = U_N[:, indices]
    RCut_EV = U_N[:, 1:classNum]
    # RCut_EV = U_N[:, 0:classNum]
    return RCut_EV



def SpectralClusteringFromEV(ev, true_labels, classNum):
    '''
    performe spectral clutering from the spectral embedding
    inputs:
    ev:                     the eigenvectors of the graph Laplacian
    true_labels:            data true labels
    classNum:               number of classes

    returns:    
    RCut_labels:            spectral clustering assignment 
    model_nmi:              nmi value
    model_acc:              acc value
    '''
    RCut_kmeans = KMeans(n_clusters=classNum, random_state=0).fit(ev)
    RCut_labels = RCut_kmeans.labels_
    model_nmi = normalized_mutual_info_score(true_labels, RCut_labels)
    model_acc, _ = get_acc(RCut_labels, true_labels, classNum)
    return RCut_labels, model_nmi, model_acc



# Performance measures
def get_orthogonality_measure(U, classNum):
    '''
    calcute the orthogonality measure
    inputs:
    U:                      the matrix whose orthogonality is tested
    classNum:               number of classes

    returns:    
    orthogonality_measure:  orthogonality measure 
    '''
    n, m = U.shape
    ev_norm = np.linalg.norm(U, axis=0)
    ev_norm = 1 / ev_norm
    ev_norm_matrix = np.tile(ev_norm, (m, 1))
    orthogonality_matrix = U.T @ U
    orthogonality_matrix = np.multiply(np.multiply(ev_norm_matrix.T, orthogonality_matrix), ev_norm_matrix)

    dim = orthogonality_matrix.shape[0]
    I = np.eye(dim)

    orthogonality_measure = np.linalg.norm(orthogonality_matrix - I)
    return orthogonality_measure


def grassmann(A, B):
    '''
    calcute grassmann distance 
    inputs:
    A, B:                   the matrices for which the distance is checked

    returns:    
    grassmann_val:          grassmann distance between A and B 
    '''
    n, m = A.shape

    A_col_norm = torch.linalg.norm(A, dim=0)
    A_col_norm = 1 / A_col_norm
    A_norm_matrix = torch.tile(A_col_norm, (n, 1))
    A_normalized = A_norm_matrix * A  #elmentwise
    A_normalized = A_normalized.float()

    B_col_norm = torch.linalg.norm(B, dim=0)
    B_col_norm = 1 / B_col_norm
    B_norm_matrix = torch.tile(B_col_norm, (n, 1))
    B_normalized = B_norm_matrix * B  #elmentwise
    B_normalized = B_normalized.float()

    M = A_normalized.T @ B_normalized
    _, s, _ = torch.linalg.svd(M) # return ev with norm 1
    s = 1 - torch.square(s)
    grassmann_val = torch.sum(s)
    return grassmann_val


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_y_preds(cluster_assignments, y_true, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred, confusion_matrix


def get_acc(cluster_assignments, y_true, n_clusters):
    '''
    Computes the accuracy based on the provided kmeans cluster assignments
    and true labels, using the Munkres algorithm
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    y_true = y_true.numpy()
    y_pred, confusion_matrix = get_y_preds(cluster_assignments, y_true, n_clusters)
    # calculate the accuracy
    return np.mean(y_pred == y_true), confusion_matrix


def Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes):
    '''
    Dirichlet multiclass clustering via interpolation 
    inputs:
    L:                      graph Laplacian
    labeled_index:          labeled set index array
    unlabeled_index:        unlabeled set index array
    y:                      true labeles
    classes:                classes array
    

    returns:    
    grassmann_val:          grassmann distance between A and B 
    '''
    #classes should be numbers from 0 to classNum-1.
    n = L.shape[0]
    classNum = len(classes)
    y_labeled = y[labeled_index]
    y_unlabeled = y[unlabeled_index]

    A = L.clone()
    A[labeled_index, :] = 0
    A[labeled_index, labeled_index] = 1

    for i in range(classNum):
        b = torch.zeros(n)
        b[labeled_index] = 1
        b[y != i] = 0

        phi = torch.linalg.lstsq(A, b, rcond=None).solution
        phi_unlabeled = phi[unlabeled_index]
        phi_unlabeled = torch.unsqueeze(phi_unlabeled, 1)
        if i == 0:
            totalPhi = phi_unlabeled
        else:

            totalPhi = torch.cat((totalPhi, phi_unlabeled), dim=1)

    max_values, clusteringRes = torch.max(totalPhi, 1)
    model_nmi = normalized_mutual_info_score(y_unlabeled, clusteringRes)
    model_acc, _ = get_acc(clusteringRes, y_unlabeled, classNum)

    return clusteringRes, model_nmi, model_acc


def Dirichlet_Interploation(L, labeled_index, unlabeled_index, y, classes):
    '''
    Dirichlet interpolation solution (2 classes)
    inputs:
    L:                      graph Laplacian
    labeled_index:          labeled set index array
    unlabeled_index:        unlabeled set index array
    y:                      true labeles
    classes:                classes array
    

    returns:    
    grassmann_val:          grassmann distance between A and B 
    '''
    n = L.shape[0]
    classNum = len(classes)
    y_labeled = y[labeled_index]

    indx0_labeled = np.where(y_labeled == 0)[0]
    labeled_index0 = np.array(labeled_index)[indx0_labeled.astype(int)]

    indx1_labeled = np.where(y_labeled == 1)[0]
    labeled_index1 = np.array(labeled_index)[indx1_labeled.astype(int)]

    y_unlabeled = y[unlabeled_index]

    A = L.clone()
    A[labeled_index, :] = 0
    A[labeled_index, labeled_index] = 1

    b = torch.zeros(n)
    b[labeled_index0] = -1
    b[labeled_index1] = 1

    phi = torch.linalg.lstsq(A, b, rcond=None).solution
    return phi



def TwoMoons_SSL_Solutions(X, y, option_index, ms, ms_normal, sigmaFlag, classes, mu1, labeled_index, model_path):
    '''
    SSL solutions for 2 moons dataset
    inputs:
    X:                      array of data featrues
    y:                      array of data labels
    option_index:           index of current labeled subset
    ms:                     neighbors number per node
    ms_normal:              neighbor for the kernel std. 
    sigmaFlag:              flag for the kernel variance calculation
    classes:                classes array
    mu1:                    unsupervised affinity parameter 
    labeled_index:          labeled set index array
    
    L:                      graph Laplacian
    labeled_index:          labeled set index array
    unlabeled_index:        unlabeled set index array
    y:                      true labeles
    classes:                classes array
    model_path:             path for images

    '''
    print("option: ", option_index)
    n_samples = len(X)
    nodes_indx_list = range(0, n_samples)
    classNum = len(classes)

    unlabeled_index = [indx for indx in nodes_indx_list if indx not in labeled_index]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c='white', edgecolor="blue")
    ax.scatter(X[labeled_index, 0], X[labeled_index, 1], c='red', s=100)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    savefig_path = model_path + "/images/labeled_option_" + str(option_index) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    number_of_labeled_nodes = len(labeled_index)
    mu2 = (n_samples / number_of_labeled_nodes) - 1

    y_unlabeled = y[unlabeled_index]
    X0_unlabeled = X[unlabeled_index, 0]
    X0_unlabeled_index = np.argmax(X0_unlabeled)

    print("Spectral WNLL")
    W_WNLL = createAffinityWNLL(X, ms, ms_normal, sigmaFlag, labeled_index)
    ev = ev_calculation_L(W_WNLL, classNum)
    ev_unlabeled = ev[unlabeled_index]
    RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev_unlabeled, y[unlabeled_index], classNum)
    print("NMI:", model_nmi)
    print("ACC:", model_acc)

    RCut_labels_max = RCut_labels[X0_unlabeled_index]
    if RCut_labels_max == 0:
        RCut_labels[RCut_labels == 1] = -1
        RCut_labels[RCut_labels == 0] = 1
        RCut_labels[RCut_labels == -1] = 0

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[labeled_index, 0], X[labeled_index, 1], c=y[labeled_index])
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=RCut_labels)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/spectral_wnll_option_" + str(option_index) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=ev)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/spectral_wnll_option_" + str(option_index) + "_ev.png"
    plt.savefig(savefig_path)
    plt.show()


    print("Spectral SSL")
    W_ssl = createAffinitySSL(X, y, ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2)
    ev = ev_calculation_L(W_ssl, classNum)
    ev_unlabeled = ev[unlabeled_index]
    RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev_unlabeled, y[unlabeled_index], classNum)
    print("NMI:", model_nmi)
    print("ACC:", model_acc)

    RCut_labels_max = RCut_labels[X0_unlabeled_index]
    if RCut_labels_max == 0:
        RCut_labels[RCut_labels == 1] = -1
        RCut_labels[RCut_labels == 0] = 1
        RCut_labels[RCut_labels == -1] = 0

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[labeled_index, 0], X[labeled_index, 1], c=y[labeled_index])
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=RCut_labels)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/spectral_ssl_option_" + str(option_index) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=ev)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/spectral_ssl_option_" + str(option_index) + "_ev.png"
    plt.savefig(savefig_path)
    plt.show()


    print("Dirichlet  US")
    W_US = createAffinity(X, ms, ms_normal, sigmaFlag)
    s0 = torch.sum(W_US, axis=0)
    D = torch.diag(s0)
    L = D - W_US
    phi = Dirichlet_Interploation(L, labeled_index, unlabeled_index, y, classes)
    clusteringRes_kmeans = KMeans(n_clusters=classNum, random_state=0).fit(phi.reshape(-1, 1))
    clusteringRes = clusteringRes_kmeans.labels_
    clusteringRes = clusteringRes[unlabeled_index]

    model_nmi = normalized_mutual_info_score(y_unlabeled, clusteringRes)
    model_acc, _ = get_acc(clusteringRes, y_unlabeled, classNum)
    print("NMI:", model_nmi)
    print("ACC:", model_acc)

    clusteringRes_max = clusteringRes[X0_unlabeled_index]
    if clusteringRes_max == 0:
        clusteringRes[clusteringRes == 1] = -1
        clusteringRes[clusteringRes == 0] = 1
        clusteringRes[clusteringRes == -1] = 0

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=phi)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/dirichlet_US_option_" + str(option_index) + "_phi.png"
    plt.savefig(savefig_path)
    plt.show()

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[labeled_index, 0], X[labeled_index, 1], c=y[labeled_index])
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=clusteringRes)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/dirichlet_US_option_" + str(option_index) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    print("Dirichlet WNLL")
    s0 = torch.sum(W_WNLL, axis=0)
    D = torch.diag(s0)
    L = D - W_WNLL
    phi = Dirichlet_Interploation(L, labeled_index, unlabeled_index, y, classes)
    clusteringRes_kmeans = KMeans(n_clusters=classNum, random_state=0).fit(phi.reshape(-1, 1))
    clusteringRes = clusteringRes_kmeans.labels_
    clusteringRes = clusteringRes[unlabeled_index]

    model_nmi = normalized_mutual_info_score(y_unlabeled, clusteringRes)
    model_acc, _ = get_acc(clusteringRes, y_unlabeled, classNum)
    print("NMI:", model_nmi)
    print("ACC:", model_acc)

    clusteringRes_max = clusteringRes[X0_unlabeled_index]
    if clusteringRes_max == 0:
        clusteringRes[clusteringRes == 1] = -1
        clusteringRes[clusteringRes == 0] = 1
        clusteringRes[clusteringRes == -1] = 0

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=phi)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/dirichlet_WNLL_option_" + str(option_index) + "_phi.png"
    plt.savefig(savefig_path)
    plt.show()

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[labeled_index, 0], X[labeled_index, 1], c=y[labeled_index])
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=clusteringRes)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/dirichlet_WNLL_option_" + str(option_index) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    print("Dirichlet SSL")
    s0 = torch.sum(W_ssl, axis=0)
    D = torch.diag(s0)
    L = D - W_ssl

    phi = Dirichlet_Interploation(L, labeled_index, unlabeled_index, y, classes)
    clusteringRes_kmeans = KMeans(n_clusters=classNum, random_state=0).fit(phi.reshape(-1, 1))
    clusteringRes = clusteringRes_kmeans.labels_
    clusteringRes = clusteringRes[unlabeled_index]

    model_nmi = normalized_mutual_info_score(y_unlabeled, clusteringRes)
    model_acc, _ = get_acc(clusteringRes, y_unlabeled, classNum)
    print("NMI:", model_nmi)
    print("ACC:", model_acc)

    clusteringRes_max = clusteringRes[X0_unlabeled_index]
    if clusteringRes_max == 0:
        clusteringRes[clusteringRes == 1] = -1
        clusteringRes[clusteringRes == 0] = 1
        clusteringRes[clusteringRes == -1] = 0

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=phi)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/dirichlet_SSL_option_" + str(option_index) + "_phi.png"
    plt.savefig(savefig_path)
    plt.show()

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(X[labeled_index, 0], X[labeled_index, 1], c=y[labeled_index])
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=clusteringRes)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    #plt.colorbar(sc)
    savefig_path = model_path + "/images/dirichlet_SSL_option_" + str(option_index) + ".png"
    plt.savefig(savefig_path)
    plt.show()








