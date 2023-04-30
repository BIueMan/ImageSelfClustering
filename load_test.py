import pygame
import numpy as np
import cv2
from funcrions import *
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == "__main__":
    img_test = cv2.imread('images/Amanita_muscaria_test.jpeg')
    img_test = cv2.imread('images/lion_and_giraffe.jpeg')
    img_test = cv2.imread('images/flower.jpeg')
    img_test = cv2.imread('images/car.jpeg')
    a = np.load('images/ae_d_28_28_lt.npy')
    b = np.load('images/ae_d_28_28_rec.npy')
    c = np.load('images/hist_28_28_32.npy')
    # img_test = cv2.imread('images/object.png')
    print(img_test.shape)
    h, w, _ = img_test.shape
    # Calculate the new size of the image
    new_h = h 
    new_w = w
    # Resize the image using cv2.resize()
    img_test = cv2.resize(img_test, (new_w, new_h))
    print(img_test.shape)

    splited = split_image(img_test, [28,28], [28, 28])
    size = splited.shape
    print(size)

    # user label image
    from funcrions.pygame_label import *
    pygame.init()
    ## create small image for Pygame window
    step = 1
    small_splited = splited[:,:,range(0,size[2],step),:,:]
    small_splited = small_splited[:,:,:,range(0,size[3],step),:]
    print(small_splited.shape)
    window_size = (small_splited.shape[0]*small_splited.shape[2], 
                   small_splited.shape[1]*small_splited.shape[3])
    screen_size = np.array(window_size[::-1]) + np.array([100, 0])
    screen = pygame.display.set_mode(screen_size)
    ## label image
    k = 3  # number of clusters
    labels = select_filter_ui(small_splited, screen, k)

    # Weighted Nonlocal Laplacian
    l, h = c.shape[0:2]
    X_data = torch.tensor(c).reshape(l * h, -1)
    X_data = X_data / torch.max(X_data)
    y_label = torch.tensor(labels.astype(np.int64)).reshape(l * h, -1).squeeze() - 1

    labeled_index = torch.where(y_label != -1)[0].numpy()
    unlabeled_index = torch.where(y_label == -1)[0].numpy()

    from funcrions.SSL_Laplacian.utils import *
    ms = 50
    ms_normal = 20
    sigmaFlag = 0
    mu1 = 2.0
    if len(labeled_index) == 0:
        mu2 = 0
    else:
        mu2 = (l * h / len(labeled_index)) - 1
    W_ssl = createAffinitySSL(X_data, y_label, ms, ms_normal, sigmaFlag, labeled_index, k, mu1, mu2)
    ev = ev_calculation_L(W_ssl, k)
    ev_unlabeled = ev[unlabeled_index]
    RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev_unlabeled, y_label[unlabeled_index], k)

    label_out = y_label.numpy()
    label_out[unlabeled_index] = RCut_labels
    label_out = label_out.reshape(l, h) +1

    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    labels = select_filter_ui(small_splited, screen, k, label_out)

    # # Gaussian kernel
    # ## create weights
    # sigma = 8.5
    # W = weight(splited/np.max(splited), sigma)
    # if np.sum(labels) != 0:
    #     W_ssl = weight_ssl(W, labels)
    # else:
    #     W_ssl = W
    # # L = laplacian_graph(W)
    # L_ssl = laplacian_graph(W_ssl)

    # eigenvals, eigenvects = np.linalg.eig(L_ssl)
    # # if np.sum(np.imag(eigenvals)) != 0 or np.sum(np.imag(eigenvects)) != 0:
    # #     print('find a complex eigen')
    # # else:
    # eigenvals = np.real(eigenvals)
    # eigenvects = np.real(eigenvects)

    # indices = np.argsort(eigenvals)[:k]
    # U = eigenvects[:, indices]

    # # solve laplasian with kmeans
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=k)
    # kmeans.fit(U)
    # laplasian_labels = kmeans.labels_
    # laplasian_labels = np.reshape(laplasian_labels, [splited.shape[0], splited.shape[1]])
    # print('KMean')
    # print(np.flip(laplasian_labels, axis=1))

    # # # solve laplasian with SpectralClustering
    # # from sklearn.cluster import SpectralClustering
    # # spectal = SpectralClustering(n_clusters=k)
    # # spectal.fit(eigenvects[:, np.argsort(eigenvals)])
    # # laplasian_labels = spectal.labels_
    # # laplasian_labels = np.reshape(laplasian_labels, [splited.shape[0], splited.shape[1]])
    # # print('SpectralClustering')
    # # print(np.flip(laplasian_labels, axis=1))

    # # show label image
    # pygame.init()
    # screen = pygame.display.set_mode(screen_size)
    # labels = select_filter_ui(small_splited, screen, k, laplasian_labels+1)

    # print('\n')
    # print(labels)

    