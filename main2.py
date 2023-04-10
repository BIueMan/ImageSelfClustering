import pygame
import numpy as np
import cv2
from funcrions import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_test = cv2.imread('images/Amanita_muscaria_test.jpeg')
    print(img_test.shape)

    splited = split_image(img_test, [200,200], [200, 200])
    size = splited.shape
    print(size)

    # user label image
    from funcrions.pygame_label import *
    pygame.init()
    ## create small image for Pygame window
    step = 4
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

    # create weights
    sigma = 1
    norm_scale = 0.1
    W = weight(splited, sigma, norm_scale)
    W_ssl = weight_ssl(splited, W, labels)
    L_ssl = laplacian_graph(W_ssl)

    eigenvals, eigenvects = np.linalg.eig(L_ssl)
    indices = np.argsort(eigenvals)[:k]
    U = eigenvects[:, indices]

    # solve laplasian with kmeans
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(U)
    laplasian_labels = kmeans.labels_
    laplasian_labels = np.reshape(laplasian_labels, [splited.shape[0], splited.shape[1]])
    print(np.flip(laplasian_labels, axis=1))

    labels = select_filter_ui(small_splited, screen, k, laplasian_labels+1)

    print('\n')
    print(labels)

    