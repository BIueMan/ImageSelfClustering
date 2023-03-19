import pygame
import numpy as np
import cv2
from funcrions import *

if __name__ == "__main__":
    img_test = cv2.imread('images/Amanita_muscaria_test.jpeg')
    print(img_test.shape)

    splited = split_image(img_test, [200,200], [200, 200])
    # splited = np.zeros_like(splited).astype('int')
    # splited[1,:,:,:,0] = 250
    # splited[5,:,:,:,1] = 250
    size = splited.shape
    print(size)

    sigma = 1
    norm_scale = 0.1
    W = weight(splited, sigma, norm_scale)
    L = laplacian_graph(W)

    k = 3  # number of clusters
    eigenvals, eigenvects = np.linalg.eig(L)
    indices = np.argsort(eigenvals)[:k]
    U = eigenvects[:, indices]

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(U)
    init_labels = kmeans.labels_
    init_labels = np.reshape(init_labels, [splited.shape[0], splited.shape[1]])
    print(np.flip(init_labels, axis=1))

    from funcrions.pygame_label import *
    # Initialize Pygame
    pygame.init()

    # Set the size of the Pygame window
    step = 4
    small_splited = splited[:,:,range(0,size[2],step),:,:]
    small_splited = small_splited[:,:,:,range(0,size[3],step),:]
    print(small_splited.shape)
    window_size = (small_splited.shape[0]*small_splited.shape[2], 
                   small_splited.shape[1]*small_splited.shape[3])

    def display_splited_image():
        # Create the Pygame window
        screen = pygame.display.set_mode(window_size[::-1])

        display_big_image(small_splited, screen)
        pygame.display.flip()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    display_splited_image()

    screen_size = np.array(window_size[::-1]) + np.array([100, 0])
    screen = pygame.display.set_mode(screen_size)
    labels = select_filter_ui(small_splited, screen, k, init_labels)

    print(labels)