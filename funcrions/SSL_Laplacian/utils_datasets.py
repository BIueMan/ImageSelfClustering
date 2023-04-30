import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math

# scikit-learn imports
from sklearn.datasets import make_circles, make_moons



def generate_2moons(n_samples=10000, noise_param=0.07, toPlot=False):
    X, y = make_moons(n_samples, noise=noise_param)

    X0 = X[:, 0]
    X0max_index = np.argmax(X0)
    ymax = y[X0max_index]

    if ymax == 0:
        y[y == 1] = -1
        y[y == 0] = 1
        y[y == -1] = 0

    # plot
    if toPlot == True:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X[:, 0], X[:, 1], c=y)
        #ax.set_title("Moons Dataset")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.grid(True)
        plt.show()
    return X, y



def generate_2moons_with_anchors(n_samples=10000, noise_param=0.07, toPlot=False):
    # X, y = make_moons(n_samples, noise=noise_param, random_state=0)
    X, y = make_moons(n_samples, noise=noise_param, random_state=0)
    # y[np.where(y == 0)[0]] = -1
    my_X = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.4], [1.0, -0.5], [2.0, 0.4],
                     [-0.6, 0.8], [0.6, 0.8], [0.4, -0.3], [1.6, -0.3]])
    my_y = np.array([0, 0, 0, 1, 1, 1,
                     0, 0, 1, 1])
    plot_y = np.array([2, 2, 2, 2, 2, 2,
                       2, 2, 2, 2])

    total_X = np.concatenate([my_X, X])
    total_y = np.concatenate([my_y, y])
    plot_total_y = np.concatenate([plot_y, y])

    # plot
    if toPlot == True:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(total_X[:, 0], total_X[:, 1], c=plot_total_y)
        #ax.set_title("Moons Dataset")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.grid(True)
        plt.show()
    return total_X, total_y





def generate_3moons(n_samples=10000, noise_param=0.07, toPlot=False):
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
    cmap_name = 'my_cmap'

    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)

    n_samples_one_moon = round(n_samples/3)

    R = 1
    one_circ_x = R * np.cos(np.linspace(0, math.pi, n_samples_one_moon))
    one_circ_y = R * np.sin(np.linspace(0, math.pi, n_samples_one_moon))

    two_circ_x = 1.5 * R * np.cos(np.linspace(math.pi, 2 * math.pi, n_samples_one_moon)) + 1.5
    two_circ_y = 1.5 * R * np.sin(np.linspace(math.pi, 2 * math.pi, n_samples_one_moon)) + 0.35

    three_circ_x = R * np.cos(np.linspace(0, math.pi, n_samples_one_moon)) + 3
    three_circ_y = R * np.sin(np.linspace(0, math.pi, n_samples_one_moon))

    X1 = np.concatenate((one_circ_x, two_circ_x, three_circ_x), axis=0)
    X2 = np.concatenate((one_circ_y, two_circ_y, three_circ_y), axis=0)
    X = np.stack((X1, X2), axis=1)

    noise = noise_param * np.random.randn(X.shape[0], X.shape[1])
    X = X + noise
    y = np.concatenate((np.zeros(n_samples_one_moon), np.ones(n_samples_one_moon), 2 * np.ones(n_samples_one_moon)))

    # plot
    if toPlot == True:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(X[:, 0], X[:, 1], c=y, cmap='cividis')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm)
        #ax.set_title("3-Moons Dataset")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.grid(True)
        plt.show()
    return X, y