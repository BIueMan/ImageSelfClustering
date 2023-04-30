import os
import math
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch

from utils_datasets import generate_3moons
from utils import createAffinity, createAffinitySSL, createAffinityWNLL, ev_calculation_L, \
    createAffinityMaxOnly, createAffinityDisconnectOnly

import warnings
warnings.simplefilter("ignore", UserWarning)


colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
cmap_name = 'my_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
font_size = 22

plotFlag = True

if not os.path.exists('./results'):
    os.mkdir('./results')
    
model_path = r"./results/laplacian_illustration"
if not os.path.exists(model_path):
    os.mkdir(model_path)
    os.mkdir(model_path + '/images')

n_samples = 900

classes = ('0', '1', '2')
classNum = len(classes)
nodes_indx_list = range(0, n_samples)

number_of_labeled_nodes = 30
ms = n_samples
ms_normal = 7
sigmaFlag = 0
mu1 = 2.0
mu2 = (n_samples/number_of_labeled_nodes) - 1

noise_param = 0.175

X, y = generate_3moons(n_samples=n_samples, noise_param=noise_param, toPlot=False)


n_samples_one_moon = 10
R = 1
one_circ_x = R * np.cos(np.linspace(0, math.pi, n_samples_one_moon))
one_circ_y = R * np.sin(np.linspace(0, math.pi, n_samples_one_moon))

two_circ_x = 1.5 * R * np.cos(np.linspace(math.pi, 2 * math.pi, n_samples_one_moon)) + 1.5
two_circ_y = 1.5 * R * np.sin(np.linspace(math.pi, 2 * math.pi, n_samples_one_moon)) + 0.35

three_circ_x = R * np.cos(np.linspace(0, math.pi, n_samples_one_moon)) + 3
three_circ_y = R * np.sin(np.linspace(0, math.pi, n_samples_one_moon))

X1 = np.concatenate((one_circ_x, two_circ_x, three_circ_x), axis=0)
X2 = np.concatenate((one_circ_y, two_circ_y, three_circ_y), axis=0)
my_X = np.stack((X1, X2), axis=1)

my_y= np.concatenate((np.zeros(n_samples_one_moon), np.ones(n_samples_one_moon), 2 * np.ones(n_samples_one_moon)))

X = np.concatenate([my_X, X])
y = np.concatenate([my_y, y])

X = torch.tensor(X)
y = torch.tensor(y)


labeled_index = np.arange(0, len(my_y), 1)
unlabeled_index = [indx for indx in nodes_indx_list if indx not in labeled_index]

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.grid(True)
savefig_path = model_path + "/images/li_3_moons_dataset.png"
plt.savefig(savefig_path)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c='white', edgecolor ="blue")
c = ax.scatter(X[labeled_index, 0], X[labeled_index, 1], c='red', s=80)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.grid(True)
savefig_path = model_path + "/images/labeled_nodes.png"
plt.savefig(savefig_path)
plt.show()


W_US = createAffinity(X, ms, ms_normal, sigmaFlag)
ev_us = ev_calculation_L(W_US, classNum)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.scatter(ev_us[:, 0], ev_us[:, 1], c=y, marker='o', cmap=cm, s=200)
ax.set_xlabel("$u_{1}$", fontsize=font_size)
ax.set_ylabel("$u_{2}$", fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(y = 0.0, color = 'black', linestyle = '-')
plt.axvline(x = 0.0, color = 'black', linestyle = '-')
plt.grid(True)
savefig_path = model_path + "/images/embd_us.png"
plt.savefig(savefig_path)
plt.show()


W_ssl = createAffinitySSL(X, y, ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2)
ev_ssl = ev_calculation_L(W_ssl, classNum)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.scatter(ev_ssl[:, 0], ev_ssl[:, 1], c=y, marker='o', cmap=cm, s=200)
ax.set_xlabel("$u_{1}$", fontsize=font_size)
ax.set_ylabel("$u_{2}$", fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(y = 0.0, color = 'black', linestyle = '-')
plt.axvline(x = 0.0, color = 'black', linestyle = '-')
plt.grid(True)
savefig_path = model_path + "/images/embd_ssl.png"
plt.savefig(savefig_path)
plt.show()


W_WNLL = createAffinityWNLL(X,  ms, ms_normal, sigmaFlag, labeled_index)
ev_wnll = ev_calculation_L(W_WNLL, classNum)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.scatter(ev_wnll[:, 0], ev_wnll[:, 1], c=y, marker='o', cmap=cm, s=200)
ax.set_xlabel("$u_{1}$", fontsize=font_size)
ax.set_ylabel("$u_{2}$", fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(y = 0.0, color = 'black', linestyle = '-')
plt.axvline(x = 0.0, color = 'black', linestyle = '-')
plt.grid(True)
savefig_path = model_path + "/images/embd_wnll.png"
plt.savefig(savefig_path)
plt.show()


W_maxonly = createAffinityMaxOnly(X, y,  ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2)
ev_maxonly = ev_calculation_L(W_maxonly, classNum)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.scatter(ev_maxonly[:, 0], ev_maxonly[:, 1], c=y, marker='o', cmap=cm, s=200)
ax.set_xlabel("$u_{1}$", fontsize=font_size)
ax.set_ylabel("$u_{2}$", fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(y = 0.0, color = 'black', linestyle = '-')
plt.axvline(x = 0.0, color = 'black', linestyle = '-')
plt.grid(True)
savefig_path = model_path + "/images/embd_maxonly.png"
plt.savefig(savefig_path)
plt.show()


W_disconnectonly = createAffinityDisconnectOnly(X, y,  ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2)
ev_disconnectonly = ev_calculation_L(W_disconnectonly, classNum)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.scatter(ev_disconnectonly[:, 0], ev_disconnectonly[:, 1], c=y, marker='o', cmap=cm, s=200)
ax.set_xlabel("$u_{1}$", fontsize=font_size)
ax.set_ylabel("$u_{2}$", fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.axhline(y = 0.0, color = 'black', linestyle = '-')
plt.axvline(x = 0.0, color = 'black', linestyle = '-')
plt.grid(True)
savefig_path = model_path + "/images/embd_disconnectonly.png"
plt.savefig(savefig_path)
plt.show()


