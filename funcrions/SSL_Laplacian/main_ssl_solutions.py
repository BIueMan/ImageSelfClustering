import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from utils_datasets import generate_2moons_with_anchors
from utils import createAffinity, ev_calculation_L, SpectralClusteringFromEV, TwoMoons_SSL_Solutions


import warnings
warnings.simplefilter("ignore", UserWarning)

plotFlag = False

if not os.path.exists('./results'):
    os.mkdir('./results')
    
model_path = r"./results/extreamly_ssl_clustering"
if not os.path.exists(model_path):
    os.mkdir(model_path)
    os.mkdir(model_path + '/images')

n_samples = 1000
nodes_indx_list = range(0, n_samples)
classes = ('0', '1')
classNum = len(classes)
noise_param = 0.12
X, y = generate_2moons_with_anchors(n_samples=n_samples, noise_param=noise_param, toPlot=plotFlag)
anchors_num = 10

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 1, 1)
sc = ax.scatter(X[:, 0], X[:, 1], c=y)
# ax.set_title("Moons Dataset")
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.grid(True)
# plt.colorbar(sc)
plt.show()


X = torch.tensor(X)
y = torch.tensor(y)
X0 = X[:, 0]
X0max_index = np.argmax(X0)

X_unlabeled = X[anchors_num:, :]
y_unlabeled = y[anchors_num:]
X_anchors = X[0:anchors_num, :]
y_anchors = y[0:anchors_num]

ms = len(y)
ms_normal = 7
sigmaFlag = 0

mu1 = 2.0

# anchors only
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.scatter(X[anchors_num:, 0], X[anchors_num:, 1], c='white', edgecolor ="blue")
c = ax.scatter(X[0:anchors_num, 0], X[0:anchors_num, 1], c='red')
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.show()


print("Spectral US")
W_US = createAffinity(X, ms, ms_normal, sigmaFlag)
ev = ev_calculation_L(W_US, classNum)
RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev, y, classNum)
print("NMI:", model_nmi)
print("ACC:", model_acc)

RCut_labels_max = RCut_labels[X0max_index]
if RCut_labels_max == 0:
    RCut_labels[RCut_labels == 1] = -1
    RCut_labels[RCut_labels == 0] = 1
    RCut_labels[RCut_labels == -1] = 0

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 1, 1)
sc = ax.scatter(X[:, 0], X[:, 1], c=RCut_labels)
# ax.set_title("Moons Dataset")
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.grid(True)
# plt.colorbar(sc)
savefig_path = model_path + "/images/rcut_us.png"
plt.savefig(savefig_path)
plt.show()

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 1, 1)
sc = ax.scatter(X[:, 0], X[:, 1], c=ev)
# ax.set_title("Moons Dataset")
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.grid(True)
# plt.colorbar(sc)
savefig_path = model_path + "/images/rcut_us_ev.png"
plt.savefig(savefig_path)
plt.show()

print("---SSL---")
option_index = 0
labeled_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TwoMoons_SSL_Solutions(X, y, option_index, ms, ms_normal, sigmaFlag, classes, mu1, labeled_index, model_path)

option_index = 1
labeled_index = [0, 1, 2, 3, 4, 5]
TwoMoons_SSL_Solutions(X, y, option_index, ms, ms_normal, sigmaFlag, classes, mu1, labeled_index, model_path)

option_index = 2
labeled_index = [6,1,9,4]
TwoMoons_SSL_Solutions(X, y, option_index, ms, ms_normal, sigmaFlag, classes, mu1, labeled_index, model_path)

option_index = 3
labeled_index = [0, 5]
TwoMoons_SSL_Solutions(X, y, option_index, ms, ms_normal, sigmaFlag, classes, mu1, labeled_index, model_path)

option_index = 4
labeled_index = [2, 3]
TwoMoons_SSL_Solutions(X, y, option_index, ms, ms_normal, sigmaFlag, classes, mu1, labeled_index, model_path)

option_index = 5
labeled_index = [0, 3]
TwoMoons_SSL_Solutions(X, y, option_index, ms, ms_normal, sigmaFlag, classes, mu1, labeled_index, model_path)

option_index = 6
labeled_index = [2, 5]
TwoMoons_SSL_Solutions(X, y, option_index, ms, ms_normal, sigmaFlag, classes, mu1, labeled_index, model_path)

option_index = 7
labeled_index = [6,9]
TwoMoons_SSL_Solutions(X, y, option_index, ms, ms_normal, sigmaFlag, classes, mu1, labeled_index, model_path)
