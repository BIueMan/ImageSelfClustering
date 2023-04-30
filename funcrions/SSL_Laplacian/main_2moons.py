import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from utils_datasets import generate_2moons
from utils import createAffinity, createAffinitySSL, createAffinityWNLL, ev_calculation_L, SpectralClusteringFromEV, Dirichlet_Clustering

import warnings
warnings.simplefilter("ignore", UserWarning)

plotFlag = False

if not os.path.exists('./results'):
    os.mkdir('./results')
    
model_path = r"./results/2moons"
if not os.path.exists(model_path):
    os.mkdir(model_path)
    os.mkdir(model_path + '/images')


# noise test
print("noise test")
n_samples = 500
classes = ('0', '1')
classNum = len(classes)
nodes_indx_list = range(0, n_samples)
number_of_labeled_nodes = 20
ms = n_samples
ms_normal = 15
sigmaFlag = 0
mu1 = 2.0
mu2 = (n_samples/number_of_labeled_nodes) - 1
noise_param_array = [0.05, 0.075, 0.1, 0.125, 0.15]

noise_param_array_len = len(noise_param_array)

for i in range(noise_param_array_len):
    noise_param = noise_param_array[i]
    print("Noise parameter", noise_param)

    X, y = generate_2moons(n_samples=n_samples, noise_param=noise_param, toPlot=plotFlag)
    X = torch.tensor(X)
    y = torch.tensor(y)
    X0 = X[:, 0]
    X0max_index = np.argmax(X0)

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
    ax.scatter(X[:, 0], X[:, 1], c=RCut_labels)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    savefig_path = model_path + "/images/rcut_us_iter_" + str(i) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    print("Spectral SSL")
    labeled_index = random.sample(nodes_indx_list, number_of_labeled_nodes)
    unlabeled_index = [indx for indx in nodes_indx_list if indx not in labeled_index]
    y_unlabeled = y[unlabeled_index]
    X0_unlabeled = X[unlabeled_index, 0]
    X0_unlabeled_index = np.argmax(X0_unlabeled)

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
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=RCut_labels)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    savefig_path = model_path + "/images/rcut_ssl_iter_" + str(i) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    print("Spectral WNLL")
    W_WNLL = createAffinityWNLL(X,  ms, ms_normal, sigmaFlag, labeled_index)
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
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=RCut_labels)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    savefig_path = model_path + "/images/rcut_wnll_iter_" + str(i) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    print("Dirichlet US")
    s0 = torch.sum(W_US, axis=0)
    D = torch.diag(s0)
    L = D - W_US
    clusteringRes, model_nmi, model_acc = Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes)
    print("NMI:", model_nmi)
    print("ACC:", model_acc)

    clusteringRes_max = clusteringRes[X0_unlabeled_index]
    if clusteringRes_max == 0:
        clusteringRes[clusteringRes == 1] = -1
        clusteringRes[clusteringRes == 0] = 1
        clusteringRes[clusteringRes == -1] = 0

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=clusteringRes)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    savefig_path = model_path + "/images/rcut_GL_US_iter_" + str(i) + ".png"
    plt.savefig(savefig_path)
    plt.show()


    print("Dirichlet SSL")
    s0 = torch.sum(W_ssl, axis=0)
    D = torch.diag(s0)
    L = D - W_ssl
    clusteringRes, model_nmi, model_acc = Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes)
    print("NMI:", model_nmi)
    print("ACC:", model_acc)

    clusteringRes_max = clusteringRes[X0_unlabeled_index]
    if clusteringRes_max == 0:
        clusteringRes[clusteringRes == 1] = -1
        clusteringRes[clusteringRes == 0] = 1
        clusteringRes[clusteringRes == -1] = 0

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=clusteringRes)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    savefig_path = model_path + "/images/rcut_GL_ssl_iter_" + str(i) + ".png"
    plt.savefig(savefig_path)
    plt.show()

    print("Dirichlet WNLL")
    s0 = torch.sum(W_WNLL, axis=0)
    D = torch.diag(s0)
    L = D - W_WNLL
    clusteringRes, model_nmi, model_acc = Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes)
    print("NMI:", model_nmi)
    print("ACC:", model_acc)

    clusteringRes_max = clusteringRes[X0_unlabeled_index]
    if clusteringRes_max == 0:
        clusteringRes[clusteringRes == 1] = -1
        clusteringRes[clusteringRes == 0] = 1
        clusteringRes[clusteringRes == -1] = 0

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[unlabeled_index, 0], X[unlabeled_index, 1], c=clusteringRes)
    # ax.set_title("Moons Dataset")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    savefig_path = model_path + "/images/rcut_GL_WNLL_iter_" + str(i) + ".png"
    plt.savefig(savefig_path)
    plt.show()


print("noise test- statistic")
drawNum = 100

nmi_us_mean_array = []
nmi_us_std_array = []
acc_us_mean_array = []
acc_us_std_array = []

nmi_ssl_mean_array = []
nmi_ssl_std_array = []
acc_ssl_mean_array = []
acc_ssl_std_array = []

nmi_wnll_mean_array = []
nmi_wnll_std_array = []
acc_wnll_mean_array = []
acc_wnll_std_array = []

nmi_GL_us_mean_array = []
nmi_GL_us_std_array = []
acc_GL_us_mean_array = []
acc_GL_us_std_array = []

nmi_GL_ssl_mean_array = []
nmi_GL_ssl_std_array = []
acc_GL_ssl_mean_array = []
acc_GL_ssl_std_array = []

nmi_GL_wnll_mean_array = []
nmi_GL_wnll_std_array = []
acc_GL_wnll_mean_array = []
acc_GL_wnll_std_array = []

for i in range(noise_param_array_len):
    nmi_us_array_curr_param = []
    acc_us_array_curr_param = []

    nmi_ssl_array_curr_param = []
    acc_ssl_array_curr_param = []

    nmi_wnll_array_curr_param = []
    acc_wnll_array_curr_param = []

    nmi_GL_us_array_curr_param = []
    acc_GL_us_array_curr_param = []

    nmi_GL_ssl_array_curr_param = []
    acc_GL_ssl_array_curr_param = []

    nmi_GL_wnll_array_curr_param = []
    acc_GL_wnll_array_curr_param = []

    for j in range(drawNum):
        noise_param = noise_param_array[i]
        X, y = generate_2moons(n_samples=n_samples, noise_param=noise_param, toPlot=plotFlag)
        X = torch.tensor(X)
        y = torch.tensor(y)

        # Spectral US
        W_US = createAffinity(X, ms, ms_normal, sigmaFlag)
        ev = ev_calculation_L(W_US, classNum)
        RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev, y, classNum)
        nmi_us_array_curr_param.append(model_nmi)
        acc_us_array_curr_param.append(model_acc)

        labeled_index = random.sample(nodes_indx_list, number_of_labeled_nodes)
        unlabeled_index = [indx for indx in nodes_indx_list if indx not in labeled_index]
        # Spectral SSL
        W_ssl = createAffinitySSL(X, y, ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2)
        ev = ev_calculation_L(W_ssl, classNum)
        ev_unlabeled = ev[unlabeled_index]
        RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev_unlabeled, y[unlabeled_index], classNum)
        nmi_ssl_array_curr_param.append(model_nmi)
        acc_ssl_array_curr_param.append(model_acc)

        # Spectral WNLL
        W_WNLL = createAffinityWNLL(X, ms, ms_normal, sigmaFlag, labeled_index)
        ev = ev_calculation_L(W_WNLL, classNum)
        ev_unlabeled = ev[unlabeled_index]
        RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev_unlabeled, y[unlabeled_index], classNum)
        nmi_wnll_array_curr_param.append(model_nmi)
        acc_wnll_array_curr_param.append(model_acc)

        # Dirichlet US
        s0 = torch.sum(W_US, axis=0)
        D = torch.diag(s0)
        L = D - W_US
        clusteringRes, model_nmi, model_acc = Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes)
        nmi_GL_us_array_curr_param.append(model_nmi)
        acc_GL_us_array_curr_param.append(model_acc)

        # Dirichlet SSL
        s0 = torch.sum(W_ssl, axis=0)
        D = torch.diag(s0)
        L = D - W_ssl
        clusteringRes, model_nmi, model_acc = Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes)
        nmi_GL_ssl_array_curr_param.append(model_nmi)
        acc_GL_ssl_array_curr_param.append(model_acc)

        # Dirichlet WNLL
        s0 = torch.sum(W_WNLL, axis=0)
        D = torch.diag(s0)
        L = D - W_WNLL
        clusteringRes, model_nmi, model_acc = Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes)
        nmi_GL_wnll_array_curr_param.append(model_nmi)
        acc_GL_wnll_array_curr_param.append(model_acc)

    # Spectral US
    nmi_us_array_curr_param = torch.Tensor(nmi_us_array_curr_param)
    acc_us_array_curr_param = torch.Tensor(acc_us_array_curr_param)
    nmi_us_array_curr_param_mean = torch.mean(nmi_us_array_curr_param)
    nmi_us_array_curr_param_std = torch.std(nmi_us_array_curr_param)
    nmi_us_mean_array.append(nmi_us_array_curr_param_mean)
    nmi_us_std_array.append(nmi_us_array_curr_param_std)
    acc_us_array_curr_param_mean = torch.mean(acc_us_array_curr_param)
    acc_us_array_curr_param_std = torch.std(acc_us_array_curr_param)
    acc_us_mean_array.append(acc_us_array_curr_param_mean)
    acc_us_std_array.append(acc_us_array_curr_param_std)

    # Spectral SSL
    nmi_ssl_array_curr_param = torch.Tensor(nmi_ssl_array_curr_param)
    acc_ssl_array_curr_param = torch.Tensor(acc_ssl_array_curr_param)
    nmi_ssl_array_curr_param_mean = torch.mean(nmi_ssl_array_curr_param)
    nmi_ssl_array_curr_param_std = torch.std(nmi_ssl_array_curr_param)
    nmi_ssl_mean_array.append(nmi_ssl_array_curr_param_mean)
    nmi_ssl_std_array.append(nmi_ssl_array_curr_param_std)
    acc_ssl_array_curr_param_mean = torch.mean(acc_ssl_array_curr_param)
    acc_ssl_array_curr_param_std = torch.std(acc_ssl_array_curr_param)
    acc_ssl_mean_array.append(acc_ssl_array_curr_param_mean)
    acc_ssl_std_array.append(acc_ssl_array_curr_param_std)

    # Spectral wnll
    nmi_wnll_array_curr_param = torch.Tensor(nmi_wnll_array_curr_param)
    acc_wnll_array_curr_param = torch.Tensor(acc_wnll_array_curr_param)
    nmi_wnll_array_curr_param_mean = torch.mean(nmi_wnll_array_curr_param)
    nmi_wnll_array_curr_param_std = torch.std(nmi_wnll_array_curr_param)
    nmi_wnll_mean_array.append(nmi_wnll_array_curr_param_mean)
    nmi_wnll_std_array.append(nmi_wnll_array_curr_param_std)
    acc_wnll_array_curr_param_mean = torch.mean(acc_wnll_array_curr_param)
    acc_wnll_array_curr_param_std = torch.std(acc_wnll_array_curr_param)
    acc_wnll_mean_array.append(acc_wnll_array_curr_param_mean)
    acc_wnll_std_array.append(acc_wnll_array_curr_param_std)


    # Dirichlet us
    nmi_GL_us_array_curr_param = torch.Tensor(nmi_GL_us_array_curr_param)
    acc_GL_us_array_curr_param = torch.Tensor(acc_GL_us_array_curr_param)
    nmi_GL_us_array_curr_param_mean = torch.mean(nmi_GL_us_array_curr_param)
    nmi_GL_us_array_curr_param_std = torch.std(nmi_GL_us_array_curr_param)
    nmi_GL_us_mean_array.append(nmi_GL_us_array_curr_param_mean)
    nmi_GL_us_std_array.append(nmi_GL_us_array_curr_param_std)
    acc_GL_us_array_curr_param_mean = torch.mean(acc_GL_us_array_curr_param)
    acc_GL_us_array_curr_param_std = torch.std(acc_GL_us_array_curr_param)
    acc_GL_us_mean_array.append(acc_GL_us_array_curr_param_mean)
    acc_GL_us_std_array.append(acc_GL_us_array_curr_param_std)

    # Dirichlet ssl
    nmi_GL_ssl_array_curr_param = torch.Tensor(nmi_GL_ssl_array_curr_param)
    acc_GL_ssl_array_curr_param = torch.Tensor(acc_GL_ssl_array_curr_param)
    nmi_GL_ssl_array_curr_param_mean = torch.mean(nmi_GL_ssl_array_curr_param)
    nmi_GL_ssl_array_curr_param_std = torch.std(nmi_GL_ssl_array_curr_param)
    nmi_GL_ssl_mean_array.append(nmi_GL_ssl_array_curr_param_mean)
    nmi_GL_ssl_std_array.append(nmi_GL_ssl_array_curr_param_std)
    acc_GL_ssl_array_curr_param_mean = torch.mean(acc_GL_ssl_array_curr_param)
    acc_GL_ssl_array_curr_param_std = torch.std(acc_GL_ssl_array_curr_param)
    acc_GL_ssl_mean_array.append(acc_GL_ssl_array_curr_param_mean)
    acc_GL_ssl_std_array.append(acc_GL_ssl_array_curr_param_std)

    # Dirichlet wnll
    nmi_GL_wnll_array_curr_param = torch.Tensor(nmi_GL_wnll_array_curr_param)
    acc_GL_wnll_array_curr_param = torch.Tensor(acc_GL_wnll_array_curr_param)
    nmi_GL_wnll_array_curr_param_mean = torch.mean(nmi_GL_wnll_array_curr_param)
    nmi_GL_wnll_array_curr_param_std = torch.std(nmi_GL_wnll_array_curr_param)
    nmi_GL_wnll_mean_array.append(nmi_GL_wnll_array_curr_param_mean)
    nmi_GL_wnll_std_array.append(nmi_GL_wnll_array_curr_param_std)
    acc_GL_wnll_array_curr_param_mean = torch.mean(acc_GL_wnll_array_curr_param)
    acc_GL_wnll_array_curr_param_std = torch.std(acc_GL_wnll_array_curr_param)
    acc_GL_wnll_mean_array.append(acc_GL_wnll_array_curr_param_mean)
    acc_GL_wnll_std_array.append(acc_GL_wnll_array_curr_param_std)


nmi_us_mean_array = torch.Tensor(nmi_us_mean_array)
nmi_us_std_array = torch.Tensor(nmi_us_std_array)
acc_us_mean_array = torch.Tensor(acc_us_mean_array)
acc_us_std_array = torch.Tensor(acc_us_std_array)

nmi_ssl_mean_array = torch.Tensor(nmi_ssl_mean_array)
nmi_ssl_std_array = torch.Tensor(nmi_ssl_std_array)
acc_ssl_mean_array = torch.Tensor(acc_ssl_mean_array)
acc_ssl_std_array = torch.Tensor(acc_ssl_std_array)

nmi_wnll_mean_array = torch.Tensor(nmi_wnll_mean_array)
nmi_wnll_std_array = torch.Tensor(nmi_wnll_std_array)
acc_wnll_mean_array = torch.Tensor(acc_wnll_mean_array)
acc_wnll_std_array = torch.Tensor(acc_wnll_std_array)

nmi_GL_us_mean_array = torch.Tensor(nmi_GL_us_mean_array)
nmi_GL_us_std_array = torch.Tensor(nmi_GL_us_std_array)
acc_GL_us_mean_array = torch.Tensor(acc_GL_us_mean_array)
acc_GL_us_std_array = torch.Tensor(acc_GL_us_std_array)

nmi_GL_ssl_mean_array = torch.Tensor(nmi_GL_ssl_mean_array)
nmi_GL_ssl_std_array = torch.Tensor(nmi_GL_ssl_std_array)
acc_GL_ssl_mean_array = torch.Tensor(acc_GL_ssl_mean_array)
acc_GL_ssl_std_array = torch.Tensor(acc_GL_ssl_std_array)

nmi_GL_wnll_mean_array = torch.Tensor(nmi_GL_wnll_mean_array)
nmi_GL_wnll_std_array = torch.Tensor(nmi_GL_wnll_std_array)
acc_GL_wnll_mean_array = torch.Tensor(acc_GL_wnll_mean_array)
acc_GL_wnll_std_array = torch.Tensor(acc_GL_wnll_std_array)

print("Spectral US")
print("NMI Mean : ", nmi_us_mean_array)
print("NMI STD : ", nmi_us_std_array)
print("ACC Mean : ", acc_us_mean_array)
print("ACC STD : ", acc_us_std_array)

print("Spectral WNLL")
print("NMI Mean : ", nmi_wnll_mean_array)
print("NMI STD : ", nmi_wnll_std_array)
print("ACC Mean : ", acc_wnll_mean_array)
print("ACC STD : ", acc_wnll_std_array)

print("Spectral ssl")
print("NMI Mean : ", nmi_ssl_mean_array)
print("NMI STD : ", nmi_ssl_std_array)
print("ACC Mean : ", acc_ssl_mean_array)
print("ACC STD : ", acc_ssl_std_array)

print("Dirichlet US")
print("NMI Mean : ", nmi_GL_us_mean_array)
print("NMI STD : ", nmi_GL_us_std_array)
print("ACC Mean : ", acc_GL_us_mean_array)
print("ACC STD : ", acc_GL_us_std_array)

print("Dirichlet WNLL")
print("NMI Mean : ", nmi_GL_wnll_mean_array)
print("NMI STD : ", nmi_GL_wnll_std_array)
print("ACC Mean : ", acc_GL_wnll_mean_array)
print("ACC STD : ", acc_GL_wnll_std_array)

print("Dirichlet ssl")
print("NMI Mean : ", nmi_GL_ssl_mean_array)
print("NMI STD : ", nmi_GL_ssl_std_array)
print("ACC Mean : ", acc_GL_ssl_mean_array)
print("ACC STD : ", acc_GL_ssl_std_array)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(noise_param_array, nmi_us_mean_array, label="$L$", color='red')
ax.fill_between(noise_param_array, nmi_us_mean_array - nmi_us_std_array, torch.minimum(nmi_us_mean_array + nmi_us_std_array,
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(noise_param_array, nmi_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(noise_param_array, nmi_wnll_mean_array - nmi_wnll_std_array, torch.minimum(nmi_wnll_mean_array + nmi_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(noise_param_array, nmi_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(noise_param_array, nmi_ssl_mean_array - nmi_ssl_std_array, torch.minimum(nmi_ssl_mean_array + nmi_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)

ax.set_xlabel("Noise std.", fontsize=22)
ax.set_ylabel("NMI", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/nmi_vs_noise_L.png"
plt.savefig(savefig_path)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(noise_param_array, nmi_GL_us_mean_array, label="$L$", color='red')
ax.fill_between(noise_param_array, nmi_GL_us_mean_array - nmi_GL_us_std_array, torch.minimum(nmi_GL_us_mean_array + nmi_GL_us_std_array,
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(noise_param_array, nmi_GL_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(noise_param_array, nmi_GL_wnll_mean_array - nmi_GL_wnll_std_array, torch.minimum(nmi_GL_wnll_mean_array + nmi_GL_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(noise_param_array, nmi_GL_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(noise_param_array, nmi_GL_ssl_mean_array - nmi_GL_ssl_std_array, torch.minimum(nmi_GL_ssl_mean_array + nmi_GL_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("Noise std.", fontsize=22)
ax.set_ylabel("NMI", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/nmi_vs_noise_GL.png"
plt.savefig(savefig_path)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(noise_param_array, acc_us_mean_array, label="$L$", color='red')
ax.fill_between(noise_param_array, acc_us_mean_array - acc_us_std_array, torch.minimum(acc_us_mean_array + acc_us_std_array,
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(noise_param_array, acc_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(noise_param_array, acc_wnll_mean_array - acc_wnll_std_array, torch.minimum(acc_wnll_mean_array + acc_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(noise_param_array, acc_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(noise_param_array, acc_ssl_mean_array - acc_ssl_std_array, torch.minimum(acc_ssl_mean_array + acc_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)

ax.set_xlabel("Noise std.", fontsize=22)
ax.set_ylabel("ACC", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/acc_vs_noise_L.png"
plt.savefig(savefig_path)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(noise_param_array, acc_GL_us_mean_array, label="$L$", color='red')
ax.fill_between(noise_param_array, acc_GL_us_mean_array - acc_GL_us_std_array, torch.minimum(acc_GL_us_mean_array + acc_GL_us_std_array,
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(noise_param_array, acc_GL_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(noise_param_array, acc_GL_wnll_mean_array - acc_GL_wnll_std_array, torch.minimum(acc_GL_wnll_mean_array + acc_GL_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(noise_param_array, acc_GL_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(noise_param_array, acc_GL_ssl_mean_array - acc_GL_ssl_std_array, torch.minimum(acc_GL_ssl_mean_array + acc_GL_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("Noise std.", fontsize=22)
ax.set_ylabel("ACC", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/acc_vs_noise_GL.png"
plt.savefig(savefig_path)
plt.show()



# number of labeled test
print("number of labeled test")
n_samples = 500
classes = ('0', '1')
classNum = len(classes)
nodes_indx_list = range(0, n_samples)
number_of_labeled_nodes_array = [0, 6, 10, 20, 30, 50, 70, 100]
ms = n_samples
ms_normal = 15
sigmaFlag = 0
mu1 = 2.0
noise_param = 0.1

number_of_labeled_nodes_array_len = len(number_of_labeled_nodes_array)

drawNum = 100

nmi_ssl_mean_array = []
nmi_ssl_std_array = []
acc_ssl_mean_array = []
acc_ssl_std_array = []

nmi_wnll_mean_array = []
nmi_wnll_std_array = []
acc_wnll_mean_array = []
acc_wnll_std_array = []

nmi_GL_us_mean_array = []
nmi_GL_us_std_array = []
acc_GL_us_mean_array = []
acc_GL_us_std_array = []

nmi_GL_ssl_mean_array = []
nmi_GL_ssl_std_array = []
acc_GL_ssl_mean_array = []
acc_GL_ssl_std_array = []

nmi_GL_wnll_mean_array = []
nmi_GL_wnll_std_array = []
acc_GL_wnll_mean_array = []
acc_GL_wnll_std_array = []

X, y = generate_2moons(n_samples=n_samples, noise_param=noise_param, toPlot=plotFlag)
X = torch.tensor(X)
y = torch.tensor(y)

for i in range(number_of_labeled_nodes_array_len):
    number_of_labeled_nodes = number_of_labeled_nodes_array[i]
    if number_of_labeled_nodes == 0:
        mu2 = 0
    else:
        mu2 = (n_samples / number_of_labeled_nodes) - 1

    nmi_ssl_array_curr_param = []
    acc_ssl_array_curr_param = []

    nmi_wnll_array_curr_param = []
    acc_wnll_array_curr_param = []

    nmi_GL_us_array_curr_param = []
    acc_GL_us_array_curr_param = []

    nmi_GL_ssl_array_curr_param = []
    acc_GL_ssl_array_curr_param = []

    nmi_GL_wnll_array_curr_param = []
    acc_GL_wnll_array_curr_param = []

    for j in range(drawNum):
        labeled_index = random.sample(nodes_indx_list, number_of_labeled_nodes)
        unlabeled_index = [indx for indx in nodes_indx_list if indx not in labeled_index]
        # Spectral SSL
        W_ssl = createAffinitySSL(X, y, ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2)
        ev = ev_calculation_L(W_ssl, classNum)
        ev_unlabeled = ev[unlabeled_index]
        RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev_unlabeled, y[unlabeled_index], classNum)
        nmi_ssl_array_curr_param.append(model_nmi)
        acc_ssl_array_curr_param.append(model_acc)

        # Spectral WNLL
        W_WNLL = createAffinityWNLL(X, ms, ms_normal, sigmaFlag, labeled_index)
        ev = ev_calculation_L(W_WNLL, classNum)
        ev_unlabeled = ev[unlabeled_index]
        RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev_unlabeled, y[unlabeled_index], classNum)
        nmi_wnll_array_curr_param.append(model_nmi)
        acc_wnll_array_curr_param.append(model_acc)

        # Dirichlet US
        W_US = createAffinity(X, ms, ms_normal, sigmaFlag)
        s0 = torch.sum(W_US, axis=0)
        D = torch.diag(s0)
        L = D - W_US
        clusteringRes, model_nmi, model_acc = Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes)
        nmi_GL_us_array_curr_param.append(model_nmi)
        acc_GL_us_array_curr_param.append(model_acc)

        # Dirichlet SSL
        s0 = torch.sum(W_ssl, axis=0)
        D = torch.diag(s0)
        L = D - W_ssl
        clusteringRes, model_nmi, model_acc = Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes)
        nmi_GL_ssl_array_curr_param.append(model_nmi)
        acc_GL_ssl_array_curr_param.append(model_acc)

        # Dirichlet WNLL
        s0 = torch.sum(W_WNLL, axis=0)
        D = torch.diag(s0)
        L = D - W_WNLL
        clusteringRes, model_nmi, model_acc = Dirichlet_Clustering(L, labeled_index, unlabeled_index, y, classes)
        nmi_GL_wnll_array_curr_param.append(model_nmi)
        acc_GL_wnll_array_curr_param.append(model_acc)

    # Spectral SSL
    nmi_ssl_array_curr_param = torch.Tensor(nmi_ssl_array_curr_param)
    acc_ssl_array_curr_param = torch.Tensor(acc_ssl_array_curr_param)
    nmi_ssl_array_curr_param_mean = torch.mean(nmi_ssl_array_curr_param)
    nmi_ssl_array_curr_param_std = torch.std(nmi_ssl_array_curr_param)
    nmi_ssl_mean_array.append(nmi_ssl_array_curr_param_mean)
    nmi_ssl_std_array.append(nmi_ssl_array_curr_param_std)
    acc_ssl_array_curr_param_mean = torch.mean(acc_ssl_array_curr_param)
    acc_ssl_array_curr_param_std = torch.std(acc_ssl_array_curr_param)
    acc_ssl_mean_array.append(acc_ssl_array_curr_param_mean)
    acc_ssl_std_array.append(acc_ssl_array_curr_param_std)

    # Spectral wnll
    nmi_wnll_array_curr_param = torch.Tensor(nmi_wnll_array_curr_param)
    acc_wnll_array_curr_param = torch.Tensor(acc_wnll_array_curr_param)
    nmi_wnll_array_curr_param_mean = torch.mean(nmi_wnll_array_curr_param)
    nmi_wnll_array_curr_param_std = torch.std(nmi_wnll_array_curr_param)
    nmi_wnll_mean_array.append(nmi_wnll_array_curr_param_mean)
    nmi_wnll_std_array.append(nmi_wnll_array_curr_param_std)
    acc_wnll_array_curr_param_mean = torch.mean(acc_wnll_array_curr_param)
    acc_wnll_array_curr_param_std = torch.std(acc_wnll_array_curr_param)
    acc_wnll_mean_array.append(acc_wnll_array_curr_param_mean)
    acc_wnll_std_array.append(acc_wnll_array_curr_param_std)

    # Dirichlet us
    nmi_GL_us_array_curr_param = torch.Tensor(nmi_GL_us_array_curr_param)
    acc_GL_us_array_curr_param = torch.Tensor(acc_GL_us_array_curr_param)
    nmi_GL_us_array_curr_param_mean = torch.mean(nmi_GL_us_array_curr_param)
    nmi_GL_us_array_curr_param_std = torch.std(nmi_GL_us_array_curr_param)
    nmi_GL_us_mean_array.append(nmi_GL_us_array_curr_param_mean)
    nmi_GL_us_std_array.append(nmi_GL_us_array_curr_param_std)
    acc_GL_us_array_curr_param_mean = torch.mean(acc_GL_us_array_curr_param)
    acc_GL_us_array_curr_param_std = torch.std(acc_GL_us_array_curr_param)
    acc_GL_us_mean_array.append(acc_GL_us_array_curr_param_mean)
    acc_GL_us_std_array.append(acc_GL_us_array_curr_param_std)

    # Dirichlet ssl
    nmi_GL_ssl_array_curr_param = torch.Tensor(nmi_GL_ssl_array_curr_param)
    acc_GL_ssl_array_curr_param = torch.Tensor(acc_GL_ssl_array_curr_param)
    nmi_GL_ssl_array_curr_param_mean = torch.mean(nmi_GL_ssl_array_curr_param)
    nmi_GL_ssl_array_curr_param_std = torch.std(nmi_GL_ssl_array_curr_param)
    nmi_GL_ssl_mean_array.append(nmi_GL_ssl_array_curr_param_mean)
    nmi_GL_ssl_std_array.append(nmi_GL_ssl_array_curr_param_std)
    acc_GL_ssl_array_curr_param_mean = torch.mean(acc_GL_ssl_array_curr_param)
    acc_GL_ssl_array_curr_param_std = torch.std(acc_GL_ssl_array_curr_param)
    acc_GL_ssl_mean_array.append(acc_GL_ssl_array_curr_param_mean)
    acc_GL_ssl_std_array.append(acc_GL_ssl_array_curr_param_std)

    # Dirichlet wnll
    nmi_GL_wnll_array_curr_param = torch.Tensor(nmi_GL_wnll_array_curr_param)
    acc_GL_wnll_array_curr_param = torch.Tensor(acc_GL_wnll_array_curr_param)
    nmi_GL_wnll_array_curr_param_mean = torch.mean(nmi_GL_wnll_array_curr_param)
    nmi_GL_wnll_array_curr_param_std = torch.std(nmi_GL_wnll_array_curr_param)
    nmi_GL_wnll_mean_array.append(nmi_GL_wnll_array_curr_param_mean)
    nmi_GL_wnll_std_array.append(nmi_GL_wnll_array_curr_param_std)
    acc_GL_wnll_array_curr_param_mean = torch.mean(acc_GL_wnll_array_curr_param)
    acc_GL_wnll_array_curr_param_std = torch.std(acc_GL_wnll_array_curr_param)
    acc_GL_wnll_mean_array.append(acc_GL_wnll_array_curr_param_mean)
    acc_GL_wnll_std_array.append(acc_GL_wnll_array_curr_param_std)

nmi_ssl_mean_array = torch.Tensor(nmi_ssl_mean_array)
nmi_ssl_std_array = torch.Tensor(nmi_ssl_std_array)
acc_ssl_mean_array = torch.Tensor(acc_ssl_mean_array)
acc_ssl_std_array = torch.Tensor(acc_ssl_std_array)

nmi_wnll_mean_array = torch.Tensor(nmi_wnll_mean_array)
nmi_wnll_std_array = torch.Tensor(nmi_wnll_std_array)
acc_wnll_mean_array = torch.Tensor(acc_wnll_mean_array)
acc_wnll_std_array = torch.Tensor(acc_wnll_std_array)

nmi_GL_us_mean_array = torch.Tensor(nmi_GL_us_mean_array)
nmi_GL_us_std_array = torch.Tensor(nmi_GL_us_std_array)
acc_GL_us_mean_array = torch.Tensor(acc_GL_us_mean_array)
acc_GL_us_std_array = torch.Tensor(acc_GL_us_std_array)

nmi_GL_ssl_mean_array = torch.Tensor(nmi_GL_ssl_mean_array)
nmi_GL_ssl_std_array = torch.Tensor(nmi_GL_ssl_std_array)
acc_GL_ssl_mean_array = torch.Tensor(acc_GL_ssl_mean_array)
acc_GL_ssl_std_array = torch.Tensor(acc_GL_ssl_std_array)

nmi_GL_wnll_mean_array = torch.Tensor(nmi_GL_wnll_mean_array)
nmi_GL_wnll_std_array = torch.Tensor(nmi_GL_wnll_std_array)
acc_GL_wnll_mean_array = torch.Tensor(acc_GL_wnll_mean_array)
acc_GL_wnll_std_array = torch.Tensor(acc_GL_wnll_std_array)


print("Spectral WNLL")
print("NMI Mean : ", nmi_wnll_mean_array)
print("NMI STD : ", nmi_wnll_std_array)
print("ACC Mean : ", acc_wnll_mean_array)
print("ACC STD : ", acc_wnll_std_array)

print("Spectral ssl")
print("NMI Mean : ", nmi_ssl_mean_array)
print("NMI STD : ", nmi_ssl_std_array)
print("ACC Mean : ", acc_ssl_mean_array)
print("ACC STD : ", acc_ssl_std_array)

print("Dirichlet US")
print("NMI Mean : ", nmi_GL_us_mean_array)
print("NMI STD : ", nmi_GL_us_std_array)
print("ACC Mean : ", acc_GL_us_mean_array)
print("ACC STD : ", acc_GL_us_std_array)

print("Dirichlet WNLL")
print("NMI Mean : ", nmi_GL_wnll_mean_array)
print("NMI STD : ", nmi_GL_wnll_std_array)
print("ACC Mean : ", acc_GL_wnll_mean_array)
print("ACC STD : ", acc_GL_wnll_std_array)

print("Dirichlet ssl")
print("NMI Mean : ", nmi_GL_ssl_mean_array)
print("NMI STD : ", nmi_GL_ssl_std_array)
print("ACC Mean : ", acc_GL_ssl_mean_array)
print("ACC STD : ", acc_GL_ssl_std_array)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array, nmi_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(number_of_labeled_nodes_array, nmi_wnll_mean_array - nmi_wnll_std_array, torch.minimum(nmi_wnll_mean_array + nmi_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(number_of_labeled_nodes_array, nmi_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(number_of_labeled_nodes_array, nmi_ssl_mean_array - nmi_ssl_std_array, torch.minimum(nmi_ssl_mean_array + nmi_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("NMI", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/nmi_vs_S_size_L.png"
plt.savefig(savefig_path)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array[1:], nmi_GL_us_mean_array[1:], label="$L$", color='red')
ax.fill_between(number_of_labeled_nodes_array[1:], nmi_GL_us_mean_array[1:] - nmi_GL_us_std_array[1:], torch.minimum(nmi_GL_us_mean_array[1:] + nmi_GL_us_std_array[1:],
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], nmi_GL_wnll_mean_array[1:], label="$L_{WNLL}$", color='green')
ax.fill_between(number_of_labeled_nodes_array[1:], nmi_GL_wnll_mean_array[1:] - nmi_GL_wnll_std_array[1:], torch.minimum(nmi_GL_wnll_mean_array[1:] + nmi_GL_wnll_std_array[1:],
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], nmi_GL_ssl_mean_array[1:], label="$L_{SSL}$", color='blue')
ax.fill_between(number_of_labeled_nodes_array[1:], nmi_GL_ssl_mean_array[1:] - nmi_GL_ssl_std_array[1:], torch.minimum(nmi_GL_ssl_mean_array[1:] + nmi_GL_ssl_std_array[1:],
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("NMI", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/nmi_vs_S_size_GL.png"
plt.savefig(savefig_path)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array, acc_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(number_of_labeled_nodes_array, acc_wnll_mean_array - acc_wnll_std_array, torch.minimum(acc_wnll_mean_array + acc_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(number_of_labeled_nodes_array, acc_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(number_of_labeled_nodes_array, acc_ssl_mean_array - acc_ssl_std_array, torch.minimum(acc_ssl_mean_array + acc_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("ACC", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/acc_vs_S_size_L.png"
plt.savefig(savefig_path)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array[1:], acc_GL_us_mean_array[1:], label="$L$", color='red')
ax.fill_between(number_of_labeled_nodes_array[1:], acc_GL_us_mean_array[1:] - acc_GL_us_std_array[1:], torch.minimum(acc_GL_us_mean_array[1:] + acc_GL_us_std_array[1:],
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], acc_GL_wnll_mean_array[1:], label="$L_{WNLL}$", color='green')
ax.fill_between(number_of_labeled_nodes_array[1:], acc_GL_wnll_mean_array[1:] - acc_GL_wnll_std_array[1:], torch.minimum(acc_GL_wnll_mean_array[1:] + acc_GL_wnll_std_array[1:],
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], acc_GL_ssl_mean_array[1:], label="$L_{SSL}$", color='blue')
ax.fill_between(number_of_labeled_nodes_array[1:], acc_GL_ssl_mean_array[1:] - acc_GL_ssl_std_array[1:], torch.minimum(acc_GL_ssl_mean_array[1:] + acc_GL_ssl_std_array[1:],
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("ACC", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/acc_vs_S_size_GL.png"
plt.savefig(savefig_path)
plt.show()

