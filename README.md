# Variational methods - Course project

| Pre-label patches      | Labels Patches       |
| :-------------: | :-------------: |
| <img src="https://github.com/BIueMan/ImageSelfClustering/blob/main/images_for_readme/weighted_nonlocal_laplacian_5.png" width="300"> | <img src="https://github.com/BIueMan/ImageSelfClustering/blob/main/images_for_readme/weighted_nonlocal_laplacian_6.png" width="300"> |

# Summarize
Spectral clustering is a technique for clustering data based on their similarity. It transforms the data into a new space where clustering is easier. This transformation is done using a Laplacian graph, which represents the connections between the data points.
The graph is used to create a similarity matrix, and the $n$ most dominant eigenvectors of this matrix are extracted to create a low-dimensional embedding of the data. This embedding is then used as input to the clustering algorithm.

In this project we are usint Semi Superviced Laplacian in order to label patches of a splited image, in order to label simulare object in the image.

# Introduction
## Spectral clustering
![Graph Laplacian Diagram](https://github.com/BIueMan/ImageSelfClustering/raw/main/images_for_readme/laplasian_diagram.png)

Let $X={x_i} \text{ ; } i\in[1,N]$ be a set of $N$ data points, with a known dimension $x_i \in \mathbb{R}^d$. A weight function is used to compare the data points and create a $N\times N$ symmetric matrix that holds the relations between them.
$$W_{ij} = f(x_i,x_j) = W_{ji}$$
Given a weighted adjacency matrix $W$ of an undirected graph, the Laplacian matrix $L$ can be computed using equations for $L$ and $D$, where $D$ is the degree matrix of the graph.
$$L = D - W$$
$$D_{ii} = \sum_{j=1}^{N} W_{ij}$$
To extract the $n$ most dominant eigenvectors, each representing a low-dimensional embedding of the data points, we use spectral clustering.
For a given data point $x_{ij}$, we pick the eigenvector $v_k$ that is closest to it. This clustering technique is particularly useful when the clusters are not well-defined in the original space.
$$\text{cluster}\left(x_i\right)=\text{argmin}_j\left||v_j-x_i\right||_2$$

## SSL - Semi superviced Laplacian
Similar to the spectral clustering discussed in the previous sections we will add a small supervised element to the affinity matrix.

Given an affinity matrix $W$ that is calculated without the labeled information, we will use the labeled info to create a second matrix $W_{lables}$. This matrix will be based on the labeled portion of the data.
Let there be a subset of the data $S$ the labeled data in this group can belong to $K$ mutually exclusive groups

$$\begin{aligned}
\bigcup_{k=1}^K = S \subset X \\
\bigcap_{k=1}^KS_K= \emptyset
\end{aligned}$$


Now each data point will get a value according to the group it belongs (or doesn't belong)

$$W_{labels}(i,j) = \begin{cases} 
\max(W) & x_i, x_j \in S_k \\
-\frac{2}{a}W_{ij} & x_i \in S_k ; x_j \in S_l ; k \neq l \\
W_{ij} & x_i \in S_k ; x_j \in X \setminus S \\
W_{ij} & x_j \in S_k ; x_i \in X \setminus S \\
0 & x_i, x_j \in X \setminus S \end{cases}$$

Where $a$ is defined as the ration between the groups 
$$a=\frac{|X|}{|S|}-1$$
We will combine both matrices into the $W_{SLL}$ matrix
$$W_{SLL}= 2W + a\cdot W_{labels}$$
Now the matrix is increasing the weights associated with the same sub group, and completely eliminates the affinity weight between data points of different subgroups.

# Results

| Pre-label patches      | Labels Patches       |
| :-------------: | :-------------: |
| <img src="https://github.com/BIueMan/ImageSelfClustering/blob/main/images_for_readme/weighted_nonlocal_laplacian_1.png" width="300"> | <img src="https://github.com/BIueMan/ImageSelfClustering/blob/main/images_for_readme/weighted_nonlocal_laplacian_2.png" width="300"> |
| <img src="https://github.com/BIueMan/ImageSelfClustering/blob/main/images_for_readme/weighted_nonlocal_laplacian_3.png" width="300"> | <img src="https://github.com/BIueMan/ImageSelfClustering/blob/main/images_for_readme/weighted_nonlocal_laplacian_4.png" width="300"> |
| <img src="https://github.com/BIueMan/ImageSelfClustering/blob/main/images_for_readme/weighted_nonlocal_laplacian_5.png" width="300"> | <img src="https://github.com/BIueMan/ImageSelfClustering/blob/main/images_for_readme/weighted_nonlocal_laplacian_6.png" width="300"> |
