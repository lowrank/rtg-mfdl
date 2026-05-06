# Topic 06: Practice, Exercises, and Solutions

This module provides theoretical and practical exercises in Geometric Deep Learning and Topological Data Analysis.

---

## 1. Theoretical Exercises

### 1.1 Permutation Equivariance in Sets
Let $f(X) = \sigma(XW + \mathbf{1}b^\top)$ be a linear layer followed by an activation.

1.  Is $f(X)$ permutation-equivariant?
2.  If not, how must $W$ be constrained to make $f(X)$ equivariant?

### 1.2 The Graph Laplacian and Heat Diffusion
The Graph Laplacian is $L = D - A$. The heat equation on a graph is $\frac{\partial h}{\partial t} = -L h$.

1.  Show that the solution is $h(t) = e^{-Lt} h(0)$.
2.  Relate $e^{-Lt}$ to the concept of a "Spectral Filter" in GNNs.

### 1.3 Euler Characteristic of a Simplical Complex
The Euler characteristic $\chi$ is given by $\chi = V - E + F$ (Vertices - Edges + Faces).

1.  Calculate $\chi$ for a tetrahedron.
2.  Calculate $\chi$ for a torus.
3.  Show that $\chi = \beta_0 - \beta_1 + \beta_2$, where $\beta_k$ are the Betti numbers (ranks of $H_k$).

### 1.4 Invariance vs. Equivariance
Let $G = \{e, g\}$ where $g$ is the reflection across the y-axis.

1.  Define the representation of $G$ on $\mathbb{R}^2$.
2.  Give an example of a $G$-invariant function $f: \mathbb{R}^2 \to \mathbb{R}$.
3.  Give an example of a $G$-equivariant function $f: \mathbb{R}^2 \to \mathbb{R}^2$.

### 1.5 The WL-Test and GNNs
Construct two graphs that are non-isomorphic but have the same node degree distribution. Show how the 1-WL test (and thus a basic GCN) fails to distinguish them.

### 1.6 Sinkhorn Algorithm Convergence
Show that the Sinkhorn algorithm is equivalent to alternating projections in the KL-divergence sense.

---

## 2. Coding Practice

### Task 2.1: Persistent Homology of a Noisy Circle

1.  Generate 100 points sampled from a circle with Gaussian noise.
2.  Use the `ripser` library to compute the $H_0$ and $H_1$ persistence diagrams.
3.  Identify the "persistent" $H_1$ feature. What is its approximate birth and death?

### Task 2.2: Verify Equivariance of a GCN

1.  Implement a simple GCN layer: $H' = \sigma(D^{-1/2} A D^{-1/2} H W)$.
2.  Generate a random graph and a random permutation matrix $P$.
3.  Verify that $\text{GCN}(P A P^\top, P H) = P \text{GCN}(A, H)$.

### Task 2.3: Geodesic Distances on a Mesh

1.  Load a 3D mesh (e.g., a "Stanford Bunny").
2.  Compute the adjacency matrix of the mesh.
3.  Use Dijkstra's algorithm to compute the geodesic distance between two distant points and compare it to the Euclidean distance.

---

## 3. Hints and Solutions

### Solution 1.1

1. No.
2. For $f(PX) = Pf(X)$ to hold for all permutation matrices $P$, $W$ must satisfy $PW = WP$. This implies $W = \alpha I + \beta \mathbf{1}\mathbf{1}^\top$.

### Hint 1.3
For a tetrahedron: $V=4, E=6, F=4$. $\chi = 4 - 6 + 4 = 2$.
For a torus: $\chi = 0$.

### Solution 2.2
In PyTorch, you can check this with `torch.allclose(out_perm, perm_out, atol=1e-6)`. If it fails, check if your normalization $D^{-1/2}$ is being computed correctly after the permutation.

### Hint 2.1
The $H_1$ feature (the loop) should have a birth near $\epsilon \approx 0$ (once edges form) and a death near $\epsilon \approx R$ (where $R$ is the radius, as the circle fills in).
