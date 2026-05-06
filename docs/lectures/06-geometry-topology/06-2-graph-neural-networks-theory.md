# 6.2 Graph Neural Networks and Spectral Theory

Graph Neural Networks (GNNs) have emerged as the standard architecture for modeling relational data, where information is structured as a set of entities (nodes) and their relationships (edges). Unlike standard Euclidean data like images or sequences, graphs lack a canonical ordering of nodes. This leads to the fundamental requirement of **permutation invariance** (for graph-level tasks) and **permutation equivariance** (for node-level tasks). This chapter explores the theoretical limits of GNNs through the lens of the Weisfeiler-Lehman hierarchy and provides a rigorous treatment of Spectral Graph Theory.

## 1. Graph Representation and Symmetries

A graph is represented as $G = (V, E)$, where $V$ is a set of $n$ nodes and $E \subseteq V \times V$ is a set of edges. We often associate each node $i \in V$ with a feature vector $\mathbf{x}_i \in \mathbb{R}^d$.

### 1.1 Permutation Invariance and Equivariance

Let $\mathbf{X} \in \mathbb{R}^{n \times d}$ be the matrix of node features and $\mathbf{A} \in \{0, 1\}^{n \times n}$ be the adjacency matrix. A permutation $\pi \in S_n$ acts on the graph by reordering the nodes. This is represented by a permutation matrix $\mathbf{P} \in \{0, 1\}^{n \times n}$.

The transformed features and adjacency matrix are:

- $\mathbf{X}' = \mathbf{P} \mathbf{X}$
- $\mathbf{A}' = \mathbf{P} \mathbf{A} \mathbf{P}^T$

!!! info "Definition (Invariance)"
    A function $f(\mathbf{A}, \mathbf{X})$ is permutation invariant if $f(\mathbf{P}\mathbf{A}\mathbf{P}^T, \mathbf{P}\mathbf{X}) = f(\mathbf{A}, \mathbf{X})$ for all $\mathbf{P} \in S_n$.
!!! info "Definition (Equivariance)"
    A function $\mathbf{F}(\mathbf{A}, \mathbf{X})$ is permutation equivariant if $\mathbf{F}(\mathbf{P}\mathbf{A}\mathbf{P}^T, \mathbf{P}\mathbf{X}) = \mathbf{P} \mathbf{F}(\mathbf{A}, \mathbf{X})$ for all $\mathbf{P} \in S_n$.

## 2. The Weisfeiler-Lehman Hierarchy and GNN Expressivity

The fundamental question of GNN theory is: "Which graphs can a GNN distinguish?" If two graphs are non-isomorphic but a GNN maps them to the same embedding, the GNN has "failed."

### 2.1 The 1-WL Algorithm (Color Refinement)

The 1-WL test is a heuristic for graph isomorphism.

1. Initialize color $c_i^{(0)} = 1$ for all nodes (or use node labels).
2. At step $t$, refine colors:

$$
c_i^{(t)} = \text{HASH}\left( c_i^{(t-1)}, \{ c_j^{(t-1)} : j \in \mathcal{N}(i) \} \right)
$$

   where $\{ \dots \}$ denotes the multiset of neighbor colors.

3. Terminate when the partition of nodes into color classes stabilizes.

If the sorted multiset of colors for graph $G_1$ differs from $G_2$, they are definitely non-isomorphic. If they are the same, they *might* be isomorphic.

### 2.2 MPNNs are Bounded by 1-WL

Standard Message Passing Neural Networks (MPNNs) use an aggregation function to update node states.

!!! success "Theorem (MPNN Expressivity Limit)"
    Any MPNN with continuous aggregation and update functions is at most as expressive as the 1-WL test in terms of distinguishing non-isomorphic graphs.

**Proof:**
We proceed by induction on the iteration number $t$. Let $h_v^{(t)}$ be the embedding of node $v$ in an MPNN and $c_v^{(t)}$ be its color in 1-WL.

**Inductive Hypothesis:** If $c_u^{(t)} = c_v^{(t)}$, then $h_u^{(t)} = h_v^{(t)}$ for any nodes $u, v$ in any graphs $G, G'$.

**Base Case ($t=0$):**
Standard initialization is $c_v^{(0)} = \text{label}(v)$ and $h_v^{(0)} = \mathbf{x}_v$. If we assume node features are the labels, then $c_u^{(0)} = c_v^{(0)} \implies h_u^{(0)} = h_v^{(0)}$.

**Inductive Step:**
Assume the hypothesis holds for $t-1$. Suppose $c_u^{(t)} = c_v^{(t)}$. In the 1-WL algorithm, the hash function is injective. Thus, $c_u^{(t)} = c_v^{(t)}$ implies:

1. $c_u^{(t-1)} = c_v^{(t-1)}$
2. $\{ c_i^{(t-1)} : i \in \mathcal{N}(u) \} = \{ c_j^{(t-1)} : j \in \mathcal{N}(v) \}$

By the inductive hypothesis:

1. $h_u^{(t-1)} = h_v^{(t-1)}$
2. The multiset of embeddings of neighbors must also be identical:

$$
\{ h_i^{(t-1)} : i \in \mathcal{N}(u) \} = \{ h_j^{(t-1)} : j \in \mathcal{N}(v) \}
$$

The MPNN update is $h_u^{(t)} = \text{UP}^{(t)}(h_u^{(t-1)}, \text{AGG}^{(t)}(\{ h_i^{(t-1)} : i \in \mathcal{N}(u) \}))$. Since both inputs to the deterministic functions $\text{UP}$ and $\text{AGG}$ are identical for $u$ and $v$, it must be that $h_u^{(t)} = h_v^{(t)}$.

Thus, if 1-WL cannot distinguish two graphs (mapping them to the same multiset of colors), no MPNN can distinguish them either. $\blacksquare$

### 2.3 The Graph Isomorphic Network (GIN)

While MPNNs are *bounded* by 1-WL, can we reach that bound? **GIN** achieves this by using a sum-aggregator and an MLP.

!!! success "Theorem (Injectivity of Sum-Aggregation)"
    Let $\mathcal{X}$ be a countable set. There exists a function $f: \mathcal{X} \to \mathbb{R}^n$ such that $h(M) = \sum_{x \in M} f(x)$ is injective for all multisets $M$ of finite size.

    **Proof Sketch:**
    For any countable set, we can map its elements to a set of numbers (e.g., primes) such that the sums over multisets are unique (by the fundamental theorem of arithmetic or properties of transcendental numbers). GIN uses an MLP to learn such an injective mapping.

### 2.4 Higher-Order $k$-WL Hierarchy

Since 1-WL fails on simple cases (like 3-regular graphs of different sizes), higher-order variants of the WL algorithm examine tuples of nodes:

- **2-WL**: Operates on pairs of nodes $(i, j)$.
- **k-WL**: Operates on $k$-tuples of nodes $(v_1, \dots, v_k)$.

**Hierarchy Property:** For $k \geq 2$, $(k+1)$-WL is strictly more powerful than $k$-WL. That is, there exist graphs distinguished by $(k+1)$-WL that are not distinguished by $k$-WL.

## 3. Spectral Graph Theory

Spectral GNNs define convolutions in the "frequency domain" of the graph.

### 3.1 The Graph Laplacian

Let $\mathbf{D}$ be the degree matrix ($D_{ii} = \sum_j A_{ij}$).

- **Combinatorial Laplacian:** $\mathbf{L} = \mathbf{D} - \mathbf{A}$
- **Normalized Laplacian:** $\tilde{\mathbf{L}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$

**Properties of $\tilde{\mathbf{L}}$:**

1. It is symmetric and positive semi-definite.
2. Its eigenvalues lie in $[0, 2]$.
3. The smallest eigenvalue is 0, with the constant vector $\mathbf{D}^{1/2} \mathbf{1}$ as an eigenvector.

### 3.2 Graph Fourier Transform

Let $\tilde{\mathbf{L}} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$ be the eigendecomposition. The eigenvectors $\mathbf{u}_l$ (columns of $\mathbf{U}$) are the "Graph Fourier Bases," and eigenvalues $\lambda_l$ are the "frequencies."
The **Graph Fourier Transform** of a signal $\mathbf{x}$ is $\hat{\mathbf{x}} = \mathbf{U}^T \mathbf{x}$.

A convolution in the spectral domain is defined as pointwise multiplication in the Fourier domain:

$$
\mathbf{y} = \mathbf{g}_\theta * \mathbf{x} = \mathbf{U} (\mathbf{U}^T \mathbf{g}_\theta \odot \mathbf{U}^T \mathbf{x}) = \mathbf{U} \mathbf{g}_\theta(\mathbf{\Lambda}) \mathbf{U}^T \mathbf{x}
$$

where $\mathbf{g}_\theta(\mathbf{\Lambda})$ is a diagonal matrix of filter coefficients.

### 3.3 Spectral Localization and Chebyshev Polynomials

A filter is **localized** if its action only depends on a local neighborhood.

!!! success "Theorem (Polynomial Filters are Localized)"
    A spectral filter of the form $g(\lambda) = \sum_{k=0}^K \theta_k \lambda^k$ is $K$-localized. That is, $(\mathbf{U} g(\mathbf{\Lambda}) \mathbf{U}^T)_{ij} = 0$ if the distance $d(i, j) > K$.

**Proof:**

$$
\mathbf{U} \left( \sum_{k=0}^K \theta_k \mathbf{\Lambda}^k \right) \mathbf{U}^T = \sum_{k=0}^K \theta_k (\mathbf{U} \mathbf{\Lambda} \mathbf{U}^T)^k = \sum_{k=0}^K \theta_k \mathbf{L}^k
$$

We know $(\mathbf{A}^k)_{ij}$ is the number of paths of length $k$ between $i$ and $j$. Similarly, the $ij$-th entry of $\mathbf{L}^k$ is non-zero only if there is a path of length at most $k$ between $i$ and $j$. Thus, if $d(i, j) > K$, all terms in the sum are zero. $\blacksquare$

To avoid the $O(n^3)$ cost of eigendecomposition, **ChebNet** uses Chebyshev polynomials $T_k(x)$ which can be computed recursively:

$$
T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
$$

This allows computing $\mathbf{L}^k \mathbf{x}$ in $O(k|E|)$ time.

## 4. Worked Examples

### Example 1: Failure of 1-WL on the Hexagon vs. Two Triangles
Consider:

1. $G_1$: A cycle of 6 nodes.
2. $G_2$: Two disjoint cycles of 3 nodes.
Both graphs are 2-regular. In 1-WL, every node initializes with the same color $1$. 

- In $G_1$, each node has 2 neighbors of color 1.
- In $G_2$, each node has 2 neighbors of color 1.
Thus, at step 1, 1-WL assigns the same new color to all nodes in both graphs. The partition never refines further. 1-WL fails to distinguish them despite $G_2$ being disconnected and $G_1$ being connected.

### Example 2: The Graph Laplacian of a Path Graph $P_3$
$P_3$ has 3 nodes and edges $(1, 2)$ and $(2, 3)$.

$$
\mathbf{A} = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}, \quad \mathbf{D} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

$$
\mathbf{L} = \mathbf{D} - \mathbf{A} = \begin{pmatrix} 1 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{pmatrix}
$$

Eigenvalues: $0, 1, 3$.
Eigenvectors:

- $\lambda=0: \mathbf{u}_0 = \frac{1}{\sqrt{3}}(1, 1, 1)^T$ (DC component)
- $\lambda=1: \mathbf{u}_1 = \frac{1}{\sqrt{2}}(1, 0, -1)^T$
- $\lambda=3: \mathbf{u}_2 = \frac{1}{\sqrt{6}}(1, -2, 1)^T$ (High frequency)

### Example 3: GCN as a First-Order Spectral Filter
Hammond et al. (2011) showed $g_\theta(\mathbf{L}) \mathbf{x} \approx \theta_0 \mathbf{x} + \theta_1 \mathbf{L} \mathbf{x}$. Kipf & Welling (2017) simplified this further:

1. Approximate $\lambda_{max} \approx 2$.
2. Set $\theta = \theta_0 = -\theta_1$.
3. Result: $\mathbf{y} = \theta (\mathbf{I} + \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}) \mathbf{x}$.
Using the "renormalization trick" $(\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I})$, we get the standard GCN layer.

## 5. Coding Demonstrations

### Demo 1: Implementing a GIN Layer (PyTorch)

```python
import torch
import torch.nn as nn

class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(1))
        # MLP for injective mapping
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, adj):
        # x: [N, in_dim], adj: [N, N] (sparse or dense)
        # 1. Neighbor aggregation (Sum)
        neighbor_sum = torch.matmul(adj, x)
        
        # 2. Combine with self
        out = (1 + self.epsilon) * x + neighbor_sum
        
        # 3. Apply MLP
        return self.mlp(out)

# Test on a small graph
x = torch.eye(3) # Identity features
adj = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
layer = GINLayer(3, 8)
print("Output shape:", layer(x, adj).shape)
```

```
Output shape: torch.Size([3, 8])
```

### Demo 2: Chebyshev Spectral Convolution

```python
import torch

def chebyshev_conv(x, L, theta, K):
    # L: normalized Laplacian [N, N]
    # theta: filter weights [K]
    # K: polynomial order
    
    N = x.shape[0]
    # Shift L to have eigenvalues in [-1, 1]
    L_tilde = L - torch.eye(N)
    
    # Chebyshev recursion: T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
    T_prev = x
    T_curr = L_tilde @ x
    
    out = theta[0] * T_prev + theta[1] * T_curr
    
    for k in range(2, K):
        T_next = 2 * L_tilde @ T_curr - T_prev
        out += theta[k] * T_next
        T_prev, T_curr = T_curr, T_next
        
    return out

# Example usage
L = torch.tensor([[1., -0.5, -0.5], [-0.5, 1., 0.], [-0.5, 0., 1.]])
x = torch.randn(3, 4)
theta = torch.randn(5) # 4th order polynomial
y = chebyshev_conv(x, L, theta, 5)
print("Cheby-filtered signal:\n", y)
```

```
Cheby-filtered signal:
 tensor([[ 1.3605, -1.0962, -3.7441,  0.3547],
        [ 1.4876, -2.5949, -1.1164,  1.1059],
        [ 0.7037,  1.3417, -4.1509, -0.3740]])
```

## 6. Advanced Theoretical Topic: The $k$-WL vs. MPNN Gap

Recent research has shown that while MPNNs are bounded by 1-WL, they often fall short of even 1-WL when using certain non-injective aggregators (like MEAN or MAX).

- **MEAN Aggregator**: Fails to distinguish multisets of different sizes if their distributions are the same (e.g., $\{1, 1\}$ and $\{1, 1, 1\}$).
- **MAX Aggregator**: Fails to distinguish multisets with the same unique elements (e.g., $\{1, 2\}$ and $\{1, 2, 2\}$).

To surpass 1-WL, we must move beyond local message passing. Techniques include:

1. **Subgraph GNNs**: Aggregating over subgraphs of each node.
2. **Graph Rewiring**: Adding virtual edges (e.g., via heat kernels or diffusion).
3. **Positional Encodings**: Adding Laplacian eigenvectors as features to break node symmetries.

## 7. Summary

Graph Neural Networks combine structural topology with node features through a process that can be understood both as **message passing** (spatial) and **filtering** (spectral).

1. **Spatial View**: GNNs refine node embeddings by aggregating neighbor information, a process limited by the 1-WL isomorphism test.
2. **Spectral View**: GNNs apply localized polynomial filters to the graph signal, approximating the Graph Fourier Transform without explicit eigendecomposition.
3. **Universal Design**: To achieve maximum expressivity, one must use injective aggregators (GIN) or higher-order structural motifs ($k$-WL).

## References

1. Xu, K., et al. (2019). How Powerful are Graph Neural Networks? ICLR.
2. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
3. Defferrard, M., et al. (2016). Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. NeurIPS.
4. Bronstein, M. M., et al. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.
5. Shuman, D. I., et al. (2013). The Emerging Field of Signal Processing on Graphs. IEEE Signal Processing Magazine.

