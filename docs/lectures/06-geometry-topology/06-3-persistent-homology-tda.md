# 6.3 Persistent Homology and Topological Data Analysis

Topological Data Analysis (TDA) analyzes the shape of data. Rather than looking at local geometries, TDA extracts invariant features such as connected components, holes, and voids. Persistent Homology is the workhorse of TDA.

## 1. Simplicial Complexes and Homology

A $k$-simplex is the convex hull of $k+1$ affinely independent points (e.g., a 0-simplex is a vertex, 1-simplex an edge, 2-simplex a triangle). A **simplicial complex** $\Sigma$ is a set of simplices closed under taking subsets.

Homology groups $H_k(\Sigma)$ characterize the $k$-dimensional holes in $\Sigma$. The Betti number $\beta_k = \text{rank}(H_k)$ counts these holes:

- $\beta_0$: Number of connected components.
- $\beta_1$: Number of 1D loops.
- $\beta_2$: Number of 2D voids.

## 2. Persistent Homology

Given a point cloud, we build a nested sequence of simplicial complexes (a **filtration**) parameterized by a scale factor $\epsilon$. For example, the Vietoris-Rips complex $VR(\epsilon)$ includes a simplex if all pairwise distances between its vertices are $\leq \epsilon$.

As $\epsilon$ increases, topological features (like loops) appear (are "born") and disappear (are "killed" by being filled in). Persistent homology tracks the multiset of intervals $[b_i, d_i)$, representing the birth and death of these features. This multiset forms a **Persistence Diagram**.

## 3. The Stability Theorem

A metric on persistence diagrams must be defined to compare topological summaries. The bottleneck distance between two diagrams $D_1, D_2$ is:

$$
W_\infty(D_1, D_2) = \inf_{\gamma} \sup_{x \in D_1} ||x - \gamma(x)||_\infty
$$

where $\gamma$ is a bijection between the points of $D_1$ and $D_2$ (allowing matching to the diagonal $y=x$).


!!! success "Theorem (Stability of Persistent Homology)"
    Let $f, g: X \to \mathbb{R}$ be continuous tame functions on a triangulable topological space $X$. Let $D(f)$ and $D(g)$ be their corresponding persistence diagrams. Then the bottleneck distance is bounded by the $L_\infty$ distance between the functions:
    
    
    $$
    W_\infty(D(f), D(g)) \leq ||f - g||_\infty = \sup_{x \in X} |f(x) - g(x)|
    $$
    
    
**Rigorous Proof:**
The proof relies on the algebraic structure of persistence modules. A persistence module $\mathbb{V}$ is a functor from the poset $(\mathbb{R}, \leq)$ to the category of vector spaces, meaning it consists of vector spaces $V_t$ for $t \in \mathbb{R}$ and linear maps $v_{s,t}: V_s \to V_t$ for $s \leq t$ satisfying $v_{t,r} \circ v_{s,t} = v_{s,r}$.
    
Let $f$ and $g$ be our functions, and let their sublevel sets be $X_t^f = f^{-1}((-\infty, t])$ and $X_t^g = g^{-1}((-\infty, t])$.
Assume $||f - g||_\infty = \epsilon$. This implies that for any $x \in X$:
    
$$
g(x) - \epsilon \leq f(x) \leq g(x) + \epsilon
$$
    
Therefore, the sublevel sets are intertwined:
    
$$
X_t^f \subseteq X_{t+\epsilon}^g \subseteq X_{t+2\epsilon}^f
$$
    
Applying the homology functor $H_k$ preserves inclusions, yielding maps between the persistence modules:
    
$$
H_k(X_t^f) \xrightarrow{\phi_t} H_k(X_{t+\epsilon}^g) \xrightarrow{\psi_{t+\epsilon}} H_k(X_{t+2\epsilon}^f)
$$
    
The composition $\psi_{t+\epsilon} \circ \phi_t$ is exactly the internal transition map of the persistence module for $f$ from $t$ to $t+2\epsilon$.
This structural intertwining algebraicly defines an $\epsilon$-interleaving between the persistence modules of $f$ and $g$.
    
The Algebraic Stability Theorem states that if two persistence modules are $\epsilon$-interleaved, the bottleneck distance between their barcodes is bounded by $\epsilon$. The existence of the morphisms $\phi$ and $\psi$ fulfilling the diagram implies that every interval in $D(f)$ can be matched to an interval in $D(g)$ shifted by at most $\epsilon$ in birth and death coordinates, or matched to the diagonal (for intervals shorter than $2\epsilon$).
Hence, $W_\infty(D(f), D(g)) \leq \epsilon = ||f - g||_\infty$. $\blacksquare$
    
## 4. TDA of Neural Network Weights
    
TDA is increasingly used to analyze the topology of neural network weight matrices or activation landscapes. Treating the weight matrix as a bipartite graph, one can construct filtrations on edge weights to observe when structural patterns emerge, finding that networks with higher generalization exhibit characteristic topological complexity.
    
## 5. Worked Examples
    
### Example 1: Betti Numbers of a Circle
A circle $S^1$ is topologically a single connected loop.
    
- $\beta_0 = 1$ (one component)
- $\beta_1 = 1$ (one hole)
- $\beta_2 = 0$
    
### Example 2: Vietoris-Rips on 3 Points
Let 3 points form an equilateral triangle of side length 1.
    
- For $\epsilon < 1$, we have 3 isolated vertices ($\beta_0 = 3$).
- At $\epsilon = 1$, 3 edges appear simultaneously forming a loop ($\beta_0 = 1, \beta_1 = 1$).
- To kill the hole, we need a 2-simplex (triangle). This occurs when the pairwise distance allows the entire triangle to be filled, which in VR is exactly when the largest edge is formed ($\epsilon=1$). However, in the Čech complex, the radius to cover the triangle is $1/\sqrt{3} \approx 0.57$.
    
### Example 3: Bottleneck Distance
Diagram 1: $(1, 3)$, $(2, 4)$. Diagram 2: $(1.1, 2.9)$, $(2, 5)$.
Matching $(1,3)$ to $(1.1, 2.9)$ yields max diff $0.1$.
Matching $(2,4)$ to $(2,5)$ yields max diff $1.0$.
Thus, $W_\infty = 1.0$.
    
## 6. Coding Demonstrations
    
### Demo 1: Computing Persistence with Ripser
```python
import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt
    
# Create a noisy circle point cloud
theta = np.linspace(0, 2*np.pi, 50)
x = np.cos(theta) + np.random.normal(0, 0.1, 50)
y = np.sin(theta) + np.random.normal(0, 0.1, 50)
data = np.column_stack([x, y])
    
# Compute Persistent Homology up to 1D holes
result = ripser(data, maxdim=1)
dgms = result['dgms']
    
print("H0 features (components):", len(dgms[0]))
print("H1 features (loops):", len(dgms[1]))
```
    
### Demo 2: TDA on a Weight Matrix
```python
import numpy as np
from ripser import ripser
    
# Mock weight matrix (e.g. 10x10 Linear layer)
W = np.random.randn(10, 10)
# Use 1 - |W| as a distance metric (highly correlated = short distance)
distance_matrix = 1.0 - np.abs(np.corrcoef(W))
    
# Rips requires a distance matrix
result = ripser(distance_matrix, distance_matrix=True, maxdim=1)
h1_bars = result['dgms'][1]
    
if len(h1_bars) > 0:
persistence = h1_bars[:, 1] - h1_bars[:, 0]
print(f"Max H1 Persistence in weights: {persistence.max():.4f}")
```
