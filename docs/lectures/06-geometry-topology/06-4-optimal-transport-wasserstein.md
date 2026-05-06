# 6.4 Optimal Transport and Wasserstein Metrics

Optimal Transport (OT) provides a principled geometric framework for comparing probability distributions. Unlike Kullback-Leibler (KL) divergence, OT distances incorporate the underlying metric geometry of the space, preventing vanishing gradients when distributions have disjoint supports.

## 1. Monge and Kantorovich Formulations

Given two probability measures $\mu$ on $\mathcal{X}$ and $\nu$ on $\mathcal{Y}$, and a cost function $c(x,y)$.

**Monge Formulation:**
Find a mapping $T: \mathcal{X} \to \mathcal{Y}$ such that $T_{\#} \mu = \nu$ (the pushforward of $\mu$ is $\nu$) minimizing:

$$
\inf_{T_{\#} \mu = \nu} \int_{\mathcal{X}} c(x, T(x)) d\mu(x)
$$

**Kantorovich Formulation:**
Monge's problem may have no solution if $\mu$ is discrete and $\nu$ is continuous. Kantorovich relaxed this by looking for a joint coupling $\pi \in \Pi(\mu, \nu)$ (marginals are $\mu$ and $\nu$):

$$
\inf_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) d\pi(x, y)
$$

When $c(x,y) = ||x - y||^p$, the $p$-th root of the cost is the $p$-Wasserstein distance $W_p(\mu, \nu)$.

## 2. Brenier's Theorem

A cornerstone of OT in $\mathbb{R}^d$ for quadratic costs ($p=2$) is Brenier's Theorem, which establishes that the optimal transport map is the gradient of a convex function.


!!! success "Theorem (Brenier's Theorem)"
    Let $\mu, \nu$ be probability measures on $\mathbb{R}^d$ with finite second moments. Suppose $\mu$ is absolutely continuous with respect to the Lebesgue measure (it has a density). Then for the quadratic cost $c(x,y) = \frac{1}{2}||x - y||^2$:
    
    
    1. There exists a unique optimal transport plan $\pi$.
    2. This plan is deterministic, induced by a unique transport map $T$, meaning $\pi = (Id \times T)_{\#} \mu$.
    3. $T = \nabla \varphi$ for some convex function $\varphi : \mathbb{R}^d \to \mathbb{R} \cup \{+\infty\}$.
    
    
**Rigorous Proof (Sketch of Kantorovitch Dual argument):**
The Kantorovich dual problem for cost $c(x,y)$ is:
    
$$
\sup_{\varphi \oplus \psi \leq c} \int \varphi(x) d\mu(x) + \int \psi(y) d\nu(y)
$$
    
For $c(x,y) = \frac{1}{2}||x - y||^2 = \frac{1}{2}||x||^2 + \frac{1}{2}||y||^2 - x \cdot y$, we can absorb the norms into the potentials. Let $\tilde{\varphi}(x) = \frac{1}{2}||x||^2 - \varphi(x)$ and $\tilde{\psi}(y) = \frac{1}{2}||y||^2 - \psi(y)$. The constraint becomes:
    
$$
\tilde{\varphi}(x) + \tilde{\psi}(y) \geq x \cdot y
$$
    
To maximize the dual objective, we choose $\tilde{\varphi}$ and $\tilde{\psi}$ to be as small as possible while satisfying the bound. This dictates that they must be convex conjugates (Legendre-Fenchel transforms) of each other:
    
$$
\tilde{\psi}(y) = \tilde{\varphi}^*(y) = \sup_{x} \{ x \cdot y - \tilde{\varphi}(x) \}
$$
    
Because $\tilde{\varphi}$ is the supremum of affine functions, it is a closed convex function. 
At optimality, the primal-dual relations (complementary slackness) imply that the optimal coupling $\pi$ is concentrated on the set where the inequality is tight:
    
$$
\tilde{\varphi}(x) + \tilde{\varphi}^*(y) = x \cdot y
$$
    
By the properties of convex analysis, this equality holds if and only if $y \in \partial \tilde{\varphi}(x)$, where $\partial$ denotes the subdifferential.
Since $\mu$ is absolutely continuous with respect to Lebesgue measure, by Rademacher's theorem, the convex function $\tilde{\varphi}$ is differentiable almost everywhere with respect to $\mu$.
Thus, the subdifferential is a singleton: $y = \nabla \tilde{\varphi}(x)$ for $\mu$-a.e. $x$.
    
This means the support of the optimal coupling $\pi$ is exactly the graph of the function $T = \nabla \tilde{\varphi}$. Because $T$ maps every $x$ to exactly one $y$, the coupling is deterministic, meaning $\pi = (Id \times T)_{\#} \mu$. 
Since $T$ must push $\mu$ to $\nu$, we have $T = \nabla \tilde{\varphi}$ as the unique optimal transport map, where $\tilde{\varphi}$ is convex. $\blacksquare$
    
## 3. Gromov-Wasserstein Distance
    
Standard Wasserstein distance requires both distributions to exist in the same metric space. Gromov-Wasserstein (GW) compares distributions across *different* metric spaces by comparing intra-space pairwise distances.
Given $(\mathcal{X}, d_{\mathcal{X}}, \mu)$ and $(\mathcal{Y}, d_{\mathcal{Y}}, \nu)$:
    
$$
GW_p(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \left( \int \int |d_{\mathcal{X}}(x, x') - d_{\mathcal{Y}}(y, y')|^p d\pi(x, y) d\pi(x', y') \right)^{1/p}
$$
    
This is a non-convex Quadratic Assignment Problem, often used to align point clouds of different dimensions or graph structures without predefined correspondence.
    
## 4. Worked Examples
    
### Example 1: 1D Wasserstein
For 1D distributions with CDFs $F(x)$ and $G(y)$, the $p$-Wasserstein distance has a closed form:
    
$$
W_p(\mu, \nu) = \left( \int_0^1 |F^{-1}(q) - G^{-1}(q)|^p dq \right)^{1/p}
$$
    
### Example 2: Discrete OT (Sinkhorn)
Given discrete measures $\mu = \sum u_i \delta_{x_i}$ and $\nu = \sum v_j \delta_{y_j}$, we minimize $\sum_{i,j} P_{ij} C_{ij}$ subject to $P \mathbf{1} = u$, $P^T \mathbf{1} = v$. Adding entropy regularization $\epsilon H(P)$ makes the problem strictly convex, solvable efficiently via Sinkhorn iterations: $P = \text{diag}(a) K \text{diag}(b)$ where $K = \exp(-C/\epsilon)$.
    
### Example 3: Gromov-Wasserstein for Graph Alignment
Two graphs with adjacency matrices $A, B$ can be aligned by setting the intra-distances to shortest paths $D_A, D_B$. GW finds a coupling $\pi$ that maps nodes in $A$ to nodes in $B$ such that the distances are preserved as much as possible, providing a probabilistic graph isomorphism tool.
    
## 5. Coding Demonstrations
    
### Demo 1: Sinkhorn Transport (POT library)
```python
import numpy as np
import ot
    
# Two 1D discrete distributions
a = np.array([0.5, 0.5])
b = np.array([0.2, 0.8])
# Cost matrix (squared distance)
M = np.array([[0.0, 1.0], [1.0, 0.0]])
    
# Exact OT
P_exact = ot.emd(a, b, M)
print("Exact Transport Plan:\n", P_exact)
    
# Regularized OT (Sinkhorn)
P_sinkhorn = ot.sinkhorn(a, b, M, reg=0.1)
print("Sinkhorn Transport Plan:\n", P_sinkhorn)
```
    
### Demo 2: 1D Closed Form Wasserstein
```python
import numpy as np
from scipy.stats import wasserstein_distance
    
# Samples from two distributions
u_samples = np.random.normal(0, 1, 100)
v_samples = np.random.normal(5, 2, 100)
    
# Compute 1-Wasserstein via scipy (uses 1D CDF inverse formula)
w_dist = wasserstein_distance(u_samples, v_samples)
print(f"1D Wasserstein Distance: {w_dist:.4f}")
```
