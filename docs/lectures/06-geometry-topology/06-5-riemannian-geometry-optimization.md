# 6.5 Riemannian Geometry and Optimization

Many fundamental problems in machine learning and signal processing involve constraints that define non-Euclidean geometries. Examples include orthogonal weight matrices in Recurrent Neural Networks (RNNs), covariance matrices in Gaussian modeling, and subspaces in dimensionality reduction. Riemannian geometry provides the formal mathematical language to perform optimization directly on these "smooth surfaces" (manifolds), leading to algorithms that are often more stable and geometrically meaningful than projection-based Euclidean methods.

## 1. Smooth Manifolds and Riemannian Metrics

A **smooth manifold** $\mathcal{M}$ is a topological space that locally resembles Euclidean space $\mathbb{R}^d$. Formally, it is equipped with an **atlas** of charts that allow us to perform calculus on the surface.

### 1.1 Tangent Spaces

At every point $x \in \mathcal{M}$, we define a **tangent space** $T_x \mathcal{M}$, which is a vector space consisting of the velocity vectors of all smooth curves passing through $x$. For an $n$-dimensional manifold, $T_x \mathcal{M}$ is an $n$-dimensional vector space.

### 1.2 The Riemannian Metric

A **Riemannian metric** $g$ is a family of inner products $g_x: T_x \mathcal{M} \times T_x \mathcal{M} \to \mathbb{R}$ that varies smoothly with $x$. In local coordinates, it is represented by a symmetric positive-definite matrix $G(x)$:
$$ g_x(u, v) = u^T G(x) v $$
The metric defines the concept of length, angle, and volume on the manifold. The length of a curve $\gamma: [a, b] \to \mathcal{M}$ is:
$$ L(\gamma) = \int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t), \dot{\gamma}(t))} dt $$

## 2. Geodesics and the Exponential Map

**Geodesics** are the "straight lines" of Riemannian geometry—curves that locally minimize the distance between points.

### 2.1 The Geodesic Equation

A curve $\gamma(t)$ is a geodesic if its acceleration vector is purely normal to the manifold. In local coordinates, this is expressed via the **Christoffel symbols** $\Gamma_{ij}^k$:
$$ \ddot{\gamma}^k + \sum_{i,j} \Gamma_{ij}^k \dot{\gamma}^i \dot{\gamma}^j = 0 $$
where $\Gamma_{ij}^k = \frac{1}{2} \sum_l g^{kl} (\partial_j g_{il} + \partial_i g_{jl} - \partial_l g_{ij})$.

### 2.2 Exponential and Logarithmic Maps

- **Exponential Map ($\exp_x$):** Given $x \in \mathcal{M}$ and $v \in T_x \mathcal{M}$, $\exp_x(v)$ is the point reached by traveling along the geodesic starting at $x$ with initial velocity $v$ for time $t=1$.
- **Logarithmic Map ($\log_x$):** The inverse map, $\log_x(y) = v$, gives the tangent vector at $x$ that points toward $y$ with magnitude equal to the geodesic distance $d(x, y)$.

## 3. Riemannian Gradient Descent (RGD)

To minimize a cost function $f: \mathcal{M} \to \mathbb{R}$, we must account for the manifold's curvature.

### 3.1 The Riemannian Gradient

The Riemannian gradient $\text{grad} f(x)$ is the unique tangent vector in $T_x \mathcal{M}$ such that:
$$ g_x(\text{grad} f(x), v) = df(x)[v] \quad \text{for all } v \in T_x \mathcal{M} $$
For a manifold embedded in $\mathbb{R}^n$, the Riemannian gradient is often the orthogonal projection of the Euclidean gradient $\nabla f(x)$ onto the tangent space:
$$ \text{grad} f(x) = \text{Proj}_{T_x \mathcal{M}}(\nabla f(x)) $$

### 3.2 The Update Rule

The RGD update generalizes the Euclidean update $x_{k+1} = x_k - \eta \nabla f(x_k)$:
$$ x_{k+1} = \exp_{x_k}(- \eta \text{grad} f(x_k)) $$
In practice, computing the true exponential map (geodesics) can be expensive. We often use a **Retraction** $R_x: T_x \mathcal{M} \to \mathcal{M}$, which is a first-order approximation:
$$ x_{k+1} = R_{x_k}(- \eta \text{grad} f(x_k)) $$

## 4. Convergence Theory

Analysis of RGD depends on the manifold's curvature and **geodesic convexity**.

!!! info "Definition (Geodesic Convexity)"
    A function $f: \mathcal{M} \to \mathbb{R}$ is geodesically convex if for every geodesic $\gamma(t)$:
    $$ f(\gamma(t)) \leq (1-t)f(\gamma(0)) + t f(\gamma(1)) $$

!!! success "Theorem (Convergence of RGD)"
    Let $\mathcal{M}$ have sectional curvature bounded below by $K \leq 0$. Let $f$ be $L$-smooth and geodesically convex. With step size $\eta \leq 1/L$, RGD converges to the global minimum $x^*$ at a rate:
    $$ f(x_T) - f(x^*) \leq \frac{d^2(x_0, x^*)}{2\eta T} $$

**Rigorous Proof:**
We use the **Riemannian Hinge Triangle Inequality**. For a manifold with curvature $K \leq 0$, the distance between $x_{k+1} = \exp_{x_k}(v)$ and $x^*$ satisfies:
$$ d^2(x_{k+1}, x^*) \leq d^2(x_k, x^*) + \|v\|^2_{x_k} - 2 \langle \log_{x_k}(x^*), v \rangle_{x_k} $$
Let $v = -\eta \text{grad} f(x_k)$. Then:
$$ d^2(x_{k+1}, x^*) \leq d^2(x_k, x^*) + \eta^2 \|\text{grad} f(x_k)\|^2 + 2\eta \langle \log_{x_k}(x^*), \text{grad} f(x_k) \rangle $$
From geodesic convexity: $f(x^*) \geq f(x_k) + \langle \text{grad} f(x_k), \log_{x_k}(x^*) \rangle$.
Substituting this in:
$$ d^2(x_{k+1}, x^*) \leq d^2(x_k, x^*) + \eta^2 \|\text{grad} f(x_k)\|^2 + 2\eta (f(x^*) - f(x_k)) $$
By $L$-smoothness, $f(x_{k+1}) \leq f(x_k) - \frac{\eta}{2} \|\text{grad} f(x_k)\|^2$.
Summing over $T$ iterations and telescoping the $d^2$ terms yields the $O(1/T)$ rate. $\blacksquare$

## 5. Important Manifolds in Machine Learning

### 5.1 The Stiefel Manifold $St(n, p)$

The space of $n \times p$ matrices with orthonormal columns: $X^T X = I_p$.
- **Tangent Space:** $T_X St = \{ \Delta : \Delta^T X + X^T \Delta = 0 \}$.
- **Projection:** $\text{Proj}_X(Z) = Z - X \text{sym}(X^T Z)$, where $\text{sym}(A) = \frac{1}{2}(A + A^T)$.
- **Applications:** Orthogonal RNNs, PCA.

### 5.2 The SPD Manifold $\mathcal{P}_n$

The space of $n \times n$ symmetric positive-definite matrices.
- **Metric:** The Affine-Invariant metric $g_X(U, V) = \text{Tr}(X^{-1} U X^{-1} V)$.
- **Geodesic:** $X(t) = X^{1/2} \exp(t X^{-1/2} V X^{-1/2}) X^{1/2}$.
- **Applications:** Covariance tracking, Diffusion Tensor Imaging (DTI).

## 6. Worked Examples

### Example 1: Christoffel Symbols of the 2-Sphere
Using spherical coordinates $(\theta, \phi)$, the metric is $G = \begin{pmatrix} 1 & 0 \\ 0 & \sin^2 \theta \end{pmatrix}$.
The non-zero Christoffel symbols are:
- $\Gamma_{\phi \phi}^\theta = -\sin \theta \cos \theta$
- $\Gamma_{\theta \phi}^\phi = \Gamma_{\phi \theta}^\phi = \cot \theta$
This shows that moving along a line of constant latitude ($\phi$) requires a "centripetal force" (acceleration in $\theta$) to stay on the manifold.

### Example 2: Riemannian Gradient of PCA
Maximize $f(x) = x^T A x$ subject to $\|x\|=1$.
Euclidean gradient: $\nabla f = 2Ax$.
Riemannian gradient: $\text{grad} f = \text{Proj}_{T_x S}(\nabla f) = 2Ax - (2Ax^T x)x = 2(Ax - (x^T Ax)x)$.
Setting $\text{grad} f = 0$ yields $Ax = \lambda x$, the eigenvalue equation.

### Example 3: Parallel Transport on a Sphere
If you start at the equator and move a vector pointing North up to the North Pole, then down another longitude back to the equator, the vector will be rotated. This **Holonomy** measures the total curvature enclosed by the path.

## 7. Coding Demonstrations

### Demo 1: Optimization on the Stiefel Manifold (Ortho-RNN initialization)

We want to find a matrix that is close to a target $W$ but is strictly orthogonal.

```python
import torch

def stiefel_projection(W):
    # Project an arbitrary matrix to the Stiefel manifold using Procrustes
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U @ Vh

def rgd_stiefel(target, lr=0.1, iters=100):
    n, p = target.shape
    X = torch.eye(n, p, requires_grad=True)
    
    for _ in range(iters):
        loss = torch.norm(X - target)**2
        loss.backward()
        
        with torch.no_grad():
            # 1. Get Euclidean gradient
            grad_e = X.grad
            # 2. Project to Tangent Space
            # For Stiefel: G = G_e - X * sym(X^T * G_e)
            sym_part = 0.5 * (X.t() @ grad_e + grad_e.t() @ X)
            grad_r = grad_e - X @ sym_part
            
            # 3. Update with Retraction (QR or SVD)
            X.copy_(stiefel_projection(X - lr * grad_r))
            X.grad.zero_()
            
    return X

target = torch.randn(5, 3)
ortho_X = rgd_stiefel(target)
print("Orthogonality check (X^T X):\n", ortho_X.t() @ ortho_X)
```

### Demo 2: Hyperbolic SGD for Hierarchical Embeddings

The Poincaré ball manifold is used to embed trees with low distortion.

```python
def mobius_add(x, y):
    # Möbius addition in the Poincaré ball
    norm_x2 = torch.sum(x**2, dim=-1, keepdim=True)
    norm_y2 = torch.sum(y**2, dim=-1, keepdim=True)
    inner_xy = torch.sum(x * y, dim=-1, keepdim=True)
    
    num = (1 + 2*inner_xy + norm_y2) * x + (1 - norm_x2) * y
    den = 1 + 2*inner_xy + norm_x2 * norm_y2
    return num / den

def hyperbolic_exp_map(x, v):
    # Exp map at x=0 is just tanh
    # In general, involves Möbius transformations
    norm_v = torch.norm(v, p=2, dim=-1, keepdim=True)
    return mobius_add(x, torch.tanh(norm_v / 2) * (v / norm_v))
```

## 8. Summary and Future Directions

Riemannian geometry transforms constrained optimization into unconstrained optimization on a curved space.
1. **Metrics** define the local geometry.
2. **Retractions** provide efficient update steps.
3. **Curvature** affects the convergence rate and the existence of local minima.

Future frontiers include **Riemannian Adam**, optimization on **Infinite-Dimensional Manifolds** (like the space of probability densities), and **Gauge-Equivariant** neural networks that operate on curved manifolds.

## References

1. Absil, P. A., et al. (2008). Optimization Algorithms on Matrix Manifolds. Princeton University Press.
2. Boumal, N. (2023). An Introduction to Optimization on Smooth Manifolds. Cambridge University Press.
3. Jost, J. (2017). Riemannian Geometry and Geometric Analysis. Springer.
4. Ganea, O., et al. (2018). Hyperbolic Neural Networks. NeurIPS.
5. Bronstein, M. M., et al. (2021). Geometric Deep Learning.
