# 6.4 Optimal Transport and Wasserstein Metrics

Optimal Transport (OT) has revolutionized machine learning by providing a geometrically grounded framework for comparing probability distributions. Unlike the Kullback-Leibler (KL) divergence, which treats distributions as point-masses in an abstract space of measures, the Wasserstein distance (induced by OT) accounts for the underlying geometry of the sample space. This makes it particularly effective for tasks where the support of distributions may not overlap, such as generative modeling and domain adaptation.

## 1. Mathematical Formulations of Optimal Transport

Optimal transport asks: "What is the most efficient way to transform one distribution of mass into another, given a cost of movement?"

### 1.1 The Monge Formulation (1781)

Given two probability measures $\mu$ on $\mathcal{X}$ and $\nu$ on $\mathcal{Y}$, and a cost function $c(x, y): \mathcal{X} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$, Monge's problem seeks a transport map $T: \mathcal{X} \to \mathcal{Y}$ that minimizes the total cost while preserving mass.

Mass preservation is expressed through the **pushforward** operator $T_{\#} \mu = \nu$, defined as:
$$ \nu(B) = \mu(T^{-1}(B)) \quad \text{for all Borel sets } B \subseteq \mathcal{Y} $$

The optimization problem is:
$$ \min_{T_{\#} \mu = \nu} \int_{\mathcal{X}} c(x, T(x)) d\mu(x) $$

**Limitation:** If $\mu$ is a Dirac mass and $\nu$ is a sum of two Dirac masses, no map $T$ can satisfy $T_{\#} \mu = \nu$ because a single point cannot be split.

### 1.2 The Kantorovich Relaxation (1942)

Kantorovich generalized the problem by searching for a **coupling** (or transport plan) $\pi \in \Pi(\mu, \nu)$, where $\Pi(\mu, \nu)$ is the set of joint probability measures on $\mathcal{X} \times \mathcal{Y}$ whose marginals are $\mu$ and $\nu$.

$$ \min_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) d\pi(x, y) $$

This is a linear program in the space of measures. If $c(x, y) = \|x - y\|^p$, the $p$-th root of the optimal value is the **$p$-Wasserstein distance**, denoted $W_p(\mu, \nu)$.

## 2. Duality and Potentials

The Kantorovich problem is a constrained optimization. Its dual provides deep insights into the geometry of transport.

### 2.1 Kantorovich Duality

!!! success "Theorem (Kantorovich Duality)"
    For any lower semi-continuous cost $c$, the following duality holds:
    $$ \min_{\pi \in \Pi(\mu, \nu)} \int c d\pi = \sup_{(\varphi, \psi) \in \Phi_c} \int_{\mathcal{X}} \varphi(x) d\mu(x) + \int_{\mathcal{Y}} \psi(y) d\nu(y) $$
    where $\Phi_c = \{ (\varphi, \psi) \in L_1(d\mu) \times L_1(d\nu) : \varphi(x) + \psi(y) \leq c(x, y) \}$.

For a given $\varphi$, the best $\psi$ is given by the **$c$-transform**:
$$ \varphi^c(y) = \inf_{x \in \mathcal{X}} \{ c(x, y) - \varphi(x) \} $$

## 3. Brenier's Theorem: The Geometry of Quadratic Transport

For the quadratic cost $c(x, y) = \frac{1}{2}\|x - y\|^2$ in $\mathbb{R}^d$, the optimal transport map has an extraordinary structure.

!!! success "Theorem (Brenier's Theorem)"
    Let $\mu, \nu$ be probability measures on $\mathbb{R}^d$ with finite second moments. If $\mu$ has a density with respect to the Lebesgue measure, then:
    1. There exists a unique optimal transport plan $\pi$.
    2. This plan is induced by a map $T$, i.e., $\pi = (Id \times T)_{\#} \mu$.
    3. There exists a convex function $\varphi: \mathbb{R}^d \to \mathbb{R}$ such that $T(x) = \nabla \varphi(x)$ for $\mu$-almost all $x$.

**Rigorous Proof:**
1.  **Dual Potentials:** From the duality theorem, the optimal potentials $\varphi, \psi$ satisfy $\varphi(x) + \psi(y) \leq \frac{1}{2}\|x-y\|^2$.
2.  **Rearrangement:** Let $\tilde{\varphi}(x) = \frac{1}{2}\|x\|^2 - \varphi(x)$. Substituting this into the inequality:
    $$ (\frac{1}{2}\|x\|^2 - \tilde{\varphi}(x)) + \psi(y) \leq \frac{1}{2}\|x\|^2 + \frac{1}{2}\|y\|^2 - x \cdot y $$
    $$ \tilde{\varphi}(x) + (\frac{1}{2}\|y\|^2 - \psi(y)) \geq x \cdot y $$
    Let $\tilde{\psi}(y) = \frac{1}{2}\|y\|^2 - \psi(y)$. Then $\tilde{\varphi}(x) + \tilde{\psi}(y) \geq x \cdot y$.
3.  **Convexity:** At optimality, $\tilde{\psi}$ must be the Legendre-Fenchel conjugate of $\tilde{\varphi}$: $\tilde{\psi}(y) = \tilde{\varphi}^*(y) = \sup_x (x \cdot y - \tilde{\varphi}(x))$.
4.  **Complementary Slackness:** The optimal plan $\pi$ must be supported on the set where the dual constraint is tight: $\tilde{\varphi}(x) + \tilde{\varphi}^*(y) = x \cdot y$.
5.  **Differentiability:** For convex functions, the equality $\tilde{\varphi}(x) + \tilde{\varphi}^*(y) = x \cdot y$ implies $y \in \partial \tilde{\varphi}(x)$, where $\partial$ is the subdifferential. Since $\mu$ is absolutely continuous, Rademacher's theorem implies $\tilde{\varphi}$ is differentiable $\mu$-a.e. Thus $y = \nabla \tilde{\varphi}(x)$ is uniquely determined for almost all $x$. $\blacksquare$

### 3.2 The Monge-Ampère Equation

The condition $T_{\#} \mu = \nu$ can be expressed as a partial differential equation. If $\mu$ has density $f$ and $\nu$ has density $g$, then for a smooth $T = \nabla \varphi$:
$$ f(x) = g(\nabla \varphi(x)) \det(\nabla^2 \varphi(x)) $$
This is the **elliptic Monge-Ampère equation**, connecting OT to differential geometry.

## 4. Entropic Regularization and the Sinkhorn Algorithm

The standard OT problem is a large-scale linear program, which is slow to solve ($O(N^3)$). Entropic regularization transforms it into a strictly convex problem solvable in $O(N^2)$.

### 4.1 The Regularized Problem

We add a Kullback-Leibler penalty to the objective:
$$ \min_{\pi \in \Pi(\mu, \nu)} \int c d\pi + \epsilon KL(\pi \| \mu \otimes \nu) $$
As $\epsilon \to 0$, we recover the original OT cost. For $\epsilon > 0$, the optimal coupling has the form:
$$ \pi_{ij} = a_i K_{ij} b_j $$
where $K_{ij} = \exp(-C_{ij}/\epsilon)$ is the Gibbs kernel and $a, b$ are scaling vectors.

### 4.2 The Sinkhorn Scaling Algorithm

The marginal constraints $\pi \mathbf{1} = \mu$ and $\pi^T \mathbf{1} = \nu$ lead to the iterations:
1.  $a \leftarrow \mu / (K b)$
2.  $b \leftarrow \nu / (K^T a)$
This converges linearly to the optimal vectors $a, b$. In practice, this is highly parallelizable on GPUs.

## 5. Gromov-Wasserstein (GW) Distance

When comparing distributions across different spaces (e.g., aligning a 3D point cloud with a 2D graph), the cost function $c(x, y)$ is not well-defined. GW solves this by comparing intra-space distances.

Given $(\mathcal{X}, d_{\mathcal{X}}, \mu)$ and $(\mathcal{Y}, d_{\mathcal{Y}}, \nu)$, the $L_2$-GW distance is:
$$ GW^2(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X}^2 \times \mathcal{Y}^2} |d_{\mathcal{X}}(x, x') - d_{\mathcal{Y}}(y, y')|^2 d\pi(x, y) d\pi(x', y') $$

This is a **Quadratic Assignment Problem** (QAP), which is NP-hard in general but can be approximated using entropic regularization and alternating minimization.

## 6. Worked Examples

### Example 1: $W_2$ between Gaussians
Let $\mu = \mathcal{N}(m_1, \Sigma_1)$ and $\nu = \mathcal{N}(m_2, \Sigma_2)$. The 2-Wasserstein distance is:
$$ W_2^2(\mu, \nu) = \|m_1 - m_2\|^2 + \text{Tr}(\Sigma_1 + \Sigma_2 - 2(\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2})^{1/2}) $$
If the covariances commute, this simplifies to $\|m_1 - m_2\|^2 + \|\Sigma_1^{1/2} - \Sigma_2^{1/2}\|_F^2$.

### Example 2: Sinkhorn on a $2 \times 2$ Matrix
Let $\mu = [0.5, 0.5]$, $\nu = [0.1, 0.9]$, $C = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $\epsilon = 1$.
Then $K = \begin{pmatrix} 1 & e^{-1} \\ e^{-1} & 1 \end{pmatrix} \approx \begin{pmatrix} 1 & 0.37 \\ 0.37 & 1 \end{pmatrix}$.
Iterate $a, b$:
- Start $b = [1, 1]$.
- $a = \mu / (Kb) = [0.5/1.37, 0.5/1.37] \approx [0.36, 0.36]$.
- $b = \nu / (K^T a) = [0.1/(0.36*1.37), 0.9/(0.36*1.37)]$.
Eventually, the coupling $\pi$ concentrates mass on $(1, 2)$ to satisfy the unbalanced marginals.

### Example 3: 1D Quantile Function
For $\mu, \nu \in \mathcal{P}(\mathbb{R})$, let $F, G$ be their CDFs. The optimal map is $T(x) = G^{-1}(F(x))$. This is the only increasing map that pushes $\mu$ to $\nu$.

## 7. Coding Demonstrations

### Demo 1: Sinkhorn from Scratch in PyTorch

```python
import torch

def sinkhorn_normalized(mu, nu, C, eps=0.1, max_iter=100):
    # mu, nu: [N], [M] marginals
    # C: [N, M] cost matrix
    K = torch.exp(-C / eps)
    b = torch.ones_like(nu)
    
    for _ in range(max_iter):
        a = mu / (torch.matmul(K, b) + 1e-8)
        b = nu / (torch.matmul(K.t(), a) + 1e-8)
        
    P = torch.diag(a) @ K @ torch.diag(b)
    return P, torch.sum(P * C)

# Test
n, m = 5, 5
x = torch.randn(n, 2)
y = torch.randn(m, 2)
C = torch.cdist(x, y)**2
mu = torch.ones(n) / n
nu = torch.ones(m) / m

plan, dist = sinkhorn_normalized(mu, nu, C)
print("Wasserstein Distance (Entropic):", dist.item())
```

### Demo 2: WGAN-GP Loss (Conceptual)

In WGAN, we use the Kantorovich-Rubinstein duality for $W_1$:
$$ W_1(\mu, \nu) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim \mu}[f(x)] - \mathbb{E}_{y \sim \nu}[f(y)] $$
The 1-Lipschitz constraint is enforced via a **Gradient Penalty**.

```python
def wgan_gp_loss(discriminator, real_data, fake_data, lambda_gp=10.0):
    # Wasserstein Loss
    d_real = discriminator(real_data)
    d_fake = discriminator(fake_data)
    loss_wd = -torch.mean(d_real) + torch.mean(d_fake)
    
    # Gradient Penalty
    alpha = torch.rand(real_data.size(0), 1)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    d_int = discriminator(interpolates)
    
    grads = torch.autograd.grad(outputs=d_int, inputs=interpolates,
                                grad_outputs=torch.ones_like(d_int),
                                create_graph=True)[0]
    gp = lambda_gp * torch.mean((torch.norm(grads, p=2, dim=1) - 1)**2)
    
    return loss_wd + gp
```

## 8. Summary and Conclusion

Optimal Transport bridges the gap between probability theory and geometry.
1. **Monge-Kantorovich** defines the ground problem of mass movement.
2. **Brenier's Theorem** reveals that optimal transport maps are gradients of convex functions, linking OT to potential theory.
3. **Sinkhorn's Algorithm** makes OT computationally feasible for high-dimensional data.
4. **Gromov-Wasserstein** extends OT to the comparison of structural information in different spaces.

As a tool, OT allows neural networks to "feel" the metric structure of their data, leading to more robust models and geometrically meaningful latent spaces.

## References

1. Villani, C. (2003). Topics in Optimal Transportation. American Mathematical Society.
2. Villani, C. (2008). Optimal Transport: Old and New. Springer.
3. Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport. Foundations and Trends in Machine Learning.
4. Santambrogio, F. (2015). Optimal Transport for Applied Mathematicians. Birkhäuser.
5. Arjovsky, M., et al. (2017). Wasserstein Generative Adversarial Networks. ICML.
