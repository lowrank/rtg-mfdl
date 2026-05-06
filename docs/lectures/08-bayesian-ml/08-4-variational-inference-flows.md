# 08-4 Variational Inference and Normalizing Flows: The Optimization Path to Bayesian Deep Learning

## 1. Introduction: From Sampling to Optimization

In the Bayesian machine learning ecosystem, we are faced with a fundamental choice when attempting to characterize the posterior distribution $P(\theta \mid \mathcal{D})$. The first option, which we explored in the previous lecture, is Markov Chain Monte Carlo (MCMC). MCMC is theoretically elegant because it provides asymptotically exact samples from the true posterior. However, for modern deep learning models with millions of parameters and massive datasets, the computational cost of MCMC—specifically the need for thousands of sequential gradient evaluations—often becomes a deal-breaker.

The second option is **Variational Inference (VI)**. Instead of trying to sample from the intractable posterior $P(\theta \mid \mathcal{D})$, we turn the inference problem into an **optimization** problem. We propose a family of tractable distributions $\mathcal{Q}$ (e.g., Gaussians) and look for the specific member $q^*(\theta) \in \mathcal{Q}$ that is "closest" to the true posterior. By choosing the Kullback-Leibler (KL) divergence as our distance metric, we arrive at an objective function known as the Evidence Lower Bound (ELBO).

This lecture provides a rigorous and exhaustive treatment of Variational Inference. we will derive the ELBO from first principles, explore the Mean-Field approximation, and discuss the "compactness" bias of VI. We will then transition to modern innovations: Stochastic Variational Inference (SVI) for scalability, the Reparameterization Trick for gradient estimation, and Normalizing Flows for constructing highly expressive variational families that move beyond the limitations of simple Gaussians.

## 2. Theoretical Foundations: The Variational Objective

### 2.1 The Evidence Lower Bound (ELBO)

The goal of VI is to minimize the KL divergence between our approximation $q(\theta)$ and the true posterior $p(\theta \mid \mathcal{D})$:

$$
q^*(\theta) = \arg\min_{q \in \mathcal{Q}} D_{KL}(q(\theta) \parallel p(\theta \mid \mathcal{D}))
$$

However, the definition of the KL divergence includes the posterior itself:

$$
D_{KL}(q \parallel p) = \int q(\theta) \log \frac{q(\theta)}{p(\theta \mid \mathcal{D})} \, d\theta = \int q(\theta) \log q(\theta) \, d\theta - \int q(\theta) \log p(\theta \mid \mathcal{D}) \, d\theta
$$

Using Bayes' rule $p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D}, \theta)}{p(\mathcal{D})}$, we see that the second term becomes:

$$
\int q(\theta) \log \frac{p(\mathcal{D}, \theta)}{p(\mathcal{D})} \, d\theta = \int q(\theta) \log p(\mathcal{D}, \theta) \, d\theta - \log p(\mathcal{D})
$$

Substituting this back into the KL expression:

$$
D_{KL}(q \parallel p) = \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\mathcal{D}, \theta)] + \log p(\mathcal{D})
$$

Rearranging the terms, we define the **ELBO**:

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \mathbb{E}_q[\log q(\theta)]
$$

Leading to the fundamental identity:

$$
\log p(\mathcal{D}) = \text{ELBO}(q) + D_{KL}(q(\theta) \parallel p(\theta \mid \mathcal{D}))
$$

!!! success "Theorem 2.1 (The ELBO Property)"
    Because $D_{KL}(q \parallel p) \geq 0$ for any distribution $q$, it follows that:

    $$
    \text{ELBO}(q) \leq \log p(\mathcal{D})
    $$

    Maximizing the ELBO is mathematically equivalent to minimizing the KL divergence to the posterior. Furthermore, the ELBO provides a lower bound on the "evidence" (marginal likelihood), which is useful for model comparison.

### 2.2 Functional Forms of the ELBO

There are two primary ways to write the ELBO, each offering a different intuition:

- **The Energy-Entropy View:**

    $$
    \text{ELBO}(q) = \mathbb{E}_{q(\theta)}[\log p(\mathcal{D}, \theta)] + H(q)
    $$

    Here, $H(q) = -\mathbb{E}_q[\log q]$ is the entropy. To maximize the ELBO, we want to find a $q$ that assigns high probability to high-joint-probability regions of the model (energy) while remaining as spread out as possible (entropy). This prevents $q$ from collapsing to a single point.

- **The Likelihood-Prior View:**

    $$
    \text{ELBO}(q) = \mathbb{E}_{q(\theta)}[\log p(\mathcal{D} \mid \theta)] - D_{KL}(q(\theta) \parallel p(\theta))
    $$

    This is the form most common in Variational Autoencoders (VAEs). It highlights a trade-off: maximize the expected log-likelihood of the data (make the model accurate) while keeping the posterior close to our prior beliefs.

## 3. Mean-Field Variational Inference

The most common way to make the optimization tractable is the **Mean-Field** assumption. We assume that the variational distribution factors into independent components for each parameter:

$$
q(\theta) = \prod_{j=1}^M q_j(\theta_j)
$$

While this assumption is almost certainly false (parameters in neural networks are highly correlated), it allows for a very efficient optimization algorithm called Coordinate Ascent Variational Inference (CAVI).

### 3.1 Coordinate Ascent Variational Inference (CAVI)

In CAVI, we update one factor $q_j$ at a time while keeping all other factors $q_{i \neq j}$ fixed.

!!! success "Theorem 3.1 (The Optimal Mean-Field Factor)"
    The factor $q_j^*(\theta_j)$ that maximizes the ELBO, given all other factors, is:

    $$
    q_j^*(\theta_j) \propto \exp\left( \mathbb{E}_{q_{-j}} [\log p(\mathcal{D}, \theta)] \right)
    $$

    where $\mathbb{E}_{q_{-j}}$ denotes the expectation over all parameters except $\theta_j$.

**Proof of Theorem 3.1 (Rigorous Derivation):**

Start with the ELBO and isolate terms that depend on $q_j$:

$$
\text{ELBO}(q_j) = \int \prod_i q_i \log p(\mathcal{D}, \theta) \, d\theta - \int \prod_i q_i \log \prod_i q_i \, d\theta
$$

Expand the second term (entropy of the product is the sum of entropies):

$$
\int \prod_i q_i \sum_i \log q_i \, d\theta = \sum_i \int q_i \log q_i \, d\theta_i
$$

Now isolate $q_j$:

$$
\text{ELBO}(q_j) = \int q_j \left( \int \prod_{i \neq j} q_i \log p(\mathcal{D}, \theta) \, d\theta_{-j} \right) d\theta_j - \int q_j \log q_j \, d\theta_j + \text{const}
$$

The inner integral is the expectation $\mathbb{E}_{q_{-j}} [\log p(\mathcal{D}, \theta)]$. Let's call this $f_j(\theta_j)$.

$$
\text{ELBO}(q_j) = \int q_j f_j(\theta_j) \, d\theta_j - \int q_j \log q_j \, d\theta_j
$$

$$
= \int q_j \log \frac{\exp(f_j(\theta_j))}{q_j} \, d\theta_j
$$

This expression is the negative KL divergence between $q_j$ and a distribution proportional to $\exp(f_j(\theta_j))$. Since KL divergence is minimized at 0 when the distributions are equal, the optimal $q_j$ is the normalized version of $\exp(f_j(\theta_j))$. $\blacksquare$

### 3.2 The Compactness Bias of VI

A significant drawback of using $D_{KL}(q \parallel p)$ is that it is "mode-seeking." Because $q$ must be zero whenever $p$ is zero (to avoid an infinite KL), the variational distribution tends to fit a single mode of a multi-modal posterior and severely underestimate the variance. This is known as the **compactness bias**. In Bayesian Neural Networks, this means that VI often provides overly confident (under-estimated) uncertainty estimates compared to HMC.

## 4. Scalable Inference: SVI and the Reparameterization Trick

CAVI requires iterating over every parameter and computing expectations over the full dataset. This does not scale. Modern Bayesian Deep Learning relies on **Stochastic Variational Inference (SVI)**.

### 4.1 Stochastic Variational Inference (SVI)

In SVI, we assume $q$ is parameterized by a set of variables $\phi$ (e.g., the mean and variance of a Gaussian). We optimize the ELBO using stochastic gradient descent on mini-batches of data:

$$
\nabla_\phi \text{ELBO} \approx \nabla_\phi \left[ \frac{N}{|B|} \sum_{i \in B} \mathbb{E}_q[\log p(x_i \mid \theta)] - D_{KL}(q_\phi(\theta) \parallel p(\theta)) \right]
$$

### 4.2 The Reparameterization Trick

The main challenge in SVI is computing the gradient of an expectation: $\nabla_\phi \mathbb{E}_{q_\phi(\theta)} [f(\theta)]$. We cannot simply move the gradient inside the expectation because the distribution $q$ depends on $\phi$.

The **Reparameterization Trick** (Kingma & Welling, 2013) solves this by expressing $\theta$ as a deterministic function of $\phi$ and an auxiliary noise variable $\epsilon \sim p(\epsilon)$:

$$
\theta = g_\phi(\epsilon)
$$

For a Gaussian $q_\phi(\theta) = \mathcal{N}(\mu, \sigma^2)$, we write $\theta = \mu + \sigma \epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$.
Then:

$$
\nabla_\phi \mathbb{E}_{q_\phi(\theta)} [f(\theta)] = \nabla_\phi \mathbb{E}_{p(\epsilon)} [f(g_\phi(\epsilon))] = \mathbb{E}_{p(\epsilon)} [\nabla_\phi f(g_\phi(\epsilon))]
$$

We can now compute an unbiased estimate of the gradient by sampling $\epsilon$. This trick is the foundation of the Variational Autoencoder and most modern Bayesian Neural Networks (e.g., "Bayes by Backprop").

## 5. Normalizing Flows: Expressive Posterior Approximation

Mean-field VI is limited because it cannot model correlations between parameters. Normalizing Flows (NFs) allow us to construct complex, non-Gaussian distributions by transforming a simple base distribution through a sequence of invertible mappings.

### 5.1 The Change of Variables Formula

Let $\mathbf{z}_0 \sim q_0(\mathbf{z}_0)$ be a simple base variable (e.g., a standard Gaussian). Let $f$ be an invertible, differentiable mapping $\mathbf{z}_1 = f(\mathbf{z}_0)$. The density of the transformed variable is:

$$
q_1(\mathbf{z}_1) = q_0(\mathbf{z}_0) \left| \det \frac{\partial f}{\partial \mathbf{z}_0} \right|^{-1}
$$

For a sequence of $K$ transformations $\mathbf{z}_K = f_K \circ f_{K-1} \circ \dots \circ f_1(\mathbf{z}_0)$, the log-density is:

$$
\log q_K(\mathbf{z}_K) = \log q_0(\mathbf{z}_0) - \sum_{k=1}^K \log \left| \det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}} \right|
$$

### 5.2 Desiderata for Flows in VI

To use flows in the ELBO objective, we need:

1. **Efficient Inversion:** (Optional for VI, but necessary for generative modeling).
2. **Efficient Jacobian Determinant:** Computing the determinant of a $D \times D$ matrix is $O(D^3)$, which is impossible for neural networks. We need architectures where the Jacobian is triangular (so the determinant is the product of the diagonal).

### 5.3 Flow Architectures

1. **Planar Flows:** $f(\mathbf{z}) = \mathbf{z} + \mathbf{u} \cdot \sigma(\mathbf{w}^T \mathbf{z} + b)$. The Jacobian determinant can be computed in $O(D)$ using the Matrix Determinant Lemma.
2. **Sylvester Flows:** A generalization of planar flows that allows for multi-rank updates.
3. **RealNVP / Glow:** Uses **Affine Coupling Layers**. We split the dimensions of $\mathbf{z}$ into two halves $(A, B)$. We leave $A$ unchanged and transform $B$ using a function of $A$:

   $$
   \mathbf{y}_A = \mathbf{z}_A
   $$

   $$
   \mathbf{y}_B = \mathbf{z}_B \odot \exp(s(\mathbf{z}_A)) + t(\mathbf{z}_A)
   $$

   The Jacobian is lower triangular, so the determinant is simply $\sum s(\mathbf{z}_A)$. This allows for extremely high-dimensional flows.

## 6. Worked Examples

### 6.1 Worked Example 1: ELBO for a Gaussian Mean

**Problem:** We have a single observation $x=10$ from a Gaussian $p(x \mid \mu) = \mathcal{N}(\mu, 1)$. Our prior is $p(\mu) = \mathcal{N}(0, 1)$. Find the optimal variational distribution $q(\mu) = \mathcal{N}(m, s^2)$.

**Solution:**
Log-joint: $\log p(x, \mu) = -\frac{1}{2}(10-\mu)^2 - \frac{1}{2}\mu^2 + \text{const}$.
ELBO: $\mathbb{E}_q[-\frac{1}{2}(100 - 20\mu + \mu^2) - \frac{1}{2}\mu^2] + \log s$.
Using $\mathbb{E}_q[\mu] = m$ and $\mathbb{E}_q[\mu^2] = m^2 + s^2$:
ELBO = $-50 + 10m - \frac{1}{2}(m^2+s^2) - \frac{1}{2}(m^2+s^2) + \log s$.
ELBO = $10m - m^2 - s^2 + \log s + \text{const}$.
$\frac{\partial \text{ELBO}}{\partial m} = 10 - 2m = 0 \Rightarrow m = 5$.
$\frac{\partial \text{ELBO}}{\partial s} = -2s + 1/s = 0 \Rightarrow s^2 = 1/2$.
The variational posterior is $\mathcal{N}(5, 0.5)$, which matches the true analytic posterior.

### 6.2 Worked Example 2: Mean-Field for Correlated Variables

**Problem:** $p(w_1, w_2) = \mathcal{N}(\mathbf{0}, \begin{pmatrix} 1 & 0.9 \\ 0.9 & 1 \end{pmatrix})$. Let $q(w_1, w_2) = q_1(w_1) q_2(w_2)$. What are the variances of $q_1$ and $q_2$?

**Solution:**
The KL divergence is minimized when $q$ captures the mode. Because $q$ is factorized, it must be an axis-aligned Gaussian. To stay "inside" the diagonal mass of the true posterior, $q$ must shrink.
Calculations show that the optimal variances are $1 - 0.9^2 = 0.19$.
Notice how different this is from the true marginal variance of 1.0. This demonstrates the "variance underestimation" problem in VI.

### 6.3 Worked Example 3: The Reparameterization Trick for Gamma Distributions

**Problem:** Can we reparameterize a Gamma distribution $\text{Gamma}(\alpha, \beta)$?

**Solution:**
Unlike the Gaussian, there is no simple location-scale transformation for the shape parameter $\alpha$. However, for large $\alpha$, we can use the Central Limit Theorem approximation or the inverse CDF method (which is often not differentiable). Modern approaches use more complex transformations or "implicit" reparameterization. This example shows why Gaussian variational families are so dominant: they are uniquely easy to reparameterize.

## 7. Coding Demos

### 7.1 Coding Demo 1: Stochastic Variational Inference in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Variational parameters for weights (Mean-Field Gaussian)
        self.w_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.w_rho = nn.Parameter(torch.ones(out_features, in_features) * -3)
        
    def forward(self, x):
        # Reparameterization trick
        sigma = F.softplus(self.w_rho)
        epsilon = torch.randn_like(self.w_mu)
        w = self.w_mu + sigma * epsilon
        return F.linear(x, w)

    def kl_divergence(self):
        # KL(N(mu, sig^2) || N(0, 1))
        sigma = F.softplus(self.w_rho)
        return 0.5 * torch.sum(sigma**2 + self.w_mu**2 - 1 - 2*torch.log(sigma))

# Optimization loop
model = BayesianLinear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for i in range(500):
    optimizer.zero_grad()
    # Mini-batch prediction
    y_pred = model(torch.randn(32, 10))
    # NLL term (MSE for Gaussian likelihood)
    nll = F.mse_loss(y_pred, torch.ones(32, 1) * 5.0)
    # KL term
    kl = model.kl_divergence() / 1000.0 # Scale by total data size
    loss = nll + kl
    loss.backward()
    optimizer.step()
```

### 7.2 Coding Demo 2: Normalizing Flow (RealNVP Coupling)

```python
class CouplingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.s_net = nn.Sequential(nn.Linear(dim//2, 16), nn.ReLU(), nn.Linear(16, dim//2), nn.Tanh())
        self.t_net = nn.Sequential(nn.Linear(dim//2, 16), nn.ReLU(), nn.Linear(16, dim//2))

    def forward(self, z):
        z_a, z_b = z[:, :self.dim//2], z[:, self.dim//2:]
        s = self.s_net(z_a)
        t = self.t_net(z_a)
        y_b = z_b * torch.exp(s) + t
        return torch.cat([z_a, y_b], dim=1), torch.sum(s, dim=1)

# A simple 2-layer flow
class RealNVP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = CouplingLayer(dim)
        self.c2 = CouplingLayer(dim) # Should swap dims in practice

    def log_prob(self, z):
        # Base log prob + log-det-jacobians
        # ...
        pass
```

## 8. Historical Context and Advanced Topics

### 8.1 The Rise of Variational Inference

Variational Inference traces its roots to statistical physics, where it was developed to approximate complex systems of interacting particles. In the 1990s, Michael Jordan, David Blei, and others introduced these methods to the machine learning community (e.g., for Latent Dirichlet Allocation). The modern era of VI began in 2013 with the simultaneous discovery of the VAE by Kingma & Welling and Rezende et al., which integrated neural networks directly into the variational framework.

### 8.2 Amortized Inference

In standard VI, we optimize a separate set of parameters $\phi$ for every single data point. This is slow. **Amortized Inference** uses an "encoder" neural network $q(\theta \mid x, \lambda)$ that learns to predict the optimal variational parameters directly from the data. This allows for near-instantaneous inference during test time.

### 8.3 Information Geometry and the Natural Gradient

The space of probability distributions is not Euclidean; it is a Riemannian manifold with the Fisher Information Metric. Standard gradient descent in the space of $\phi$ can be very slow if the geometry is "curved." **Natural Gradient Variational Inference** (Hoffman et al., 2013) uses the natural gradient $\tilde{\nabla}_\phi = F^{-1} \nabla_\phi$ to take steps that account for the information geometry, often leading to much faster convergence.

## 9. Conclusion: The Future of Optimization-Based Inference

Variational Inference has fundamentally changed Bayesian machine learning by making it compatible with the high-performance optimization tools of deep learning. While it lacks the asymptotic guarantees of MCMC, its ability to scale to billions of parameters via SVI and to capture complex geometries via Normalizing Flows makes it the pragmatic choice for most real-world AI systems. As we continue to integrate more expressive flows and more accurate gradient estimators, the gap between "exact" sampling and "approximate" optimization continues to close.

