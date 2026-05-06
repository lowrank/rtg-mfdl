# Practice 08: Bayesian and Probabilistic Machine Learning

## 1. Theoretical Exercises

### 1.1 The MAP Estimate and Weight Decay
**Problem**: Prove that for a linear model $y = w^\top x + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$ and a Gaussian prior $w \sim \mathcal{N}(0, \lambda^{-1} I)$, the Maximum A Posteriori (MAP) estimate is equivalent to Ridge Regression (L2 regularization).
**Task**: Write the log-posterior $\log p(w|D)$ and show it is proportional to $-\|y - Xw\|^2 - \alpha \|w\|^2$.

### 1.2 HMC and Volume Preservation
**Problem**: The Leapfrog integrator is a **symplectic** integrator. 

1. What does it mean for a mapping to be volume-preserving in phase space?
2. Why is volume preservation (unit determinant of the Jacobian) crucial for the Metropolis-Hastings acceptance step in HMC?

### 1.3 NNGP and the Arc-Cosine Kernel
**Problem**: For a single-layer ReLU network $h(x) = \text{ReLU}(\sum w_i x_i)$, the NNGP kernel is $K(x, x') = \frac{1}{2\pi} \|x\| \|x'\| (\sin \theta + (\pi - \theta) \cos \theta)$, where $\theta = \arccos(\frac{x^\top x'}{\|x\| \|x'\|})$.

1. What happens to the correlation between $x$ and $x'$ as $\theta \to \pi$ (opposite directions)?
2. How does this compare to the standard RBF (Gaussian) kernel?

### 1.4 Variational Inference and Information Theory
**Problem**: The ELBO can be written as $\log p(x) - D_{KL}(q(z) \| p(z|x))$. 

1. Prove that the ELBO is always a lower bound on the log-evidence (i.e., $D_{KL} \ge 0$).
2. When is the bound tight?

### 1.5 Conformal Coverage Guarantee
**Problem**: Suppose you have $n=99$ calibration points and you want $90\%$ coverage ($\alpha=0.1$). 

1. Which sorted score $S_{(k)}$ should you pick as your threshold $\hat{q}$?
2. If the data is not exchangeable (e.g., there is a temporal trend), why does the coverage guarantee fail?

### 1.6 Reparameterization vs. Score Function
**Problem**: Contrast the variance of the REINFORCE gradient estimator $\nabla_\theta \mathbb{E}_{q_\theta}[f(z)]$ and the reparameterization estimator.
**Task**: Explain why the reparameterization trick is generally preferred for Variational Autoencoders (VAEs).

### 1.7 Normalizing Flows and Dimensionality
**Problem**: Standard Normalizing Flows require the mapping $f: \mathbb{R}^d \to \mathbb{R}^d$ to be a bijection (homeomorphism).

1. Can a Normalizing Flow change the dimensionality of the representation?
2. How do "Injective Flows" or "Manifold Flows" attempt to bypass this?

### 1.8 The Cold Posterior Mystery
**Problem**: Research the "Cold Posterior" effect.

1. Why do practitioners often use $p(w|D)^{1/T}$ with $T < 1$?
2. Give one hypothesis for why $T=1$ (the true Bayes posterior) often performs worse on test data than $T=0.1$.

---

## 2. Coding Practice

### 2.1 Metropolis-Hastings from Scratch
**Task**: Implement a simple Metropolis-Hastings sampler to sample from a 2D Mixture of Gaussians.

1. Visualize the chain's path.
2. Calculate the "Effective Sample Size" (ESS).

### 2.2 Conformal Regression on Synthetic Data
**Task**: 

1. Generate data from a heteroscedastic process: $y = x \sin(x) + \epsilon(x)$ where $\sigma(x)$ increases with $x$.
2. Train a simple MLP to predict the mean.
3. Use **Split Conformal Prediction** to generate $95\%$ confidence intervals.
4. Verify that the coverage is indeed $95\%$ across the test set.

---

## 3. Hints & Solutions

### 3.1 Hints

- **1.1**: Use Bayes' Rule: $p(w|D) \propto p(D|w)p(w)$. Take the negative log.
- **1.4**: Use Jensen's Inequality on the concave function $\log(\cdot)$.
- **2.2**: The non-conformity score for regression is typically $|y - \hat{y}|$.

### 3.2 Solutions (Brief)

- **1.2**: If the mapping is not volume-preserving, the Metropolis-Hastings ratio would require a complex Jacobian determinant term, which is computationally expensive to calculate for every step. Symplecticity ensures the determinant is 1.
- **1.5**: $\hat{q} = S_{(\lceil (n+1)(1-\alpha) \rceil)} = S_{(\lceil 100 \times 0.9 \rceil)} = S_{(90)}$. Pick the $90^{th}$ largest score.
- **1.7**: No, a homeomorphism must preserve dimensionality. Manifold flows typically use a padding or an encoder-decoder structure to change dims.

