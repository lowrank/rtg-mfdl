# Practice 09: Kernel Methods and RKHS

## 1. Theoretical Exercises

### 1.1 Positive Definiteness and Kernel Construction
**Problem**: Prove that if $k_1(x, y)$ and $k_2(x, y)$ are positive definite (PD) kernels, then:

1. $k(x, y) = k_1(x, y) + k_2(x, y)$ is PD.
2. $k(x, y) = k_1(x, y) \cdot k_2(x, y)$ is PD (Schur Product Theorem).
3. $k(x, y) = \exp(k_1(x, y))$ is PD.

### 1.2 The Reproducing Property
**Problem**: Let $\mathcal{H}$ be an RKHS with kernel $k$. Prove that:

1. $k(x, y) = \langle k(x, \cdot), k(y, \cdot) \rangle_{\mathcal{H}}$.
2. $\|k(x, \cdot)\|_{\mathcal{H}} = \sqrt{k(x, x)}$.
3. $|f(x)| \le \|f\|_{\mathcal{H}} \sqrt{k(x, x)}$ (Pointwise stability).

### 1.3 Mercer’s Theorem and the Gaussian Kernel
**Problem**: For the RBF kernel $k(x, y) = \exp(-\gamma \|x-y\|^2)$, the feature space is infinite-dimensional.

1. Why can't we explicitly write down the feature map $\phi(x)$ as a finite vector?
2. How does the choice of $\gamma$ affect the "smoothness" of functions in the RKHS?

### 1.4 The Representer Theorem with L2 Regularization
**Problem**: Consider Kernel Ridge Regression: $\min_f \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}}^2$.

1. Substitute $f(x) = \sum \alpha_j k(x_j, x)$ and derive the closed-form solution for the vector $\alpha$.
2. Compare this to the solution of standard Ridge Regression.

### 1.5 MMD and the Characteristic Property
**Problem**: A kernel is **characteristic** if the mean embedding $\mu_P = \int k(x, \cdot) dP(x)$ is injective.

1. Prove that the linear kernel $k(x, y) = x^\top y$ is NOT characteristic.
2. Provide two different distributions $P$ and $Q$ that have the same mean embedding under the linear kernel.

### 1.6 Bochner’s Theorem and RFF
**Problem**: For the Laplacian kernel $k(x, y) = \exp(-\lambda |x-y|)$ in 1D, find the spectral density $p(\omega)$ such that $k(\delta) = \int e^{i\omega \delta} p(\omega) d\omega$.
**Hint**: This is the Fourier transform of a double exponential.

### 1.7 Nyström vs. RFF
**Problem**: 

1. When is the Nyström method expected to perform better than RFF? (Hint: Consider the rank of the Gram matrix).
2. What is the computational complexity of the Nyström approximation for $m$ landmarks?

### 1.8 Convergence of RFF
**Problem**: Rahimi & Recht (2007) showed that the error $|k(x, y) - \hat{k}(x, y)|$ converges as $O(1/\sqrt{m})$.

1. Why is this rate independent of the dimensionality of $x$?
2. What is the downside of this slow convergence for high-precision applications?

---

## 2. Coding Practice

### 2.1 Approximating a Kernel with RFF
**Task**: Implement Random Fourier Features from scratch for the RBF kernel.

1. Generate 2D synthetic data.
2. Sample $\omega \sim \mathcal{N}(0, 2\gamma I)$.
3. Construct the feature map $\phi(x) = \sqrt{\frac{2}{m}} [\cos(\omega_1^\top x + b_1), \dots, \cos(\omega_m^\top x + b_m)]$.
4. Compare the approximated Gram matrix $\hat{K} = \Phi \Phi^\top$ with the exact $K$.

### 2.2 MMD Two-Sample Test
**Task**: Use the MMD formula to test if two samples come from the same distribution.

1. Sample $X \sim \mathcal{N}(0, 1)$ and $Y \sim \mathcal{N}(0.5, 1)$.
2. Calculate $MMD^2(X, Y)$ using the RBF kernel.
3. Perform a permutation test to determine the p-value.

---

## 3. Hints & Solutions

### 3.1 Hints

- **1.1**: Use the definition $\sum c_i c_j k(x_i, x_j) \ge 0$.
- **1.4**: The solution is $\alpha = (K + \lambda I)^{-1} y$, where $K$ is the Gram matrix.
- **2.1**: Remember to include the random phase $b \sim \text{Uniform}(0, 2\pi)$ if using the cosine-only form.

### 3.2 Solutions (Brief)

- **1.2.3**: By Cauchy-Schwarz: $|f(x)| = |\langle f, k_x \rangle| \le \|f\| \|k_x\| = \|f\| \sqrt{k(x, x)}$.
- **1.5.1**: For the linear kernel, $\mu_P = \int x dP(x) = \mathbb{E}_P[x]$. Any two distributions with the same mean (but different variances) will have the same embedding. Thus, it is not injective.
- **1.6**: $p(\omega) = \frac{1}{\pi} \frac{\lambda}{\lambda^2 + \omega^2}$ (Cauchy/Lorentzian distribution).

