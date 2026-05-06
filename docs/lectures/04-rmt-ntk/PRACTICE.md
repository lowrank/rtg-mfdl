# Topic 04: Practice, Exercises, and Solutions

This module provides theoretical and practical exercises to reinforce the concepts of Random Matrix Theory and the Neural Tangent Kernel.

---

## 1. Theoretical Exercises

### 1.1 The Stieltjes Transform of the Semicircle Law
The Stieltjes transform $G(z) = \mathbb{E}[\frac{1}{z - \lambda}]$ for Wigner's semicircle law $\rho(\lambda) = \frac{1}{2\pi} \sqrt{4 - \lambda^2}$ satisfies:

$$
G(z)^2 - z G(z) + 1 = 0
$$

1.  Solve this quadratic for $G(z)$. Which branch should you choose?
2.  Use the inversion formula $\rho(\lambda) = \lim_{\epsilon \to 0^+} \frac{-1}{\pi} \text{Im } G(\lambda + i\epsilon)$ to recover the semicircle density.

### 1.2 The Fourth Moment of Wigner Matrices
Using the moment method, show that $\mathbb{E}[\frac{1}{n} \text{Tr}(W^4)] = 2$ for a symmetric matrix with entries $\pm 1/\sqrt{n}$.
*Hint*: Identify the types of index sequences $(i_1, i_2, i_3, i_4, i_1)$ that contribute non-zero values.

### 1.3 Exact NTK for ReLU
Consider the 2-layer NTK $\Theta(x, x') = \Sigma^{(1)}(x, x') + \dot{\Sigma}^{(1)}(x, x') (x \cdot x')$.
For $\sigma(u) = \max(0, u)$ and $w \sim \mathcal{N}(0, I)$, we have:
$\Sigma^{(1)}(x, x') = \frac{\|x\| \|x'\|}{2\pi} (\sin \phi + (\pi - \phi) \cos \phi)$
$\dot{\Sigma}^{(1)}(x, x') = \frac{\pi - \phi}{2\pi}$
where $\cos \phi = \frac{x \cdot x'}{\|x\| \|x'\|}$.

1.  Prove the expression for $\dot{\Sigma}^{(1)}$.
2.  What happens to the kernel when $x = x'$?

### 1.4 Eigenvalue Spikes and Rank-1 Perturbations
Let $W$ be an $n \times n$ Wigner matrix and $P = \theta u u^\top$ be a rank-1 perturbation.
**BBP Transition**: Show that the largest eigenvalue of $W + P$ "pops out" of the semicircle (bulk) only if $\theta > 1$. What is its value if it pops out?

### 1.5 Concentration of the NTK
Show that the variance of the empirical NTK $\hat{\Theta}_m(x, x')$ decreases as $O(1/m)$, where $m$ is the width.

---

## 2. Coding Practice

### Task 2.1: Simulate the Marchenko-Pastur Law
Write a script to generate a random matrix $X \in \mathbb{R}^{p \times n}$, compute $S = \frac{1}{n} XX^\top$, and compare its eigenvalue histogram to the theoretical MP law for $\gamma = 0.2$ and $\gamma = 2.0$.

### Task 2.2: Spectral Analysis of Real Weights
Download a pre-trained ResNet or Transformer. Extract a weight matrix from a middle layer.

1.  Plot its singular value distribution.
2.  Does it look like the MP law? If not, what does the "tail" suggest about the information stored in the weights?

### Task 2.3: NTK vs. Finite MLP Training
Implement a 2-layer MLP in PyTorch.

1.  Compute the analytical NTK at initialization.
2.  Train the MLP on 100 points of a synthetic 1D function.
3.  Compute the NTK at the end of training.
4.  Measure the distance $\|\Theta_{end} - \Theta_{init}\|_F$ as you increase width $m \in \{16, 64, 256, 1024\}$.

---

## 3. Hints and Solutions

### Solution 1.1

1. $G(z) = \frac{z - \sqrt{z^2 - 4}}{2}$. We choose the branch such that $G(z) \sim 1/z$ as $z \to \infty$.
2. For $\lambda \in (-2, 2)$, $G(\lambda + i\epsilon) \approx \frac{\lambda - i\sqrt{4 - \lambda^2}}{2}$. The imaginary part is $-\frac{\sqrt{4 - \lambda^2}}{2}$. Multiplying by $-1/\pi$ gives the result.

### Hint 1.2
The paths are $(i, j, i, j, i)$, $(i, j, j, i, i)$, etc. There are $2n^2$ such paths in total for a $n \times n$ matrix, leading to $2n^2/n^2 = 2$.

### Solution 1.3

1. $\dot{\Sigma}^{(1)} = \mathbb{E}[\sigma'(w \cdot x) \sigma'(w \cdot x')]$. For ReLU, $\sigma'(u) = 1$ if $u > 0$ else 0. This is the probability that both $w \cdot x > 0$ and $w \cdot x' > 0$. For Gaussian $w$, this is $(\pi - \phi)/(2\pi)$.
2. If $x=x'$, $\phi=0$, so $\Theta(x, x) = \|x\|^2$.

### Solution 2.2
Real weights often show "Heavy Tails" or "Power Laws," deviating from RMT. This indicates that the matrices are not truly random but have learned specific structures (low-rank components).

