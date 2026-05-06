# Topic 05: Practice, Exercises, and Solutions

This module provides theoretical and practical exercises to reinforce the concepts of Information Theory in Deep Learning.

---

## 1. Theoretical Exercises

### 1.1 Entropy of a Mixture
Let $p(x) = \sum_i \pi_i p_i(x)$ be a mixture of $K$ distributions.

1.  Prove that $h(X) \ge \sum \pi_i h(X_i)$.
2.  When does equality hold?

### 1.2 KL-Divergence Between Gaussians
Derive the closed-form expression for $D_{KL}(P \| Q)$ where $P = \mathcal{N}(\mu_0, \Sigma_0)$ and $Q = \mathcal{N}(\mu_1, \Sigma_1)$ in $d$ dimensions.
*Hint*: Use the property that $\mathbb{E}_P[x^\top A x] = \text{Tr}(A\Sigma_0) + \mu_0^\top A \mu_0$.

### 1.3 MI and Correlation
For a bivariate Gaussian with correlation $\rho$, we found $I(X; Y) = -\frac{1}{2} \log(1 - \rho^2)$.

1.  Show that $I(X; Y) = 0$ if $\rho = 0$.
2.  Compare this to the case of two non-Gaussian variables where $\rho=0$ but $I(X; Y) > 0$. Provide an example.

### 1.4 The Variational Bound for MI
Prove the Donsker-Varadhan lower bound:
$D_{KL}(P \| Q) = \sup_{T} (\mathbb{E}_P[T] - \log \mathbb{E}_Q[e^T])$
*Hint*: Use the Gibbs' inequality and the fact that the optimal $T$ is $\log(P/Q) + c$.

### 1.5 Data Processing Inequality (DPI)
Let $X \to Y \to Z$ be a Markov chain. Prove that $I(X; Z) \le I(X; Y)$.
Relate this to the "Information Bottleneck" in a 3-layer neural network.

### 1.6 Entropy of a Sparse Vector
Consider a random vector $X \in \{0, 1\}^n$ where each bit is 1 with probability $p$.

1.  Calculate $h(X)$.
2.  If $p \to 0$ as $n \to \infty$ such that $np = \lambda$, what is the limiting entropy?

---

## 2. Coding Practice

### Task 2.1: MI Estimation for a Gaussian Mixture

1.  Generate a 2D dataset from a Mixture of Gaussians.
2.  Implement a simple "Neural MI Estimator" (MINE-style) using PyTorch.
3.  Compare the estimated MI to the ground truth (calculated numerically or analytically if possible).

### Task 2.2: Entropy and Perplexity of a Language Model

1.  Load a small pre-trained model (e.g., GPT-2).
2.  Compute the cross-entropy loss on a paragraph of text.
3.  Calculate the **Perplexity** ($e^{loss}$).
4.  How does Perplexity change if you "shuffle" the words in the paragraph? Why?

### Task 2.3: Visualizing the Information Plane
Implement a 3-layer MLP on MNIST.

1.  Bin the hidden activations of the middle layer to estimate $I(X; T)$ and $I(T; Y)$.
2.  Plot these two values as training progresses.
3.  Do you observe the "compression phase"?

---

## 3. Hints and Solutions

### Hint 1.2
$D_{KL} = \frac{1}{2} [\text{Tr}(\Sigma_1^{-1} \Sigma_0) + (\mu_1 - \mu_0)^\top \Sigma_1^{-1} (\mu_1 - \mu_0) - d + \ln \frac{|\Sigma_1|}{|\Sigma_0|}]$.

### Solution 1.3.2
Example: $X \sim \mathcal{N}(0, 1)$, $Y = X^2$. They are uncorrelated ($\mathbb{E}[XY] = \mathbb{E}[X^3] = 0$) but highly dependent. $I(X; Y)$ will be large.

### Hint 2.2
Shuffling words increases the entropy (specifically the conditional entropy $h(x_t | x_{t-1}, \dots)$) because the sequence becomes less predictable. Perplexity will spike.

### Solution 2.3
In ReLU networks, explicit compression (lowering $I(X; T)$) is often not observed because the mapping remains injective. Compression usually only appears if you use saturating activations (Tanh/Sigmoid) or explicit noise.

