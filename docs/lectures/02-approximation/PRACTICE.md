# Topic 02: Approximation Theory — Practice and Problem Set

This problem set is designed to test your mastery of the functional analytic, harmonic, and compositional foundations of neural network approximation.

---

## 1. Mathematical Proofs and Derivations

### Exercise 1.1: The Hahn-Banach Separation
Suppose a subspace $S$ of $C(K)$ is not dense.

1. Formally state the Hahn-Banach theorem corollary used in Cybenko's proof.
2. If there exists a target function $f$ such that $\operatorname{dist}(f, S) = \delta > 0$, construct a linear functional $L$ such that $L(S) = 0$ and $L(f) = 1$. What is the norm $\|L\|$?

### Exercise 1.2: Polynomial Approximation Limits
Prove that a neural network with a single hidden layer and activation $\sigma(x) = x^k$ (for a fixed $k \in \mathbb{N}$) cannot approximate $f(x) = \sin(x)$ on $[0, 2\pi]$ with zero error. 
*Hint:* Use the fact that polynomials of degree $k$ form a finite-dimensional subspace.

### Exercise 1.3: Barron Norm Calculation
Calculate the Barron norm $C_f$ for the function $f(x) = \cos(k x)$ on $\mathbb{R}$.

1. Find the Fourier transform $\hat{f}(\omega)$.
2. Evaluate the integral $\int |\omega| |\hat{f}(\omega)| d\omega$.
3. What happens as $k \to \infty$? What does this imply about the difficulty of learning high-frequency signals?

### Exercise 1.4: Piecewise Linear Counting
A deep ReLU network has 3 layers with widths $m_1=2, m_2=2, m_3=1$ (all 1D).

1. What is the maximum number of linear segments this network can produce?
2. Construct a set of weights and biases that achieves this maximum.

---

## 2. Conceptual Questions

### Question 2.1: Width vs. Depth
Explain in your own words why a deep network can approximate the Parity function (XOR) more efficiently than a shallow one. Use the concept of "recursive folding" or "compositional complexity."

### Question 2.2: Spectral Bias in Practice
You are training a model on noisy data $y = f(x) + \text{noise}$, where the noise is high-frequency.

1. Why does early stopping act as a regularizer in this context?
2. Based on the NTK theory, which part of the signal will be learned first?

### Question 2.3: KAN Interpretability
How does the "Functions on Edges" paradigm of KANs improve symbolic discovery compared to standard Symbolic Regression on an MLP's output?

---

## 3. Computational Exercises

### Task 3.1: The Sawtooth Explosion
Write a Python script to compute the number of linear segments in $f_L(x)$ for $L=1$ to $L=10$.

- Plot the result on a log-scale.
- Verify that it follows the $2^L$ law.

### Task 3.2: NTK Eigenvalue Decay
Using the analytical formula for the ReLU NTK $\kappa(t) = \frac{1}{\pi} (t (\pi - \arccos t) + \sqrt{1-t^2})$, plot the function on $t \in [-1, 1]$.

- Perform a numerical Fourier Transform (or Gegenbauer expansion) to estimate the decay rate of the eigenvalues.
- Check if it matches the theoretical $k^{-(d+1)}$ decay.

### Task 3.3: Pruning and Capacity

1. Create a random neural network with $10^6$ parameters.
2. Prune $99\%$ of the weights to zero.
3. Compare the "Random Sparse" network's performance on a simple sine wave task against a "Trained Dense" network. Discuss the Strong Lottery Ticket Hypothesis.

---

## 4. Solutions and Hints

- **Hint 1.3:** The Fourier transform of $\cos(kx)$ is a sum of two Dirac deltas at $\pm k$. The integral becomes $|k| + |-k| = 2|k|$. The Barron norm grows linearly with frequency.
- **Solution 1.4:** $\operatorname{cpw} \le (m_1+1)(m_2+1) \dots = 3 \cdot 3 = 9$. However, since the input is 1D, the bound is tighter. Composition yields $2^L$ for specific "sawtooth" weights.

