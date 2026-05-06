# Topic 10: Practice and Exercises - Transformer Theory

This module focuses on the mathematical derivation of Transformer properties and their practical implementation.

---

## 1. Theoretical Exercises

### Exercise 1.1: Permutation Equivariance
Prove that the Self-Attention mechanism (without positional embeddings) is **permutation equivariant**. That is, if $\sigma$ is a permutation of $\{1, \dots, n\}$ and $P_\sigma$ is the corresponding permutation matrix, show that:

$$
\text{Attn}(P_\sigma X) = P_\sigma \text{Attn}(X)
$$

**Hint**: Show that the softmax weights $A_{ij}$ simply become $A_{\sigma(i)\sigma(j)}$.

### Exercise 1.2: The Softmax Variance
Let $q, k \in \mathbb{R}^d$ be independent vectors with entries $q_i, k_i \sim \mathcal{N}(0, 1)$.

1.  Show that $\mathbb{E}[q \cdot k] = 0$.
2.  Show that $\text{Var}(q \cdot k) = d$.
3.  Explain why scaling by $1/\sqrt{d}$ is necessary to keep the input to the softmax at unit variance.

### Exercise 1.3: Rank of Attention
Prove that for a single-head attention layer with queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{n \times d_k}$, and values $V \in \mathbb{R}^{n \times d_v}$, the rank of the output $Z$ is at most $\min(n, d_k, d_v)$. How does this change if we have $H$ heads?

### Exercise 1.4: Linear Attention as Gradient Descent
Consider the Linear Attention mechanism $O = (QK^\top)V$. Assume $Q = x_{\text{test}}^\top, K = X, V = Y$.

1.  Derive the condition on the learning rate $\eta$ such that $O$ exactly matches one step of GD on $L(W) = \frac{1}{2}\|XW - Y\|^2$.
2.  What happens if we use a non-linear softmax attention? Can it be viewed as a different type of optimizer?

### Exercise 1.5: The Chinchilla Frontier
A researcher has a compute budget of $C = 1.2 \times 10^{24}$ FLOPs. Using the Chinchilla estimate that $C \approx 6ND$ and that for optimal training $D \approx 20N$:

1.  Calculate the optimal number of parameters $N$.
2.  Calculate the required number of training tokens $D$.

### Exercise 1.6: Positional Encoding Geometry
In the sinusoidal encoding used in the original Transformer:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Show that for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$. What does this imply about the model's ability to attend to relative positions?

### Exercise 1.7: Rank Collapse Bound
Given the convergence theorem $\|H^{(\ell)} - \mathbf{1}v^\top\| \le C e^{-\lambda \ell}$, if we want to maintain at least $10\%$ of the initial variance after 12 layers, what is the maximum allowable value for the convergence rate $\lambda$?

### Exercise 1.8: Induction Head Circuit
An induction head is formed by the composition of two heads. Head 1 (Layer 1) is a "Previous Token Head". Head 2 (Layer 2) is the "Induction Head".

1.  Write the mathematical expression for the Query of Head 2 if it attends to the output of Head 1.
2.  Explain why this circuit fails if the sequence length is 1.

### Exercise 1.9: Johnson-Lindenstrauss in RAG
In a RAG system, we project $10^{12}$ documents into a $d$-dimensional embedding space. According to the JL Lemma, we want to preserve distances within $\epsilon = 0.1$. What is the minimum $d$ required to ensure this with high probability? (Use the formula $d \ge 8 \ln(m) / \epsilon^2$).

### Exercise 1.10: Grokking and Weight Norm
During grokking, the training loss is near zero but the validation loss is high until a sudden drop. It is observed that the $L_2$ norm of the weights increases during the "memorization" phase and decreases during the "generalization" phase. Why would a lower weight norm correspond to better generalization in modular arithmetic?

---

## 2. Coding Practice

### Task 2.1: Manual Attention
Implement a single-head scaled dot-product attention function using only `numpy`.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
# Your code here
pass

```

**Validation**: Verify that your function is permutation equivariant (except for numerical precision).

### Task 2.2: Visualizing Induction
Using `TransformerLens` (or a simulated attention matrix), write a script to detect "induction heads" in a toy sequence `[1, 2, 3, 1, 2, ?]`.
**Goal**: Identify the head that places maximum attention on the token `3` when the current token is `2` and it has seen `1, 2` before.

### Task 2.3: Rank Analysis
Create a 50-layer "pure" attention model (no residuals, no MLPs) with random weights. Pass a random input through it and plot the singular values of the activations at layer 1, 10, 25, and 50.
**Goal**: Observe the "Rank Collapse" phenomenon where all but one singular value vanish.

---

## 3. Solutions and Hints

### Solution 1.2 (Softmax Variance)

2. $\text{Var}(q \cdot k) = \sum \text{Var}(q_i k_i)$. Since $q_i, k_i$ are independent $\mathcal{N}(0, 1)$, $q_i k_i$ has mean 0 and variance $\mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 \cdot 1 = 1$. Thus $\text{Var}(q \cdot k) = d$.

### Solution 1.5 (Chinchilla)

1. $6N(20N) = 1.2 \times 10^{24} \implies 120N^2 = 1.2 \times 10^{24} \implies N^2 = 10^{22} \implies N = 10^{11}$ (100 Billion parameters).
2. $D = 20N = 2 \times 10^{12}$ (2 Trillion tokens).

### Hint 2.3
Use `torch.linalg.svdvals` to compute singular values efficiently. You should see the distribution of singular values becoming increasingly "peaky" at the first value as depth increases.

