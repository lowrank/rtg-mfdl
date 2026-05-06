# Chapter 2.5: Practical Approximation Guidelines: Capacity, Sparsity, and Scaling Laws

## 1. Introduction: From Existence to Engineering

The Universal Approximation Theorem and Depth Separation Theorems provide the mathematical "permission" to build deep networks. However, they do not provide the "blueprints" for real-world engineering. In practice, we must decide:

- How many parameters $W$ are needed for a dataset of size $N$?
- How much can we prune a network before losing accuracy?
- How does the error scale as we increase compute and data?

This chapter translates abstract approximation theory into **Practical Guidelines**. We explore memorization capacity, sparse approximation (Lottery Ticket Hypothesis), and the empirical Neural Scaling Laws.

---

## 2. Memorization Capacity

Before a network can generalize, it must have the capacity to interpolate (memorize) the training data.

!!! success "Theorem 2"
    1 (The $W \ge N$ Rule)

    A neural network with $W$ parameters can memorize $N$ arbitrary data points if $W \ge N$.

**Rigorous Proof for 2-Layer ReLU Networks:**
1. Consider $N$ points $\{x_i\}$ and labels $\{y_i\}$.
2. A 2-layer network computes $f(x) = \sum_{j=1}^m a_j \sigma(w_j^T x + b_j)$.
3. This is a system of $N$ equations with $m(d+2)$ unknowns.
4. If the points are in "general position," the activation matrix $A_{ij} = \sigma(w_j^T x_i + b_j)$ is full-rank if $m \ge N/d$.
5. Thus, one can find a set of output weights $\{a_j\}$ such that $f(x_i) = y_i$ exactly. $\blacksquare$

*Practical Insight:* If your network has fewer parameters than data points, it is mathematically impossible to reach zero training error unless the data is highly structured.

---

## 3. Sparse Approximation and the Lottery Ticket Hypothesis

Deep networks are notoriously overparameterized. The **Lottery Ticket Hypothesis** (Frankle & Carbin, 2018) suggests that most of these parameters are unnecessary.

!!! success "Theorem 3"
    1 (Strong Lottery Ticket Theorem)

    For any sufficiently overparameterized network $N$, there exists a sparse sub-network $n \subset N$ that approximates any target function $f$ as well as $N$ can, without any weight training (only by pruning).

**Proof Insight:**
1. If we have $10^9$ random weights, the probability that some subset of them, when combined, approximates the optimal weights for a smaller network is very high.
2. Pruning acts as a "search" over the space of random sub-architectures. $\blacksquare$

*Practical Insight:* Pruning by $90-99\%$ is often possible because the dense network acts as a high-dimensional "cover" for the optimal sparse solution.

---

## 4. Neural Scaling Laws

Empirical evidence from OpenAI, DeepMind, and others shows that the test loss $L$ follows a predictable power law.

!!! info "Definition 4"
    1 (The Scaling Law)

    $$
    L(N, D) \approx \left( \frac{N_c}{N} \right)^{\alpha_N} + \left( \frac{D_c}{D} \right)^{\alpha_D}
    $$

    where $N$ is parameter count, $D$ is dataset size, and $\alpha$ are scaling exponents.

!!! success "Theorem 4"
    2 (Fundamental Scaling Bound)

    For functions in $d$ dimensions with $s$ smoothness, the optimal approximation scaling is $\alpha = s/d$.

*Proof:* This follows from the metric entropy of the function space. To reduce the error by half, we must cover the space with $2^{d/s}$ more "balls" (parameters). $\blacksquare$

---

## 5. Worked Examples

### Example 5.1: Capacity Planning for a Startup
A startup has $1,000,000$ high-resolution medical images. 

- **Minimum Model Size:** To memorize the data, they need $\sim 10^6$ parameters.
- **Overparameterization:** Standard practice suggests a $10\times$ to $100\times$ buffer for generalization.
- **Recommendation:** A model with $10^7$ to $10^8$ parameters (e.g., a standard ResNet or small Vision Transformer).

### Example 5.2: Pruning Efficiency
A model has 100 million parameters. 

- According to the Lottery Ticket Theorem, if we prune $95\%$ (leaving 5 million), we might still achieve baseline accuracy.
- **Hardware Benefit:** This reduces the memory footprint and increases inference speed by up to $20\times$.

### Example 5.3: Scaling Law Prediction
If increasing a model from 1B to 10B parameters reduced the error from $2.0$ to $1.5$:
$1.5 / 2.0 = (10/1)^{- \alpha} \implies 0.75 = 10^{- \alpha} \implies \alpha \approx 0.12$.
To get the error down to $1.0$:
$1.0 / 1.5 = (N/10)^{-0.12} \implies 0.66 = (N/10)^{-0.12} \implies N \approx 10 \cdot (0.66)^{-1/0.12} \approx 10 \cdot 30 = 300B$.
Scaling laws allow us to predict that we need a $300B$ model before we even train it!

---

## 6. Code Demonstrations

### Demo 6.1: Memorization Capacity Test

```python
import torch
import torch.nn as nn
import torch.optim as optim

def test_capacity(N, d, W):
x = torch.randn(N, d)
y = torch.randn(N, 1)
    
model = nn.Sequential(nn.Linear(d, W//2), nn.ReLU(), nn.Linear(W//2, 1))
optimizer = optim.Adam(model.parameters(), lr=0.01)
    
for _ in range(1000):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()
    if loss < 1e-4: return True
return False

# N=100 points, d=10. W=200 should succeed. W=50 should fail.
```

### Demo 6.2: Scaling Law Simulator

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_loss(N, alpha=0.1):
return 10.0 / (N**alpha)

N_range = np.logspace(1, 9, 20)
losses = compute_loss(N_range)

plt.loglog(N_range, losses, 'o-')
plt.title("Neural Scaling Law (L vs N)")
plt.xlabel("Number of Parameters (N)")
plt.ylabel("Test Loss (L)")
plt.grid(True)
plt.show()
```

---

## 7. Conclusion
Practical approximation is about managing resources. By understanding the memorization limits, the redundancy of parameters (sparsity), and the predictable yields of scaling, we can move from "trial and error" to "principled engineering" in deep learning.
