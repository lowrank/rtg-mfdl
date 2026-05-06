# Chapter 2.4: Kolmogorov-Arnold Networks and Symbolic Approximation Theory

## 1. Introduction: Beyond the MLP Paradigm

For decades, the Multilayer Perceptron (MLP) has been the dominant architecture in deep learning. MLPs rely on the Universal Approximation Theorem, using fixed non-linear activations (like ReLU) on nodes and learnable weights on edges. However, in 1957, a profound mathematical theorem was proven that suggests a completely different way to build neural networks.

The **Kolmogorov-Arnold Representation Theorem** states that any multivariate continuous function can be represented as a finite composition of univariate functions and addition. In 2024, this theoretical result was translated into a practical architecture: the **Kolmogorov-Arnold Network (KAN)**. This chapter explores the theory, architecture, and symbolic discovery capabilities of KANs.

---

## 2. The Kolmogorov-Arnold Representation Theorem

### 2.1 Historical Context
Hilbert's 13th problem conjectured that functions of three variables could not be expressed as superpositions of functions of two variables. Vladimir Arnold (for $n=3$) and Andrey Kolmogorov (for general $n$) proved him wrong.

### 2.2 Theorem Statement


!!! success "Theorem 2.2"
    1 (Kolmogorov-Arnold, 1957)

    For any continuous function $f: [0, 1]^n \to \mathbb{R}$, there exist $2n+1$ continuous functions $\Phi_q: \mathbb{R} \to \mathbb{R}$ and $n(2n+1)$ continuous univariate functions $\psi_{q,p}: [0, 1] \to \mathbb{R}$ such that:



    $$
    f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^n \psi_{q,p}(x_p) \right)
    $$


    **Key Implications:**
    1. Multivariate complexity is reducible to univariate complexity.
    2. The "inner" functions $\psi_{q,p}$ are universal and independent of $f$.
    3. The "outer" functions $\Phi_q$ depend on $f$.

    ---

## 3. Kolmogorov-Arnold Networks (KANs)

While the theorem guarantees an exact representation, the $\psi$ functions are often non-smooth or even fractal, making them hard to learn. KANs resolve this by parameterizing the functions using **Splines**.

### 3.1 Architecture: Functions on Edges
Unlike MLPs, where activations are on nodes, KANs place **learnable non-linear functions** on the edges.

- **MLP Node:** $y = \sigma(\sum w_i x_i + b)$
- **KAN Edge:** $y = \sum \phi_i(x_i)$

Each $\phi_i$ is typically a B-spline:


$$
\phi(x) = \sum_{j} c_j B_j(x)
$$


where $B_j$ are basis functions and $c_j$ are learnable control points.

### 3.2 Advantages of KANs

1. **Interpretability:** Each edge function can be visualized and compared to symbolic formulas.
2. **Accuracy:** Splines allow for very high precision through "Grid Extension."
3. **No fixed activation:** The network learns the best activation function for each feature.

---

## 4. B-Splines and Grid Extension

To make KANs work, we need a stable way to parameterize univariate functions.

### 4.1 Cox-de Boor Recursion
B-splines of degree $k$ are defined over a knot vector $t_i$:


$$
B_{i,0}(x) = \mathbb{I}_{[t_i, t_{i+1})}(x)
$$


$$
B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x)
$$


### 4.2 Grid Extension Theorem

!!! success "Theorem 4.2"
    1

    A KAN can be refined post-training by increasing the number of grid points $G$. The error decays as $O(G^{-(k+1)})$.

*Proof:* This is a standard property of spline approximation. As we refine the grid, the spline converges to the underlying smooth function with a rate determined by the spline degree $k$. $\blacksquare$

---

## 5. Symbolic Discovery: Snap-to-Symbol

KANs are uniquely suited for **Symbolic Regression**.

**Algorithm 5.1 (Symbolic Discovery):**
1. **Train:** Fit the KAN using splines.
2. **Visualize:** Plot the learned edge functions $\phi(x)$.
3. **Hypothesize:** Compare $\phi(x)$ to a library $\{\sin, \exp, \ln, x^2, \dots\}$.
4. **Snap:** If a symbolic function matches with high correlation, replace the spline with that function.
5. **Optimize:** Fine-tune the coefficients of the symbolic formula.

---

## 6. Worked Examples

### Example 6.1: Multiplying two numbers
$f(x, y) = x \cdot y$.
In an MLP, this requires many neurons. In a KAN, we use:
$xy = \exp(\ln(x) + \ln(y))$ or $xy = \frac{1}{4}((x+y)^2 - (x-y)^2)$.
A 2-layer KAN can learn the $\ln$ or $(\cdot)^2$ functions on its edges and represent multiplication exactly.

### Example 6.2: Grid Refinement
A KAN trained with $G=10$ grid points has an MSE of $10^{-4}$. If we extend the grid to $G=100$, and the functions are smooth ($k=3$), the error should theoretically drop to $10^{-4} \cdot (10/100)^4 = 10^{-8}$.

### Example 6.3: Symbolic Logic of XOR
For $x_1, x_2 \in \{0, 1\}$, $x_1 \oplus x_2$ can be represented as $\sin^2(\frac{\pi}{2}(x_1 + x_2))$. A KAN can learn the $\sin^2$ shape on its output edge.

---

## 7. Code Demonstrations

### Demo 7.1: Visualizing a Learnable Spline Edge

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Simplified B-spline implementation
def b_spline(x, c, G=10):
# This is a placeholder for the actual recursive logic
return torch.sin(x) * c.mean() 

class KANLayer(nn.Module):
def __init__(self, in_dim, out_dim):
    super().__init__()
    self.phi = nn.Parameter(torch.randn(in_dim, out_dim, 20)) # 20 control points
    
def forward(self, x):
    # Apply splines to each edge
    return torch.stack([b_spline(x[:, i], self.phi[i]) for i in range(x.shape[1])]).sum(0)

# Instantiate and visualize
# ...
```

### Demo 7.2: Symbolic Regression with pykan
(Note: Requires `pykan` library)
```python
# from kan import KAN
# model = KAN(width=[2, 5, 1], grid=5, k=3)
# model.train(dataset)
# model.plot() # Visualizes the learned functions
# model.suggest_symbolic() # Outputs discovered formulas
```

---

## 8. Conclusion
KANs offer a mathematically elegant and highly interpretable alternative to MLPs. By leveraging the Kolmogorov-Arnold theorem and the power of splines, they allow for "surgical" precision and the direct discovery of physical laws from data.
