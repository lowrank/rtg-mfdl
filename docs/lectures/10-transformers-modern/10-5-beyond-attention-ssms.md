# 10.5 Beyond Attention: State Space Models (SSMs)

While Transformers have revolutionized AI, their $O(L^2)$ complexity with respect to sequence length $L$ remains a fundamental bottleneck for long-context applications. Structured State Space Models (SSMs), such as S4 and Mamba, offer a promising alternative by combining the parallelizability of Transformers with the $O(L)$ inference efficiency of Recurrent Neural Networks (RNNs). In this section, we derive the mathematical foundations of optimal history compression, parallelized sequence processing, and the recent advancement of selective SSMs.

## 1. Foundations of State Space Models

A State Space Model maps a 1D input signal $x(t) \in \mathbb{R}$ to an output signal $y(t) \in \mathbb{R}$ through an $N$-dimensional latent state $h(t) \in \mathbb{R}^N$:

$$
h'(t) = A h(t) + B x(t)
$$

$$
y(t) = C h(t) + D x(t)
$$

To process discrete sequences $x_0, x_1, \dots$, we must discretize the system with a step size $\Delta$. Using the Bilinear (Tustin) transform or Zero-Order Hold (ZOH):

$$
h_k = \bar{A} h_{k-1} + \bar{B} x_k
$$

$$
y_k = C h_k
$$

where $\bar{A} = (I - \frac{\Delta}{2} A)^{-1} (I + \frac{\Delta}{2} A)$ and $\bar{B} = (I - \frac{\Delta}{2} A)^{-1} \Delta B$.

## 2. Theorem: Optimal History Compression via HiPPO

For an SSM to remember long-range dependencies, the matrix $A$ must be carefully designed. The **HiPPO** (High-order Polynomial Projection Operators) framework provides a mathematically optimal way to compress the history of a signal into a fixed-dimensional state.

!!! success "Theorem 2.1 (The HiPPO-Legendre Matrix)"
    Let the state $h(t)$ represent the coefficients of the best $N$-th order polynomial approximation of the history $x(s)$ for $s \leq t$, weighted by a uniform measure. The optimal matrix $A \in \mathbb{R}^{N \times N}$ that evolves these coefficients is given by:
    
    $$
    A_{ij} = \begin{cases} -(2i+1)^{1/2}(2j+1)^{1/2} & \text{if } i > j \\ -(i+1) & \text{if } i = j \\ 0 & \text{if } i < j \end{cases}
    $$

**Proof:**

1.  **Polynomial Projection:**
    Define the $n$-th Legendre polynomial $P_n(x)$ on $[-1, 1]$. We shift it to $[0, t]$ using $P_n(s/t)$.
    The projection of $x(s)$ onto $P_n$ is $h_n(t) = \int_0^t x(s) P_n(s/t) ds$ (ignoring normalization for a moment).

2.  **Differentiating the State:**
    Using the Leibniz Integral Rule:
    
    $$
    \frac{d}{dt} h_n(t) = x(t) P_n(1) + \int_0^t x(s) \frac{\partial}{\partial t} P_n(s/t) ds
    $$

    Since $P_n(1) = 1$, the first term is $x(t)$.
    The second term involves the derivative of the shifted polynomial. Using the property $x P'_n(x) = n P_n(x) + \sum_{k < n} (2k+1) P_k(x)$ (for odd/even cases), we can express the integral back in terms of the coefficients $h_k(t)$.

3.  **The Resulting System:**
    The coefficients evolve according to a linear system $h'(t) = A h(t) + B x(t)$. For the Legendre case, this specific matrix $A$ ensures that the $L^2$ error of the polynomial approximation is minimized at every time $t$. $\blacksquare$

## 3. Theorem: Parallelization via Associative Scans

The main drawback of RNNs is their sequential nature. However, a linear recurrence $h_k = A_k h_{k-1} + B_k x_k$ can be computed in parallel if the operation is associative.

!!! success "Theorem 3.1 (Parallel Scan Efficiency)"
    The sequence of states $h_1, \dots, h_L$ can be computed in $O(\log L)$ time on $O(L)$ processors using an associative scan.
**Proof:**

1.  **Defining the Operator:**
    Define a tuple $\mathcal{T}_k = (A_k, b_k)$ where $b_k = B_k x_k$. The state update is $h_k = A_k h_{k-1} + b_k$.
    Define a binary operator $\otimes$:
    
    $$
    (A_j, b_j) \otimes (A_i, b_i) = (A_j A_i, A_j b_i + b_j)
    $$

2.  **Checking Associativity:**
    
    $$
    [\mathcal{T}_k \otimes \mathcal{T}_j] \otimes \mathcal{T}_i = (A_k A_j, A_k b_j + b_k) \otimes (A_i, b_i) = (A_k A_j A_i, A_k A_j b_i + A_k b_j + b_k)
    $$

    $$
    \mathcal{T}_k \otimes [\mathcal{T}_j \otimes \mathcal{T}_i] = \mathcal{T}_k \otimes (A_j A_i, A_j b_i + b_j) = (A_k A_j A_i, A_k (A_j b_i + b_j) + b_k)
    $$

    The results are identical. The operator is associative.

3.  **Parallel Computation:**
    Because $\otimes$ is associative, we can compute the prefix products $(\mathcal{A}_{1:k}, \mathcal{B}_{1:k})$ using a prefix sum algorithm (Blelloch scan) in $O(\log L)$ steps. $\blacksquare$

## 4. Selection and the Mamba Architecture

A critical limitation of S4 was its **Time-Invariance**: the matrices $A, B, C$ were constant for all tokens. This meant the model could not "filter" the input based on content (e.g., ignoring a distractor).

**Selective SSMs (Mamba)** make $B, C,$ and $\Delta$ functions of the input $x_t$:

$$
B_t = \text{Linear}_B(x_t), \quad C_t = \text{Linear}_C(x_t), \quad \Delta_t = \text{Softplus}(\text{Linear}_\Delta(x_t))
$$

!!! success "Theorem 4.1 (Information Bottleneck in Selective SSMs)"
    By allowing $\Delta_t$ to vary, the model can effectively implement a "Gating" mechanism. If $\Delta_t \to 0$, the state $h_t \approx h_{t-1}$ (the model ignores the current input). If $\Delta_t \to \infty$, the state is reset. This allows the model to compress sequences more effectively than fixed-S4 by only updating the state when important information arrives.
## 5. Worked Examples

### Worked Example 1: Discretization of a Simple SSM

Given $A = -1, B = 1, \Delta = 0.1$. Compute $\bar{A}$ and $\bar{B}$ using ZOH.
$\bar{A} = \exp(A \Delta) = \exp(-0.1) \approx 0.9048$.
$\bar{B} = A^{-1}(\bar{A} - I) B = (-1)^{-1}(0.9048 - 1) \cdot 1 = 0.0952$.
The discrete update is $h_k = 0.9048 h_{k-1} + 0.0952 x_k$.

### Worked Example 2: Prefix Sum as Associative Scan

To compute the prefix sum of $x = [1, 2, 3, 4]$, let $A_k = 1, b_k = x_k$.
$\mathcal{T}_1 = (1, 1), \mathcal{T}_2 = (1, 2), \dots$
$\mathcal{T}_2 \otimes \mathcal{T}_1 = (1 \cdot 1, 1 \cdot 1 + 2) = (1, 3)$.
The second component is the sum $1+2=3$.

### Worked Example 3: HiPPO Matrix for $N=2$

$$
A = \begin{bmatrix} -(0+1) & 0 \\ -(3)^{1/2}(1)^{1/2} & -(1+1) \end{bmatrix} = \begin{bmatrix} -1 & 0 \\ -1.732 & -2 \end{bmatrix}
$$

### Worked Example 4: Gated RNN as an SSM

A Gated Recurrent Unit (GRU) update $h_t = (1-z_t) h_{t-1} + z_t \tilde{h}_t$ can be viewed as an SSM where $\bar{A}_t = 1-z_t$ and $\bar{B}_t = z_t$. Mamba generalizes this by allowing the state $h$ to be high-dimensional and using the HiPPO matrix for better memory.

## 6. Coding Demonstrations

### Coding Demo 1: A Selective SSM (Mamba) Logic

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_state = d_state
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1) # B, C, delta
        self.A = nn.Parameter(torch.randn(d_model, d_state))

    def forward(self, x):
        # x: [batch, L, d_model]
        batch, L, d = x.shape
        
        # Selection
        proj = self.x_proj(x)
        B, C, delta = torch.split(proj, [self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(delta)
        
        # Discretize (Simplified)
        A_bar = torch.exp(self.A.unsqueeze(1) * delta) # [batch, L, d, d_state]
        B_bar = B.unsqueeze(2) * delta.unsqueeze(2)
        
        # In a real Mamba, we use a parallel scan or a hardware-aware kernel
        h = torch.zeros(batch, d, self.d_state)
        output = []
        for i in range(L):
            h = A_bar[:, i] * h + B_bar[:, i] * x[:, i].unsqueeze(-1)
            y = torch.sum(h * C[:, i].unsqueeze(1), dim=-1)
            output.append(y)
            
        return torch.stack(output, dim=1)
```

### Coding Demo 2: Associative Scan in Python

```python
import numpy as np

def associative_scan(A, b):
    # A: list of matrices, b: list of vectors
    # Simple binary tree implementation
    L = len(A)
    if L == 1:
        return A, b
    
    # Pairwise combine
    A_next, b_next = [], []
    for i in range(0, L, 2):
        if i+1 < L:
            # (A2, b2) @ (A1, b1) = (A2 A1, A2 b1 + b2)
            A_next.append(A[i+1] @ A[i])
            b_next.append(A[i+1] @ b[i] + b[i+1])
        else:
            A_next.append(A[i])
            b_next.append(b[i])
            
    # Recursive call and then expand (simplified)
    # ... logic for full prefix sum ...
    pass
```

By moving beyond the quadratic attention bottleneck, SSMs provide a path toward truly infinite-context models that can process entire books or video streams as a single, continuous signal.
