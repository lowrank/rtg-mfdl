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
        A_bar = torch.exp(self.A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1)) # [batch, L, d_model, d_state]
        B_bar = B.unsqueeze(2) * delta.unsqueeze(-1)  # [batch, L, 1, d_state] -> broadcast
        
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

def associative_scan(A_list, b_list):
    """
    Compute all prefix states h_k for k=1..L using an associative scan.
    Each state satisfies: h_k = A_k @ h_{k-1} + b_k, with h_0 = 0.

    We represent computation via tuples T_k = (A_k, b_k) under the
    associative binary operator:
        (A2, b2) ⊗ (A1, b1) = (A2 @ A1, A2 @ b1 + b2)

    The k-th prefix product T_{k:1} = T_k ⊗ ... ⊗ T_1 satisfies
        h_k = b_{k:1}   (the second component, since h_0 = 0).

    This implementation uses a parallel binary-tree (up-sweep / down-sweep)
    reduction known as the Blelloch scan to compute all prefix products in
    O(log L) parallel steps.

    Returns: list of h_k (numpy arrays) for k = 1 .. L.
    """
    L = len(A_list)
    # Work arrays — store tuples (A, b) for each leaf / internal node.
    # We pad to the next power of two for a clean binary tree.
    size = 1
    while size < L:
        size *= 2

    # Pad with identity elements  (A=I, b=0)
    n = A_list[0].shape[0]
    I_n = np.eye(n)
    zero_n = np.zeros(n)
    A_tree = [I_n.copy() for _ in range(size)]
    b_tree = [zero_n.copy() for _ in range(size)]
    for k in range(L):
        A_tree[k] = A_list[k].copy()
        b_tree[k] = b_list[k].copy()

    # ---- Up-sweep (reduce) phase ----
    # Build a complete binary tree of partial products.
    stride = 1
    while stride < size:
        for i in range(stride - 1, size, 2 * stride):
            right = i + stride
            if right < size:
                # parent = right ⊗ left
                new_b = A_tree[right] @ b_tree[i] + b_tree[right]
                new_A = A_tree[right] @ A_tree[i]
                A_tree[right] = new_A
                b_tree[right] = new_b
        stride *= 2

    # ---- Down-sweep phase ----
    # Set the root to the identity (exclusive-scan initialisation).
    A_tree[size - 1] = I_n.copy()
    b_tree[size - 1] = zero_n.copy()

    stride = size // 2
    while stride >= 1:
        for i in range(stride - 1, size, 2 * stride):
            right = i + stride
            if right < size:
                # Save left child
                tmp_A = A_tree[i].copy()
                tmp_b = b_tree[i].copy()
                # Left child gets parent's value
                A_tree[i] = A_tree[right].copy()
                b_tree[i] = b_tree[right].copy()
                # Right child = original_left ⊗ new_parent
                A_tree[right] = tmp_A @ A_tree[i]
                b_tree[right] = tmp_A @ b_tree[i] + tmp_b
        stride //= 2

    # After down-sweep, leaf k holds the EXCLUSIVE prefix product T_{k-1:1}.
    # The INCLUSIVE prefix state h_k = A_k @ b_tree[k] + b_list[k]
    prefix_states = []
    for k in range(L):
        h_k = A_list[k] @ b_tree[k] + b_list[k]
        prefix_states.append(h_k)

    return prefix_states


# Simple binary tree implementation
# A: list of matrices, b: list of vectors
# Returns the list of prefix states h_1, ..., h_L
np.random.seed(42)
state_dim = 3
L = 8

A_mats = [np.random.randn(state_dim, state_dim) * 0.3 for _ in range(L)]
b_vecs = [np.random.randn(state_dim) for _ in range(L)]

# Sequential reference
h = np.zeros(state_dim)
ref_states = []
for k in range(L):
    h = A_mats[k] @ h + b_vecs[k]
    ref_states.append(h.copy())

# Parallel scan
scan_states = associative_scan(A_mats, b_vecs)

print("Sequential vs Associative Scan comparison:")
for k in range(L):
    err = np.max(np.abs(ref_states[k] - scan_states[k]))
    print(f"  h_{k+1}: max_err = {err:.2e}")

all_close = all(np.allclose(ref_states[k], scan_states[k], atol=1e-10) for k in range(L))
print(f"\nAll states match: {all_close}")
```

```
Sequential vs Associative Scan comparison:
  h_1: max_err = 0.00e+00
  h_2: max_err = 0.00e+00
  h_3: max_err = 0.00e+00
  h_4: max_err = 0.00e+00
  h_5: max_err = 1.11e-16
  h_6: max_err = 0.00e+00
  h_7: max_err = 2.78e-17
  h_8: max_err = 0.00e+00

All states match: True
```

By moving beyond the quadratic attention bottleneck, SSMs provide a path toward truly infinite-context models that can process entire books or video streams as a single, continuous signal.

