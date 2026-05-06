# 7.3 Physics-Informed Neural Networks and Neural Operators

## 1. Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) integrate physical laws, typically expressed as Partial Differential Equations (PDEs), directly into the loss function of a neural network. Instead of relying solely on massive amounts of data, a PINN acts as a mesh-free PDE solver that "learns" the solution by penalizing deviations from the governing equations, boundary conditions, and initial conditions.

Consider a domain $\Omega \subset \mathbb{R}^d$ and a general non-linear PDE parameterized by $\lambda$:

$$
\mathcal{F}_\lambda[u](x) = 0, \quad x \in \Omega
$$

with boundary conditions $\mathcal{B}[u](x) = 0$ on $\partial \Omega$. A PINN uses a neural network $u_\theta(x)$ to approximate the solution $u(x)$. The loss function is composed of two primary terms:

$$
\mathcal{L}(\theta) = \mathcal{L}_{data} + \mathcal{L}_{PDE}
$$

$$
\mathcal{L}_{PDE} = \frac{1}{N_{PDE}} \sum_{i=1}^{N_{PDE}} \left\| \mathcal{F}_\lambda[u_\theta](x_i) \right\|^2 + \frac{1}{N_{BC}} \sum_{j=1}^{N_{BC}} \left\| \mathcal{B}[u_\theta](x_j) \right\|^2
$$

The derivatives required for $\mathcal{F}_\lambda$ (e.g., spatial gradients, time derivatives) are computed exactly using Automatic Differentiation (AD).

---

## 2. Spectral Bias and the Neural Tangent Kernel (NTK)

A well-known challenge in training PINNs is their difficulty in learning high-frequency components of the PDE solution. This phenomenon is mathematically rigorously explained by the **Spectral Bias** theorem via the Neural Tangent Kernel (NTK).

### 2.1 Theorem Statement

!!! success "Theorem (Spectral Bias of Neural Networks)"
    In the infinite-width limit, the training dynamics of a neural network under Gradient Descent follow a linear differential equation governed by the integral operator induced by the Neural Tangent Kernel $\Theta(x, x')$. If the kernel is positive-definite and isotropic, the network learns the eigenfunctions of $\Theta$ at a rate proportional to their corresponding eigenvalues. Since the eigenvalues of standard networks (like ReLU MLPs) decay polynomially with frequency, high-frequency target components converge exponentially slower than low-frequency components.

### 2.2 Rigorous Proof

**Step 1: NTK Formulation**

Let $u_\theta(x)$ be the network output. The continuous-time gradient descent dynamics for the parameters are:

$$
\frac{d\theta}{dt} = - \eta \nabla_\theta \mathcal{L}(\theta)
$$

For a simple L2 loss on a target function $u^*(x)$: $\mathcal{L} = \frac{1}{2} \int_\Omega (u_\theta(x) - u^*(x))^2 dx$.

$$
\frac{d\theta}{dt} = - \eta \int_\Omega (u_\theta(x) - u^*(x)) \nabla_\theta u_\theta(x) dx
$$

**Step 2: Function-Space Dynamics**

We analyze how the output $u_\theta(x)$ evolves:

$$
\frac{\partial u_\theta(x)}{\partial t} = \nabla_\theta u_\theta(x)^T \frac{d\theta}{dt}
$$

Substitute the parameter dynamics:

$$
\frac{\partial u_\theta(x)}{\partial t} = - \eta \int_\Omega \left( \nabla_\theta u_\theta(x)^T \nabla_\theta u_\theta(x') \right) (u_\theta(x') - u^*(x')) dx'
$$

Define the Neural Tangent Kernel: $\Theta(x, x') = \nabla_\theta u_\theta(x)^T \nabla_\theta u_\theta(x')$.
In the infinite-width limit, $\Theta(x, x')$ remains strictly constant during training: $\Theta(x, x') = \Theta_\infty(x, x')$.

Thus, the error residual $e(x, t) = u_\theta(x, t) - u^*(x)$ evolves as:

$$
\frac{\partial e(x, t)}{\partial t} = - \eta \int_\Omega \Theta_\infty(x, x') e(x', t) dx'
$$

**Step 3: Spectral Decomposition**

By Mercer's Theorem, since $\Theta_\infty$ is a continuous symmetric positive-definite kernel, it admits a spectral decomposition:

$$
\Theta_\infty(x, x') = \sum_{k=1}^\infty \lambda_k \phi_k(x) \phi_k(x')
$$

where $\lambda_k > 0$ are the eigenvalues in strictly decreasing order ($\lambda_1 \ge \lambda_2 \ge \dots$), and $\phi_k(x)$ are the orthogonal eigenfunctions.

**Step 4: Convergence Rates**

Projecting the error onto the eigenfunctions $c_k(t) = \int e(x, t) \phi_k(x) dx$, we obtain a set of decoupled ODEs:

$$
\frac{dc_k(t)}{dt} = - \eta \lambda_k c_k(t)
$$

The exact solution is:

$$
c_k(t) = c_k(0) e^{-\eta \lambda_k t}
$$

For shift-invariant kernels on a sphere, the eigenfunctions $\phi_k(x)$ are spherical harmonics (analogous to Fourier modes). High-frequency modes correspond to large $k$. It has been shown that for ReLU networks, the eigenvalues decay rapidly: $\lambda_k \sim k^{-d}$.

Therefore, the exponential decay rate $\eta \lambda_k$ for high-frequency components (large $k$) approaches zero. The network rapidly fits the low-frequency components but requires exponentially more time to fit the high-frequency components. $\blacksquare$

---

## 3. Universal Approximation for Operators

While PINNs learn a single solution $u(x)$, **Neural Operators** aim to learn a mapping between infinite-dimensional function spaces. For example, mapping an initial condition $u(x, 0)$ to a future state $u(x, T)$ across an entire family of initial conditions.

### 3.1 Theorem Statement

!!! success "Theorem (Universal Operator Approximation, Chen & Chen 1995)"
    Let $X$ be a compact metric space, $V$ a compact set in $C(X)$, and $G$ a continuous nonlinear operator mapping $V$ to $C(X)$. For any $\epsilon > 0$, there exists a single-hidden-layer neural operator defined by:

    $$
    \hat{G}[u](x) = \sum_{i=1}^N c_i \sigma \left( \sum_{j=1}^m w_{ij} u(x_j) + b_i \right)
    $$

    such that $\sup_{u \in V} \sup_{x \in X} |G[u](x) - \hat{G}[u](x)| < \epsilon$.

### 3.2 Proof Overview

The proof hinges on projecting the infinite-dimensional function $u \in V$ into a finite-dimensional representation, and then applying the standard Universal Approximation Theorem for vector functions.

1. **Compactness and Discretization:** Since $V$ is compact in $C(X)$, by the Arzelà-Ascoli theorem, the functions in $V$ are uniformly equicontinuous. This means we can find a finite set of sensor points $\{x_1, \dots, x_m\} \subset X$ such that the discrete evaluation vector $\vec{u} = (u(x_1), \dots, u(x_m))$ captures the necessary information to reconstruct $u(x)$ up to an $\epsilon/2$ error.
2. **Continuous Mapping:** Let $E: V \to \mathbb{R}^m$ be the evaluation operator $E(u) = \vec{u}$. Because of uniform equicontinuity, there exists a continuous map $F: \mathbb{R}^m \to C(X)$ such that $G(u) \approx F(E(u))$.
3. **Standard Universal Approximation:** The mapping $F$ acts on finite-dimensional vectors $\vec{u} \in \mathbb{R}^m$ to output a scalar value at any point $x \in X$. By the classic Cybenko universal approximation theorem, a standard neural network can approximate $F(\vec{u})(x)$ to arbitrary precision $\epsilon/2$.
4. **Conclusion:** Combining the discretization bound and the neural network approximation bound yields the full operator approximation error bound of $\epsilon$. $\blacksquare$

---

## 4. Fourier Neural Operator (FNO)

The Fourier Neural Operator (FNO) is a highly efficient architecture for learning operators. Instead of point-wise discrete evaluations, FNO formulates the operation as an integral operator, evaluated efficiently in the frequency domain using the Fast Fourier Transform (FFT).

### 4.1 The Convolution Formula

An iterative layer in the FNO updates a hidden representation $v_t(x)$ via a non-linear integral operator:

$$
v_{t+1}(x) = \sigma \left( W v_t(x) + \int_\Omega \kappa_\phi(x, y) v_t(y) dy \right)
$$

If we restrict the kernel to be stationary, i.e., $\kappa_\phi(x, y) = \kappa_\phi(x - y)$, the integral becomes a continuous convolution:

$$
v_{t+1}(x) = \sigma \left( W v_t(x) + (\kappa_\phi * v_t)(x) \right)
$$

By the Convolution Theorem, the Fourier transform of a convolution is the point-wise product of their Fourier transforms. Let $\mathcal{F}$ denote the Fourier transform:

$$
\mathcal{F}(\kappa_\phi * v_t)(\xi) = \mathcal{F}(\kappa_\phi)(\xi) \cdot \mathcal{F}(v_t)(\xi)
$$

In the FNO, we parameterize the Fourier transform of the kernel directly as a learnable weight tensor $R_\phi(\xi) \equiv \mathcal{F}(\kappa_\phi)(\xi)$. We also truncate the high-frequency modes (keeping only the lowest $K$ modes) to enforce smoothness and resolution invariance.

$$
(\kappa_\phi * v_t)(x) = \mathcal{F}^{-1} \left( R_\phi(\xi) \cdot \mathcal{F}(v_t)(\xi) \right)(x)
$$

Because this is formulated as a continuous integral evaluated globally via Fourier space, the model is inherently mesh-free and resolution-invariant!

---

## 5. Worked Examples

### 5.1 Example: 1D Burgers' Equation PINN Loss
Equation: $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0$.
Given a point $(x, t)$ and network $u_\theta(x, t)$.
1. We compute $u_t = \frac{\partial u_\theta}{\partial t}$ and $u_x = \frac{\partial u_\theta}{\partial x}$, $u_{xx} = \frac{\partial^2 u_\theta}{\partial x^2}$ using auto-diff.
2. The PDE residual is $f = u_t + u_\theta u_x - \nu u_{xx}$.
3. Loss is $L = \frac{1}{N} \sum f_i^2 + \lambda \sum (u_\theta(x_{ic}, 0) - u_{true})^2$.

### 5.2 Example: Advection Equation Operator Learning
Goal: Map $u(x, 0)$ to $u(x, T)$ for $u_t + c u_x = 0$.
The analytical operator is simply a shift: $G[u](x) = u(x - cT)$.
In Fourier space, a shift is a phase multiplication: $\mathcal{F}(u(x-cT))(\xi) = e^{-i 2\pi \xi c T} \mathcal{F}(u)(\xi)$.
An FNO can perfectly learn this exact operator by setting its spectral weights $R(\xi) = e^{-i 2\pi \xi c T}$. Thus, FNO completely captures advection linearly.

### 5.3 Example: Eigenvalues of NTK for 1D Domain
Consider a 1D domain $[0, 1]$ and an NTK $\Theta(x, x') = \cos(\pi(x - x'))$.
By trigonometric identities, $\cos(\pi(x - x')) = \cos(\pi x)\cos(\pi x') + \sin(\pi x)\sin(\pi x')$.
The eigenfunctions are exactly $\cos(\pi x)$ and $\sin(\pi x)$ with eigenvalue $\lambda = 1$. The network will learn these components at a uniform rate.

---

## 6. Coding Demonstrations

### 6.1 Basic PINN for a 1D ODE

We solve $\frac{du}{dx} + u = 0$, with $u(0) = 1$. The true solution is $u(x) = e^{-x}$.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# Setup
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    
    # 1. PDE Loss
    x_collocation = torch.rand(100, 1, requires_grad=True)
    u_pred = model(x_collocation)
    # Auto-differentiation for du/dx
    du_dx = torch.autograd.grad(u_pred, x_collocation, 
                                grad_outputs=torch.ones_like(u_pred), 
                                create_graph=True)[0]
    pde_residual = du_dx + u_pred
    loss_pde = torch.mean(pde_residual**2)
    
    # 2. Boundary Condition Loss
    x_bc = torch.tensor([[0.0]])
    u_bc_pred = model(x_bc)
    loss_bc = (u_bc_pred - 1.0)**2
    
    # Total loss
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()

# Test
x_test = torch.tensor([[1.0]])
u_test = model(x_test).item()
print(f"PINN Prediction at x=1: {u_test:.4f} (True: {0.3679:.4f})")
```

### 6.2 Fourier Neural Operator (FNO) 1D Layer

Implementing the spectral convolution layer in PyTorch.

```python
import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply
        self.scale = 1.0 / (in_channels * modes1)
        
        # Learnable complex weights
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x shape: (batch, in_channels, grid_size)
        batchsize = x.shape[0]
        
        # 1. Compute Fourier transform (FFT)
        x_ft = torch.fft.rfft(x)
        
        # 2. Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  
                             device=x.device, dtype=torch.cfloat)
                             
        # Complex multiplication: (batch, in, modes) * (in, out, modes) -> (batch, out, modes)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", 
                                                  x_ft[:, :, :self.modes1], 
                                                  self.weights1)
        
        # 3. Inverse Fourier transform (IFFT)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

# Test the layer
modes = 16
layer = SpectralConv1d(in_channels=1, out_channels=1, modes1=modes)
# Batch of 4, 1 channel, spatial grid of 64
x_input = torch.randn(4, 1, 64)
output = layer(x_input)
print("FNO Input Shape:", x_input.shape)
print("FNO Output Shape:", output.shape)
```

