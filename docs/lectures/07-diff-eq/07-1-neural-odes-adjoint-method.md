# 7.1 Neural ODEs and the Adjoint Method

## 1. Introduction to Continuous-Depth Models

Traditional deep learning models, such as Residual Networks (ResNets), transform hidden states through a discrete sequence of layers:

$$
h_{t+1} = h_t + f(h_t, \theta_t)
$$

This formulation closely resembles the Euler method for numerically solving ordinary differential equations (ODEs). Neural Ordinary Differential Equations (Neural ODEs) take this analogy to its continuous limit, parameterizing the derivative of the hidden state using a neural network:

$$
\frac{dz(t)}{dt} = f(z(t), t, \theta)
$$

Here, $z(t) \in \mathbb{R}^D$ is the hidden state at "time" (or depth) $t$, and $f$ is a neural network parameterized by $\theta$. The output of the network is computed by solving this ODE using a black-box differential equation solver from time $t_0$ to $t_1$:

$$
z(t_1) = z(t_0) + \int_{t_0}^{t_1} f(z(t), t, \theta) dt
$$

This continuous-depth formulation offers several advantages: $O(1)$ memory cost during training, adaptive computation, and the ability to model continuous-time time-series data naturally. 

In this section, we provide rigorous proofs for the existence and uniqueness of the solutions to these ODEs, followed by the derivation of the Adjoint Sensitivity Method, which enables efficient gradient computation.

---

## 2. Existence and Uniqueness: The Picard-Lindelöf Theorem

Before training a Neural ODE, we must ensure that the forward pass (solving the ODE) is well-defined. This is guaranteed by the Picard-Lindelöf Theorem (also known as the Cauchy-Lipschitz Theorem), which establishes the existence and uniqueness of solutions to first-order ODEs with given initial conditions.

### 2.1 Theorem Statement

!!! success "Theorem (Picard-Lindelöf)"
    Consider the initial value problem:

    $$
    z(t_0) = z_0, \quad \frac{dz(t)}{dt} = f(z(t), t)
    $$

    Suppose $f(z, t)$ is uniformly Lipschitz continuous in $z$ and continuous in $t$ on a rectangular region $R = \{ (z, t) \in \mathbb{R}^D \times \mathbb{R} : \|z - z_0\| \le b, |t - t_0| \le a \}$. That is, there exists a constant $L > 0$ such that for all $(z_1, t), (z_2, t) \in R$:

    $$
    \|f(z_1, t) - f(z_2, t)\| \le L \|z_1 - z_2\|
    $$

    Then, there exists some $\delta > 0$ such that there is a unique continuous and differentiable function $z(t)$ defined on $[t_0 - \delta, t_0 + \delta]$ that satisfies the initial value problem.

### 2.2 Proof

The proof relies on converting the ODE into an integral equation and applying the Banach Fixed-Point Theorem.

**Step 1: Reformulation as an Integral Equation**

A function $z(t)$ solves the ODE if and only if it satisfies the Volterra integral equation:

$$
z(t) = z_0 + \int_{t_0}^t f(z(s), s) ds
$$

We define the Picard operator $T$ acting on a space of continuous functions $C([t_0 - \delta, t_0 + \delta], \mathbb{R}^D)$:

$$
T[z](t) = z_0 + \int_{t_0}^t f(z(s), s) ds
$$

Our goal is to show that $T$ has a unique fixed point, i.e., $T[z] = z$.

**Step 2: Defining the Complete Metric Space**

Let $I = [t_0 - \delta, t_0 + \delta]$. We choose $\delta < \min(a, \frac{b}{M}, \frac{1}{L})$, where $M = \sup_{(z, t) \in R} \|f(z, t)\|$. 
Consider the space of continuous functions $X = \{ z \in C(I, \mathbb{R}^D) : \|z(t) - z_0\| \le b \text{ for all } t \in I \}$. We equip $X$ with the uniform (supremum) norm:

$$
\|z\|_\infty = \sup_{t \in I} \|z(t)\|
$$

Since $X$ is a closed subset of the Banach space $C(I, \mathbb{R}^D)$, it is a complete metric space.

**Step 3: $T$ maps $X$ into $X$**

For any $z \in X$ and $t \in I$:

$$
\|T[z](t) - z_0\| = \left\| \int_{t_0}^t f(z(s), s) ds \right\| \le \left| \int_{t_0}^t \|f(z(s), s)\| ds \right| \le M |t - t_0| \le M \delta
$$

Since we chose $\delta \le \frac{b}{M}$, we have $\|T[z](t) - z_0\| \le b$. Thus, $T[z] \in X$.

**Step 4: $T$ is a Contraction Mapping**

Let $z_1, z_2 \in X$. We evaluate the distance between $T[z_1]$ and $T[z_2]$:

$$
\|T[z_1](t) - T[z_2](t)\| = \left\| \int_{t_0}^t (f(z_1(s), s) - f(z_2(s), s)) ds \right\|
$$

Using the Lipschitz condition:

$$
\|T[z_1](t) - T[z_2](t)\| \le \left| \int_{t_0}^t \|f(z_1(s), s) - f(z_2(s), s)\| ds \right| \le \left| \int_{t_0}^t L \|z_1(s) - z_2(s)\| ds \right|
$$

Taking the supremum over $t \in I$:

$$
\|T[z_1] - T[z_2]\|_\infty \le L \delta \|z_1 - z_2\|_\infty
$$

Since we chose $\delta < \frac{1}{L}$, the factor $L\delta < 1$. Therefore, $T$ is a strict contraction on the complete metric space $X$.

**Step 5: Applying Banach Fixed-Point Theorem**

By the Banach Fixed-Point Theorem, $T$ has a unique fixed point $z \in X$. This fixed point satisfies the integral equation, and hence the differential equation. The existence and uniqueness of the forward pass in a Neural ODE are thus rigorously established, provided the neural network $f$ uses Lipschitz-continuous activation functions (like ReLU, Tanh, or Sigmoid). $\blacksquare$

---

## 3. The Adjoint Sensitivity Method

Standard backpropagation through an ODE solver requires storing intermediate states $z(t)$ at every evaluation point of the solver, which scales with the number of steps (often hundreds). The **Adjoint Sensitivity Method** bypasses this by solving a second ODE backward in time, reducing memory complexity to $O(1)$.

### 3.1 Theorem and Formulation

Consider a scalar loss function $L$ dependent on the terminal state:

$$
L = \mathcal{L}(z(t_1))
$$

Our objective is to compute the gradients of $L$ with respect to the initial state $z(t_0)$ and the parameters $\theta$. We define the **adjoint state** $a(t)$ as the gradient of the loss with respect to the hidden state at time $t$:

$$
a(t) = \frac{\partial L}{\partial z(t)}
$$

!!! success "Theorem (Adjoint Dynamics)"
    The adjoint state $a(t)$ satisfies the following differential equation backward in time:

    $$
    \frac{da(t)}{dt} = -a(t) \frac{\partial f(z(t), t, \theta)}{\partial z(t)}
    $$

    Furthermore, the gradient with respect to the parameters is given by:

    $$
    \frac{d L}{d \theta} = - \int_{t_1}^{t_0} a(t) \frac{\partial f(z(t), t, \theta)}{\partial \theta} dt
    $$

### 3.2 Rigorous Derivation

**Step 1: Constraint via Lagrange Multipliers**

We formulate the problem using the calculus of variations. The ODE constraint must hold for all $t$. We incorporate it into the loss function using a time-dependent Lagrange multiplier $\lambda(t)$ (which will turn out to be the adjoint state $a(t)$):

$$
\tilde{L} = \mathcal{L}(z(t_1)) - \int_{t_0}^{t_1} \lambda(t)^T \left( \frac{dz(t)}{dt} - f(z(t), t, \theta) \right) dt
$$

Note that since the constraint is exactly satisfied, $\tilde{L} = L$. We now take the total variation of $\tilde{L}$ with respect to $z$ and $\theta$. 

**Step 2: Integration by Parts**

Consider the variation $\delta \tilde{L}$ caused by an infinitesimal change in parameters $\delta \theta$, which induces a trajectory change $\delta z(t)$:

$$
\delta \tilde{L} = \frac{\partial \mathcal{L}}{\partial z(t_1)} \delta z(t_1) - \int_{t_0}^{t_1} \lambda(t)^T \left( \frac{d(\delta z(t))}{dt} - \frac{\partial f}{\partial z} \delta z(t) - \frac{\partial f}{\partial \theta} \delta \theta \right) dt
$$

We apply integration by parts to the term involving $\frac{d(\delta z)}{dt}$:

$$
\int_{t_0}^{t_1} \lambda(t)^T \frac{d(\delta z(t))}{dt} dt = \left[ \lambda(t)^T \delta z(t) \right]_{t_0}^{t_1} - \int_{t_0}^{t_1} \frac{d\lambda(t)^T}{dt} \delta z(t) dt
$$

Substituting this back into the variation equation:

$$
\delta \tilde{L} = \frac{\partial \mathcal{L}}{\partial z(t_1)} \delta z(t_1) - \lambda(t_1)^T \delta z(t_1) + \lambda(t_0)^T \delta z(t_0) + \int_{t_0}^{t_1} \left( \frac{d\lambda(t)^T}{dt} + \lambda(t)^T \frac{\partial f}{\partial z} \right) \delta z(t) dt + \int_{t_0}^{t_1} \lambda(t)^T \frac{\partial f}{\partial \theta} \delta \theta dt
$$

**Step 3: Enforcing the Adjoint Equation**

To compute the gradient $\frac{dL}{d\theta}$, we must eliminate the dependency on the unknown variation $\delta z(t)$. We do this by judiciously choosing the Lagrange multiplier $\lambda(t)$ such that the term multiplying $\delta z(t)$ vanishes for all $t \in [t_0, t_1]$:

$$
\frac{d\lambda(t)^T}{dt} + \lambda(t)^T \frac{\partial f}{\partial z} = 0
$$

Taking the transpose, we obtain the adjoint ODE:

$$
\frac{d\lambda(t)}{dt} = - \left( \frac{\partial f}{\partial z} \right)^T \lambda(t)
$$

Similarly, to eliminate the $\delta z(t_1)$ term, we set the terminal condition for $\lambda(t_1)$:

$$
\lambda(t_1) = \left( \frac{\partial \mathcal{L}}{\partial z(t_1)} \right)^T
$$

Thus, $\lambda(t)$ is exactly the adjoint state $a(t)$.

**Step 4: Parameter Gradients**

With $\delta z(t)$ eliminated, the variation of the loss simplifies to:

$$
\delta L = a(t_0)^T \delta z(t_0) + \left( \int_{t_0}^{t_1} a(t)^T \frac{\partial f}{\partial \theta} dt \right) \delta \theta
$$

From this, the exact gradients are immediately readable:

$$
\frac{\partial L}{\partial z(t_0)} = a(t_0), \quad \frac{\partial L}{\partial \theta} = \int_{t_0}^{t_1} a(t)^T \frac{\partial f}{\partial \theta} dt
$$

This completes the proof. $\blacksquare$

By stacking the states $z(t)$, $a(t)$, and the parameter gradient integrals into a single augmented ODE state, one backward pass using an ODE solver can compute all required gradients simultaneously.

---

## 4. Worked Examples

### 4.1 Example: Linear ODE Gradient
Consider $f(z, t, \theta) = \theta z$, with $z(0) = z_0 = 1$ and target time $t_1 = 1$. The loss is $L(z(1)) = \frac{1}{2} z(1)^2$.

1. **Forward Pass:** $\frac{dz}{dt} = \theta z \implies z(1) = e^\theta$.
2. **Loss:** $L = \frac{1}{2} e^{2\theta}$. Target analytical gradient: $\frac{dL}{d\theta} = e^{2\theta}$.
3. **Adjoint Method:**
   Terminal condition: $a(1) = \frac{\partial L}{\partial z(1)} = z(1) = e^\theta$.
   Adjoint ODE: $\frac{da}{dt} = -a(t) \theta \implies a(t) = a(1) e^{-\theta(t-1)} = e^\theta e^{-\theta(t-1)} = e^{\theta(2-t)}$.

4. **Parameter Gradient Integration:**
   
   $$
   \frac{dL}{d\theta} = \int_0^1 a(t) \frac{\partial f}{\partial \theta} dt = \int_0^1 e^{\theta(2-t)} z(t) dt
   $$

   Since $z(t) = e^{\theta t}$, we have:
   
   $$
   \frac{dL}{d\theta} = \int_0^1 e^{\theta(2-t)} e^{\theta t} dt = \int_0^1 e^{2\theta} dt = e^{2\theta}
   $$

   This matches the analytical gradient exactly.

### 4.2 Example: Vector-Valued ODE
Let $z = [x, y]^T$. $f(z, \theta) = [\theta_1 y, -\theta_2 x]^T$. Loss $L = x(T)$.
Adjoint equation: $a(T) = [1, 0]^T$.
Jacobian: $\frac{\partial f}{\partial z} = \begin{pmatrix} 0 & \theta_1 \\ -\theta_2 & 0 \end{pmatrix}$.
Adjoint ODE: $\frac{da}{dt} = -\begin{pmatrix} 0 & -\theta_2 \\ \theta_1 & 0 \end{pmatrix} a(t)$.
The gradients are computed via the integral $\int_0^T [a_1 y, -a_2 x]^T dt$.

### 4.3 Example: Non-Linear ODE with Adjoint
Consider $f(z, \theta) = \tanh(\theta z)$, initial condition $z(0)=z_0$.
Jacobian: $\frac{\partial f}{\partial z} = \theta \text{sech}^2(\theta z)$.
Adjoint ODE: $\frac{da}{dt} = -a(t) \theta \text{sech}^2(\theta z(t))$.
Parameter gradient: $\frac{\partial f}{\partial \theta} = z \text{sech}^2(\theta z)$.
$\frac{dL}{d\theta} = \int_0^T a(t) z(t) \text{sech}^2(\theta z(t)) dt$.

---

## 5. Coding Demonstrations

### 5.1 Basic Neural ODE with Euler Method (PyTorch)

This demo shows a naive Euler integration. We construct a simple 1D ODE $z' = \theta z$.

```python
import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([1.0]))

    def forward(self, t, z):
        return self.theta * z

def euler_solve(func, z0, t_start, t_end, steps=100):
    dt = (t_end - t_start) / steps
    z = z0
    t = t_start
    for _ in range(steps):
        z = z + dt * func(t, z)
        t = t + dt
    return z

# Forward pass and manual backprop through solver
func = ODEFunc()
z0 = torch.tensor([1.0])
z_T = euler_solve(func, z0, 0.0, 1.0)
loss = 0.5 * z_T**2
loss.backward()

print(f"Predicted z(1): {z_T.item():.4f} (True: 2.7183)")
print(f"Gradient dL/dTheta: {func.theta.grad.item():.4f} (True: 7.3891)")
```

### 5.2 Using `torchdiffeq` and Adjoint Method

To leverage the memory advantages of the Adjoint method, we use the `torchdiffeq` library.

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class NeuralODEFunc(nn.Module):
    def __init__(self):
        super(NeuralODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        
    def forward(self, t, y):
        return self.net(y)

# Initial condition and integration time
y0 = torch.tensor([[1.0, 0.0]])
t = torch.linspace(0., 1., 10) # 10 time points to evaluate

# Instantiate and run ODE solver
func = NeuralODEFunc()

# Forward pass: uses standard ODE solver (e.g., dopri5)
y_pred = odeint(func, y0, t)

# Compute loss against some target trajectory
y_target = torch.sin(t).view(-1, 1, 1).repeat(1, 1, 2)
loss = torch.mean((y_pred - y_target)**2)

# Backward pass: internally uses the Adjoint Sensitivity Method!
# Memory is O(1) w.r.t the number of time steps.
loss.backward()

print("Loss computed:", loss.item())
print("Gradients populated for Neural ODE layers.")
```

By substituting `odeint_adjoint` for `odeint`, the backward pass automatically handles the augmented state vector integration, shielding the deep learning researcher from manually constructing the backward solver while providing identical mathematical exactness.

