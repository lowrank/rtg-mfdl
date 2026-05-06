# Practice 07: Differential Equations in Deep Learning

## 1. Theoretical Exercises

### 1.1 Picard-Lindelöf and Neural ODEs
**Problem**: A Neural ODE is defined by $\dot{h} = f(h, t, \theta)$. Suppose $f(h, t, \theta) = \text{ReLU}(Wh + b)$. 

1. Is $f$ globally Lipschitz? 
2. Does the Picard-Lindelöf theorem guarantee a unique solution for all $t \in [0, T]$?
3. What happens if we use an activation function like $x^2$?

### 1.2 The Adjoint of a Linear ODE
**Problem**: Consider $\dot{h} = \Theta h$. Prove that the adjoint state $a(t)$ satisfies $\dot{a} = -\Theta^\top a$.
**Hint**: Use the definition $\dot{a} = -a^\top \frac{\partial f}{\partial h}$.

### 1.3 Score-based Models and Langevin Dynamics
**Problem**: The Score-based generative model uses the gradient of the log-density $\nabla_x \log p(x)$. 

1. Relate the Probability Flow ODE to the Langevin SDE $dx = \nabla_x \log p(x) dt + \sqrt{2} dW$.
2. Why is the score function easier to learn than the density $p(x)$ itself?

### 1.4 PINNs and Spectral Bias
**Problem**: Research "Spectral Bias" in neural networks. 

1. Why do PINNs struggle to solve PDEs with high-frequency components?
2. How do "Fourier Features" (Random Fourier Features) help mitigate this?

### 1.5 Conservation in Hamiltonian Neural Networks
**Problem**: Prove that for an HNN with $H_\theta(q, p)$, the total derivative $\frac{dH_\theta}{dt}$ is zero.
**Task**: Show $\frac{dH}{dt} = \frac{\partial H}{\partial q} \dot{q} + \frac{\partial H}{\partial p} \dot{p} = 0$ using the HNN equations of motion.

### 1.6 Divergence and Change of Variables
**Problem**: In the Probability Flow ODE, the change in log-probability is given by the Trace of the Jacobian (Divergence). 

1. For $x \in \mathbb{R}^d$, what is the computational complexity of computing $\text{Tr}(\frac{\partial f}{\partial x})$ exactly?
2. Explain the **Hutchinson's Trace Estimator** trick: $\text{Tr}(A) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [\epsilon^\top A \epsilon]$.

### 1.7 Stability of Symplectic Integrators
**Problem**: Why is the Forward Euler method ($q_{n+1} = q_n + \Delta t p_n, p_{n+1} = p_n - \Delta t \nabla V(q_n)$) not suitable for long-term simulation of a pendulum? 
**Task**: Show that the energy $E = \frac{1}{2}p^2 + \frac{1}{2}q^2$ increases over time for a simple harmonic oscillator under Forward Euler.

### 1.8 Neural Operators vs. PINNs
**Problem**: Compare the "Loss Function" of an FNO and a PINN. 

1. Which one requires labeled data (ground truth solutions)? 
2. Which one can be used in an "unsupervised" (physics-only) manner?

---

## 2. Coding Practice

### 2.1 Solving Lotka-Volterra with Neural ODE
**Task**: Use `torchdiffeq` to model the predator-prey system:


$$
\dot{x} = \alpha x - \beta xy, \quad \dot{y} = \delta xy - \gamma y
$$


1. Generate synthetic data using a standard ODE solver.
2. Train a Neural ODE to learn the dynamics from noisy observations.
3. Visualize the learned phase portrait $(x, y)$.

### 2.2 HMC: Sampling a Banana Distribution
**Task**: Implement a simple Hamiltonian Monte Carlo (HMC) sampler using the Leapfrog integrator to sample from the "Banana" distribution:


$$
p(x, y) \propto \exp\left( -\left( \frac{x^2}{200} + \frac{(y - 0.1x^2 + 5)^2}{2} \right) \right)
$$


1. Plot the trajectory of the particle in phase space.
2. Verify that the sampler explores the distribution better than standard Metropolis-Hastings.

---

## 3. Hints & Solutions

### 3.1 Hints

- **1.1**: ReLU is Lipschitz with constant $\|W\|$, but its derivative is discontinuous.
- **1.5**: Just substitute $\dot{q} = \partial H / \partial p$ and $\dot{p} = -\partial H / \partial q$.
- **2.1**: Define the vector field as a `nn.Module`. Use `odeint(func, y0, t)`.

### 3.2 Solutions (Brief)

- **1.2**: $\frac{\partial (\Theta h)}{\partial h} = \Theta$. Thus $\dot{a} = -a^\top \Theta \implies \dot{a} = -\Theta^\top a$ (for column vector $a$).
- **1.7**: For $\dot{q}=p, \dot{p}=-q$, Forward Euler gives $q_{n+1} = q_n + \Delta t p_n$ and $p_{n+1} = p_n - \Delta t q_n$. Then $q_{n+1}^2 + p_{n+1}^2 = (q_n + \Delta t p_n)^2 + (p_n - \Delta t q_n)^2 = (1 + \Delta t^2)(q_n^2 + p_n^2) > q_n^2 + p_n^2$. The energy grows exponentially.
