# Project 07: Physical Modeling with Differential Equations

## 1. Project Overview
This project focuses on applying modern continuous-time deep learning architectures to real-world physical or time-series data. You will choose between modeling irregular time-series or solving a fluid dynamics problem.

---

## 2. Option A: Neural ODEs for Irregular Time-Series

### Goal
Implement a **Latent ODE** to handle irregularly sampled data, commonly found in medical records or financial markets.

### Dataset

- [PhysioNet Sepsis Challenge 2019](https://archive.physionet.org/challenge/2019/): Irregularly sampled physiological data from ICU patients.
- Alternative: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).

### Implementation Steps

1. **Data Preprocessing**: Handle missing values by creating a mask and recording the observation times $t_i$.
2. **Recognition Network**: Use a GRU or an ODE-RNN to encode the sequence into a latent vector $z_0$ at $t=0$.
3. **ODE Evolution**:
   - Define a Neural Network $f_\phi(z, t)$ as the vector field.
   - Use `torchdiffeq.odeint` to compute $z(t_i)$ for all observation times.
4. **Decoder**: Map each $z(t_i)$ to the observation space (e.g., linear layer).
5. **Loss Function**: $\mathcal{L} = \sum_i \| \hat{x}_i - x_i \|^2 + \text{KL}(q(z_0|x) \| p(z_0))$.

### Expected Results

- The Latent ODE should outperform standard RNNs when a large portion of the data is missing.
- Visualization: Plot the "latent trajectories" to see how the model interpolates between distant observations.

---

## 3. Option B: PINNs for Fluid Dynamics

### Goal
Solve the **2D Navier-Stokes equations** for flow over a cylinder using a Physics-Informed Neural Network.

### Dataset / Domain

- Domain: $x \in [0, 2], y \in [0, 1]$.
- Cylinder centered at $(0.5, 0.5)$ with radius $0.05$.
- Equations:

    $$
    \nabla \cdot \mathbf{u} = 0
    $$

    $$
    \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}
    $$


### Implementation Steps

1. **Network Architecture**: A deep MLP that takes $(x, y, t)$ as input and outputs $(u, v, p)$.
2. **Collocation Points**: Generate a grid of points in the domain and on the boundaries (including the cylinder surface).
3. **Automatic Differentiation**: Use `torch.autograd.grad` to compute $\partial u/\partial t$, $\partial u/\partial x$, $\partial^2 u/\partial x^2$, etc.
4. **Residual Calculation**: Plug the derivatives into the Navier-Stokes equations to get the residual $e_{NS}$.
5. **Optimization**: Use the Adam optimizer followed by L-BFGS for fine-tuning the residual to zero.

### Expected Results

- The PINN should reconstruct the characteristic "von Kármán vortex street" behind the cylinder.
- **Inverse Problem**: If you provide sparse velocity data, the model should be able to "discover" the viscosity $\nu$.

---

## 4. Analysis & Tips

### Tips for Success

- **Neural ODEs**: Use `method='rk4'` or `method='dopri5'`. If training is slow, try reducing the solver tolerance (`rtol`, `atol`).
- **PINNs**: Scaling is crucial. Ensure $x, y, t$ and the outputs $u, v, p$ are roughly in the same order of magnitude. Use `tanh` or `Sine` (SIREN) activations for better second-order derivatives.
- **Kaggle Link**: [Fluid Dynamics Datasets](https://www.kaggle.com/search?q=fluid+dynamics).

### Expected Results Section
Your final report should include:

1. **Accuracy**: Mean Squared Error (MSE) compared to a traditional numerical solver (e.g., OpenFOAM or a FDM script).
2. **Efficiency**: A plot of Loss vs. Iterations.
3. **Physics Consistency**: A plot showing the "Residual Field" — where in the domain the physics equations are most violated.
4. **Generalization**: How well the model performs on time steps outside the training range (extrapolation).
