# Topic 07: Differential Equations in Deep Learning

This module provides a rigorous, 10-hour intensity exploration of the intersection between Continuous Dynamical Systems and Neural Architectures. We move beyond viewing neural networks as static layers, treating them as discretized trajectories of Ordinary, Stochastic, and Partial Differential Equations.

## Learning Objectives

1. **Continuous Depth**: Master the Adjoint Sensitivity Method and the transition from ResNets to Neural ODEs.
2. **Stochastic Generative Modeling**: Derive the reverse-time SDEs that power state-of-the-art diffusion models.
3. **Scientific ML**: Solve PDEs using PINNs and learn infinite-dimensional operators via FNOs.
4. **Optimization Physics**: Quantify SGD dynamics as a Langevin process with implicit regularization.
5. **Geometric Invariance**: Implement symplectic integrators and Hamiltonian Neural Networks for energy conservation.

---

## Syllabus: 10-Hour Intensity Track

### Module 1: Neural ODEs & The Adjoint Method (2 Hours)

- **Theory**: The Picard-Lindelöf existence theorem; ResNets as Euler discretizations.
- **Proof**: Exhaustive derivation of the Adjoint Sensitivity Equation via the calculus of variations ($O(1)$ memory backprop).
- **Nuance**: Stability analysis (A-stability, L-stability) and the "Stiffness" problem in neural dynamics.
- **Practice**: [07-1-neural-odes-adjoint-method.md](./07-1-neural-odes-adjoint-method.md)

### Module 2: Diffusion SDEs & Score Matching (2 Hours)

- **Theory**: Itô Calculus foundations; The Fokker-Planck equation derivation.
- **Proof**: Anderson's Reverse-Time SDE Theorem; Equivalence of Denoising Score Matching and Fisher Divergence.
- **Nuance**: Variance Preserving (VP) vs. Variance Exploding (VE) noise schedules.
- **Practice**: [07-2-diffusion-sdes-score-matching.md](./07-2-diffusion-sdes-score-matching.md)

### Module 3: PINNs & Neural Operators (2 Hours)

- **Theory**: Mesh-free PDE solvers and the Operator learning paradigm.
- **Proof**: The NTK analysis of PINN "Spectral Bias" (why high frequencies fail); Universal Approximation Theorem for Operators (Chen & Chen 1995).
- **Nuance**: Resolution Invariance in Fourier Neural Operators (FNO).
- **Practice**: [07-3-pinns-neural-operators.md](./07-3-pinns-neural-operators.md)

### Module 4: The Stochastic Dynamics of SGD (2 Hours)

- **Theory**: The Langevin SDE limit of mini-batch gradient descent.
- **Proof**: Stochastic Modified Equations (SME) and the $O(\eta)$ gradient-norm penalty; The Eyring-Kramers law for escape times from local minima.
- **Nuance**: Linear Scaling Rule derivation for batch-size synchronization.
- **Practice**: [07-4-stochastic-dynamics-sgd.md](./07-4-stochastic-dynamics-sgd.md)

### Module 5: Symplectic Integrators & HNNs (2 Hours)

- **Theory**: Hamiltonian mechanics in phase space; Symplectic geometry.
- **Proof**: Symplecticity of the Leapfrog scheme; Energy conservation theorem for HNNs.
- **Nuance**: Canonical coordinates and volume preservation in deep learning.
- **Practice**: [07-5-symplectic-integrators-hamiltonian.md](./07-5-symplectic-integrators-hamiltonian.md)

---

## Advanced Reading & References

### Foundational Texts

1. **Hairer, E., et al.**: *Solving Ordinary Differential Equations I & II*. (The bible of numerical solvers).
2. **Øksendal, B.**: *Stochastic Differential Equations*. (Essential for diffusion modeling).
3. **Arnold, V. I.**: *Mathematical Methods of Classical Mechanics*. (For Symplectic geometry).

### Key Papers

- **Chen et al. (2018)**: *Neural Ordinary Differential Equations*. (NeurIPS Best Paper).
- **Song et al. (2021)**: *Score-Based Generative Modeling through SDEs*. (The SDE unification of diffusion).
- **Li et al. (2020)**: *Fourier Neural Operator for Parametric PDEs*.
- **Wang et al. (2022)**: *When and Why PINNs Fail to Train*. (NTK analysis).
- **Greydanus et al. (2019)**: *Hamiltonian Neural Networks*.

---

## Guidelines for Practice

- **Verification**: Always compare Neural ODE solutions to high-order classic solvers (e.g., `SciPy`'s `Radau`) for ground truth.
- **Visualization**: Plot phase-space trajectories $(q, p)$ to verify energy conservation visually.
- **Metrics**: Use the Jacobian norm and solver step-counts as indicators of model complexity and training health.

