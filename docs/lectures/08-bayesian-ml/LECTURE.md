# Topic 08: Bayesian and Probabilistic Machine Learning (10-Hour Intensity)

## Curriculum Overview

This module provides a rigorous, high-density exploration of Bayesian methods in modern machine learning. We transition from foundational probabilistic logic to the frontiers of infinite-width neural networks and distribution-free uncertainty quantification. Every sub-module includes formal proofs, worked examples, and engineering guidelines.

---

### [08.1 Probabilistic Foundations and Bayesian Neural Networks](./08-1-probabilistic-foundations-bnn.md)
**Focus**: Treating learning as inference and the axiomatic roots of probability.

- **Hours 1-2**:
- **Cox’s Theorem**: Formal proof that probability theory is the unique extension of Boolean logic to plausible reasoning.
- **de Finetti’s Theorem**: Rigorous derivation showing how priors naturally emerge from exchangeable data sequences.
- **Bayesian Optimality**: Proof that the posterior predictive distribution minimizes expected risk under any proper loss function.
- **Bayesian Neural Networks**: Analysis of the high-dimensional weight space and the "Evidence Gap" in deep learning.

### [08.2 MCMC: Hamiltonian Monte Carlo and Langevin Dynamics](./08-2-mcmc-hmc-sampling.md)
**Focus**: Efficient sampling from complex, non-convex posteriors.

- **Hours 3-4**:
- **Markov Chain Theory**: Proof that Detailed Balance implies stationarity; spectral gap analysis (Perron-Frobenius) for mixing times.
- **Hamiltonian Dynamics**: Formal proofs of Volume Preservation and Time-Reversibility for the Leapfrog Integrator.
- **SGLD**: Convergence of stochastic gradient Langevin dynamics to the true posterior via the Fokker-Planck equation.
- **Practical HMC**: Tuning mass matrices and diagnosing divergent transitions in NUTS.

### [08.3 Gaussian Processes and the NNGP Correspondence](./08-3-gp-nngp-correspondence.md)
**Focus**: The mathematical bridge between function-space and weight-space.

- **Hours 5-6**:
- **Mercer’s Theorem**: Rigorous proof of the eigenfunction expansion of positive-definite kernels.
- **The Infinite-Width Limit**: Step-by-step derivation (Neal, 1996) showing why wide networks converge to GPs.
- **Deep NNGP**: Kernel recursion formulas for deep ReLU networks and the "Arc-Cosine" kernel.
- **The Edge of Chaos**: Theoretical analysis of information propagation through infinite-depth kernels.

### [08.4 Variational Inference and Normalizing Flows](./08-4-variational-inference-flows.md)
**Focus**: Casting inference as an optimization problem.

- **Hours 7-8**:
- **MFVI Convergence**: Rigorous proof of the monotone increase of the ELBO during coordinate ascent (CAVI).
- **Reparameterization Trick**: Variance reduction analysis for gradient estimation in VAEs.
- **Normalizing Flows**: Change-of-variables formula and the efficiency of triangular Jacobians (RealNVP).
- **Universality**: Proof sketch that autoregressive maps can represent any continuous probability density.

### [08.5 Calibration and Conformal Prediction](./08-5-conformal-prediction-calibration.md)
**Focus**: Rigorous uncertainty guarantees without parametric assumptions.

- **Hours 9-10**:
- **Conformal Coverage**: Exhaustive proof of the $(1-\alpha)$ coverage guarantee under exchangeability.
- **Proper Scoring Rules**: Proofs that the Brier score and Log-score are proper (minimized by the true distribution).
- **Uncertainty Wrapping**: Adaptive Prediction Sets (APS) for classification and Conformal Quantile Regression (CQR).
- **Engineering SOTA**: Temperature scaling, Venn-Abers predictors, and handling distribution shift.

---

## 10-Hour Intensity Path

1.  **Phase 1: Foundations (2h)**: Study [08.1]. Focus on the de Finetti proof and the Occam's Razor effect in marginal likelihood.
2.  **Phase 2: Sampling Mechanics (2h)**: Study [08.2]. Implement HMC from scratch and visualize Hamiltonian trajectories.
3.  **Phase 3: Infinite-Width Theory (2h)**: Study [08.3]. Derive the ReLU kernel and analyze spectral decay.
4.  **Phase 4: Optimization-based Inference (2h)**: Study [08.4]. Implement a Coupling Layer and verify ELBO convergence.
5.  **Phase 5: Rigorous Guarantees (2h)**: Study [08.5]. Run conformal experiments on heteroscedastic datasets and plot calibration curves.

## Theoretical Synergy
This module connects the "top-down" approach of Bayesian priors with the "bottom-up" approach of distribution-free guarantees. Students will move from "learning weights" to "estimating densities" and finally "guaranteeing intervals," providing a complete toolkit for high-stakes AI engineering.

