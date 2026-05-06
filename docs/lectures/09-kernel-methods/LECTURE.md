# Topic 09: Kernel Methods and Reproducing Kernel Hilbert Spaces (RKHS)

Welcome to the 10-hour intensive curriculum on Kernel Methods. This module bridges the rigorous depths of functional analysis with practical statistical learning theory, providing the foundations for Gaussian Processes, two-sample testing, and modern functional optimization.

## 0. Historical Context & Motivation

Kernel methods emerged from the convergence of functional analysis (specifically Aronszajn's theory of RKHS in the 1950s) and statistical learning theory. The 1960s saw the development of the "Kernel Trick" (Aizerman et al.), but it wasn't until the late 1990s, with the rise of Support Vector Machines (SVMs), that kernel methods revolutionized machine learning. Today, kernels form the foundational language for the Neural Tangent Kernel (NTK), deep functional distribution embeddings, and infinite-width neural network analysis.

---

## 1. Curriculum Overview

This module is structured into five high-density, rigorously proved lectures.

### [09-1: RKHS Foundations & The Moore-Aronszajn Theorem](./09-1-rkhs-foundations.md)
*Focus: Functional Analysis Backbone.*

- **Rigorous Proof**: The Moore-Aronszajn Theorem (Existence, uniqueness, and pre-Hilbert completion).
- **The Reproducing Property**: Proofs of Cauchy-Schwarz in RKHS and implications for uniform continuity.
- **Kernel Algebra**: Schur Product Theorem and positive-definite tensor mapping proofs.
- **Demos**: Definiteness checks, Gram heatmaps, and GP prior sampling.

### [09-2: The Representer Theorem & Kernel Optimization](./09-2-representer-theorem-optimization.md)
*Focus: Infinite-to-Finite Dimensional Reduction.*

- **Rigorous Proof**: The Generalized Representer Theorem (via projection theorems, accommodating non-convex losses).
- **Algorithms**: Exact derivations for Kernel Ridge Regression and the rigorous KKT derivation of the SVM Dual.
- **Geometric Margins**: Proving the relation between the dual quadratic form and the RKHS margin.
- **Demos**: Regularization path dynamics and exact SVM Dual solving via CVXOPT.

### [09-3: Mercer’s Theorem & Spectral Theory](./09-3-mercer-theory-spectral.md)
*Focus: The "Inside" of a kernel and Operator Theory.*

- **Integral Operators**: Compactness, self-adjointness, and strictly positive operators in $L^2(\mu)$.
- **Rigorous Proof**: Mercer’s Theorem (Spectral decomposition, absolute/uniform convergence via Dini's theorem).
- **Eigen-expansion**: Spectral decay rates (Exponential RBF vs Polynomial Matérn) and their control over RKHS smoothness.
- **Demos**: Numerical eigen-decomposition and visual extraction of Hermite-like Mercer eigenfunctions.

### [09-4: Large-Scale Kernels: RFF & Nyström](./09-4-large-scale-kernels-rff.md)
*Focus: Scaling $O(n^3)$ to Millions of Points.*

- **Rigorous Proof**: Bochner’s Theorem (Characterization of shift-invariant kernels via Fourier measures).
- **Approximations**: Derivation of Random Fourier Features (RFF) and Hoeffding uniform convergence bounds.
- **Variance Control**: Mathematical proof of variance reduction in Orthogonal Random Features (ORF).
- **Demos**: Benchmarking RFF vs Exact RBF and Nyström explicit rank approximations.

### [09-5: Kernel Mean Embeddings & MMD](./09-5-kernel-mean-embeddings-mmd.md)
*Focus: Advanced Non-Parametric Statistics on Distributions.*

- **Mean Embeddings**: Representing probability measures as vectors via the Riesz representer.
- **Characteristic Kernels**: Rigorous Fourier proof of Gaussian injectivity for Maximum Mean Discrepancy (MMD).
- **Two-Sample Testing**: The degenerate U-statistic limit theorem and null-distribution asymptotics.
- **Demos**: Implementing unbiased MMD, Permutation tests, and mapping the Witness Function.

---

## 2. Recommended Study Path (10-Hour Intensity)

1. **Functional Foundations (Hours 1-3)**: Read 09-1 and 09-3. You must comfortably write out the proofs of Moore-Aronszajn and Mercer. Understand the relationship between positive-definiteness and operator spectrums.
2. **Optimization & Practice (Hours 4-6)**: Work through 09-2 and the coding demos in 09-4. Verify the Representer projection step yourself. Understand the transition from primal infinite dimensions to the dual QP.
3. **Advanced Measure Theory (Hours 7-10)**: Study 09-5. Focus on the Mean Embedding integration property and the exact conditions for characteristic kernels. Master the coding implementations for non-parametric two-sample testing.

## 3. Core Insights

- **The Kernel Trick is a Projection**: It's not a superficial replacement of dot products. It establishes a closed subspace in an RKHS where the orthogonal components strictly do not impact empirical risk, dropping infinite search spaces to span of data.
- **Regularization is Frequency Filtering**: Choosing an RBF kernel means applying an infinite-order differential penalty, implicitly filtering out any non-smooth components based on the spectral decay derived from Bochner's / Mercer's theorems.
- **Kernels as Feature Extractors for Probability**: We bypass estimating high-dimensional densities $p(x)$ entirely by comparing their linear projections $\mu_P$ inside an RKHS, enabling robust topological statistics.
