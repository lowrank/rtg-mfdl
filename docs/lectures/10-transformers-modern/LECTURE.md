# Topic 10: Theory of Transformers and Modern Architectures (10-Hour Intensity)

This intensive module provides a deep mathematical and mechanistic dive into the Transformer architecture and its successor State Space Models. It is designed for researchers and engineers seeking to understand the "why" behind the dominant paradigms in AI.

---

## Course Syllabus

### Hour 1-2: Mathematical Foundations of Attention

*   **The Statistical View:** Rigorous derivation of self-attention as an asymmetric Nadaraya-Watson kernel estimator.
*   **Expressivity & Turing Completeness:** Step-by-step proof of the Perez et al. (2019) theorem on the Turing Completeness of Transformers with infinite precision.
*   **Failure Modes of Pure Attention:** Exhaustive proof of the Rank Collapse Theorem (Dong et al., 2021). Why $res(\mathbf{X})$ decays doubly exponentially without skip connections.
*   **Worked Examples:** Kernel equivalence, identity limit, and hard-attention degeneracies.

### Hour 3-4: Transformers as Meta-Optimizers

*   **The Mesa-Optimization Hypothesis:** Proof of the LSA-GD equivalence (Von Oswald et al., 2023). How one layer of attention implements one step of Gradient Descent.
*   **Iterative Refinement:** Multi-layer construction for multi-step GD and preconditioned Newton methods.
*   **The Bayesian Perspective:** Xie et al. (2022) proof of In-Context Learning as implicit Bayesian Inference over latent HMM concepts.
*   **Coding Demos:** Building a "Transformer-Optimizer" for linear regression from scratch.

### Hour 5-6: Scaling Laws and the Manifold Hypothesis

*   **The Chinchilla Derivation:** Full derivation of compute-optimal $(N, D)$ allocation using Lagrange Multipliers under power-law constraints.
*   **The Geometric Origin of Exponents:** Theoretical derivation of the scaling exponent $\alpha \approx 4/d$ from the intrinsic dimensionality of the data manifold.
*   **Engineering Scaling:** Inference-optimal vs. Compute-optimal regimes. Over-training theories (Llama-3 analysis).
*   **Worked Examples:** Budget allocation for 10B to 1T parameter regimes.

### Hour 7-8: Mechanistic Interpretability and Circuits

*   **Circuit Complexity:** Rigorous proof that induction heads require $L \ge 2$ layers due to the necessity of K-composition.
*   **Grokking Phase Transitions:** Formal analysis of the grokking transition using Nanda et al.'s Fourier progress measures. Competition between memorization and generalization circuits.
*   **Superposition & SAEs:** Mathematical foundations of non-orthogonal superposition and the use of Sparse Autoencoders for feature decomposition.
*   **Coding Demos:** Induction head detection and Fourier logit spectrum analysis.

### Hour 9-10: Beyond Attention: SSMs and Linear Complexity

*   **Optimal Memory (HiPPO):** Full rigorous proof of the optimality of the HiPPO matrix for online polynomial projection.
*   **Parallel Recurrence:** Proof of the associative property of the selective scan operation. Parallel prefix scan algorithms ($O(\log L)$).
*   **Modern SSMs (Mamba):** Discretization via Zero-Order Hold (ZOH) and hardware-aware SRAM fusion.
*   **The Frontier:** State Space Duality (Mamba-2) and the formal equivalence between SSMs and Linear Attention.

---

## Prerequisites

*   **Linear Algebra:** Spectral theorem, Singular Value Decomposition (SVD), Matrix calculus.
*   **Probability:** Bayesian inference, Concentration of measure, Markov chains.
*   **Optimization:** Lagrange multipliers, Gradient descent dynamics, Newton's method.
*   **Signal Processing:** Discrete Fourier Transform (DFT), Orthogonal polynomials.

---

## Learning Objectives
By the end of this 10-hour intensive, students will be able to:

1.  **Prove** the conditions under which a Transformer is Turing complete.
2.  **Derive** compute-optimal training configurations for new data domains.
3.  **Reverse-engineer** small Transformers to identify algorithmic circuits like induction heads.
4.  **Implement** parallel associative scans for linear-time sequence modeling.
5.  **Mathematically audit** new architectures for potential rank collapse or memory bottlenecks.

