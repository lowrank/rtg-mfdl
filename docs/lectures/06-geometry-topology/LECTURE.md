# Lecture 06: Geometry, Topology, and Equivariance

## Course Overview: 10-Hour Intensity Track

This module explores the intersection of differential geometry, algebraic topology, and group theory with modern machine learning. We move beyond Euclidean assumptions to build models that respect the intrinsic structure of data, whether it be symmetries (Equivariance), global shape (TDA), or curved constraints (Riemannian Optimization).

---

## Part 1: Group Representation Theory & Steerability (2.5 Hours)
**Focus:** How to build networks that "know" about rotations, reflections, and permutations.

- **Foundations:** Group Algebras $\mathbb{C}[G]$, Group Homomorphisms, and Linear Representations.
- **The Atomic Structure of Symmetry:** Rigorous proof of **Maschke's Theorem** (Semi-simplicity) and the decomposition of feature spaces into Irreps.
- **The Equivariant Blueprint:** **Schur's Lemma** and its role in constraining linear layers to block-sparse forms.
- **Steerable CNNs:** Induced Representations $\text{Ind}_H^G \rho$, Fiber Bundles, and the derivation of the **Kernel Constraint Equation**.
- **Continuous Symmetries:** Lie Groups and Lie Algebras ($SO(3), \mathfrak{so}(3)$), the Exponential Map, and Rodrigues' Formula.

---

## Part 2: Spectral & Spatial Graph Neural Networks (2.5 Hours)
**Focus:** The mathematical limits of learning on relational structures.

- **Spectral Graph Theory:** Eigen-decomposition of the Laplacian, the Graph Fourier Transform, and Dirichlet Energy.
- **Localization Guarantees:** Rigorous proof of the **Hammond et al. Theorem** on the localization of polynomial filters in the spatial domain.
- **Expressive Power:** The **Weisfeiler-Lehman (WL) Hierarchy**. Proof that MPNNs are bounded by 1-WL.
- **Higher-Order GNNs:** $k$-WL algorithms and the separation between 1-WL and 2-WL graphs (e.g., distinguishing cycles).
- **Beyond Graphs:** Introduction to Simplicial and Cellular Neural Networks.

---

## Part 3: Persistent Homology & TDA (2 Hours)
**Focus:** Quantifying the "shape" of data and neural network manifolds.

- **Simplicial Topology:** Chains, Cycles, and Boundaries. Computation via Boundary Matrix Reduction.
- **Persistence Modules:** The evolution of Betti numbers across scales.
- **Theoretical Guarantees:** Rigorous proof of the **Stability Theorem** (Chazal et al.) using Algebraic Interleaving.
- **TDA of Weights:** The **Neural Persistence** $(\mathcal{P})$ metric. Rigorous proof of topological complexity growth during generalization (Rieck et al.).
- **Vectorization:** Persistence Landscapes and Images for integration with Deep Learning.

---

## Part 4: Optimal Transport & Wasserstein Geometry (1.5 Hours)
**Focus:** Moving mass and aligning incomparable datasets.

- **The Monge-Kantorovich Problem:** Couplings, Marginals, and the $W_p$ metric.
- **The Geometry of $W_2$:** **Brenier’s Theorem**. Rigorous proof of the existence of the optimal transport map as a gradient of a convex potential.
- **Fast Solvers:** **Sinkhorn's Theorem** and the Entropy-Regularized OT formulation.
- **Relational Alignment:** **Gromov-Wasserstein** distance. Definition and proof of the Triangle Inequality.
- **Applications:** Domain Adaptation, Color Transfer, and Graph Matching.

---

## Part 5: Riemannian Manifolds & Optimization (1.5 Hours)
**Focus:** Learning on spheres, hyperbolas, and Grassmannians.

- **Differential Geometry:** Tangent Spaces, Riemannian Metrics, and the **Levi-Civita Connection**.
- **Manifold Optimization:** The Riemannian Gradient and the **Retraction** operator.
- **Convergence Analysis:** Rigorous proof of the convergence of Retraction-based Gradient Descent for smooth functions on manifolds.
- **Non-Euclidean ML:** **Hyperbolic Embeddings** for hierarchies and **Stiefel Manifolds** for orthogonal weight constraints.
- **State-of-the-Art:** Mixed-curvature manifolds and Riemannian Transformers.

---

## Recommended Practice
See `PRACTICE.md` for 10 high-intensity mathematical problems and `PROJECT.md` for the implementation of a Geometric Clifford Algebra Network.

