# Topic 04 — Random Matrix Theory and High-Dimensional Probability

Weight matrices, Jacobians, and feature covariance matrices in deep neural networks are effectively high-dimensional random matrices. This module explores the foundational results of Random Matrix Theory (RMT), the Neural Tangent Kernel (NTK) regime, and the transition from "lazy" training to feature learning in modern large-scale models.

**Prerequisite Tier**: Tier 2 — Intermediate (Linear Algebra, Probability, Calculus)

---

## 📚 Course Modules

- [**Lecture**: Unified Mathematical Foundations](./LECTURE.md)
  *Historical context, Wigner & Marchenko-Pastur laws, NTK derivation, and scaling regimes.*

- [**Practice**: Exercises and Solutions](./PRACTICE.md)
  *Theoretical proofs on Stieltjes transforms, concentration of measure, and coding tasks for spectral analysis.*

- [**Project**: NTK and Spectral Analysis](./PROJECT.md)
  *Empirical study of infinite-width limits using the NLP Disaster Tweets dataset.*

---

## 📄 Key Research Literature

- **Vershynin, R. (2018)**: [High-Dimensional Probability](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html) - *The definitive text on concentration and random matrices.*
- **Jacot, A., et al. (2018)**: [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) - *The paper that launched the NTK field.*
- **Yang, G. (2022)**: [Tensor Programs V: Tuning Large Neural Networks at Scale](https://arxiv.org/abs/2203.03466) - *Introduction to the $\mu$P framework used for GPT-4.*
- **Belkin, M., et al. (2019)**: [Reconciling Deep Learning with Classical Overfitting](https://arxiv.org/abs/1812.11118) - *Explains the Double Descent phenomenon.*

