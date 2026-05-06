# 01 — Optimization Theory: The Engine of Neural Learning

This module provides a rigorous exploration of the mathematical foundations of optimization in high-dimensional spaces. We transition from classical convex optimization—where global convergence is guaranteed—to the modern, non-convex landscapes of deep neural networks. Students will analyze the geometry of loss surfaces, the convergence properties of gradient-based algorithms under various regularity conditions, and the information-theoretic perspectives that drive second-order optimization methods.

**Prerequisite Tier**: Tier 2 — Intermediate (Multivariable Calculus, Linear Algebra, Probability)

---

## 🎯 Learning Objectives

- Analyze the distribution of critical points in high-dimensional non-convex landscapes.
- Derive convergence rates for Gradient Descent under $L$-smoothness, Strong Convexity, and the Polyak-Łojasiewicz (PL) condition.
- Understand the geometry of Mirror Descent and its application to constrained optimization.
- Evaluate the efficacy of second-order methods like Natural Gradient Descent and K-FAC in deep learning.

---

## 📚 Course Modules

- [**Lecture**: Unified Mathematical Foundations](./LECTURE.md)
- [**Practice**: Exercises and Open Questions](./PRACTICE.md)
- [**Project**: Convergence Analysis and Visualization](./PROJECT.md)

---

## 📄 Essential Reading

- **Bottou, L., Curtis, F. E., & Nocedal, J. (2018)**: [Optimization methods for large-scale machine learning](https://arxiv.org/abs/1606.04838) - *The definitive survey on the transition from batch to stochastic optimization.*
- **Reddi, S. J., Kale, S., & Kumar, S. (2018)**: [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) - *A critical look at the convergence issues of popular adaptive methods.*
- **Cohen, J. M., et al. (2021)**: [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065) - *An exploration of non-classical dynamics in neural optimization.*

