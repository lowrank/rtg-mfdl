# 02 — Approximation Theory: The Expressive Power of Networks

Approximation theory answers the fundamental question of existence: *Which functions can a neural network represent, and how efficiently?* This module explores the transition from classical polynomial approximation to the high-dimensional regime of neural networks. We will analyze Universal Approximation Theorems (width vs. depth), the "Curse of Dimensionality" and how smoothness in the Fourier domain can overcome it, and modern architectural innovations like Kolmogorov-Arnold Networks (KANs).

**Prerequisite Tier**: Tier 2 — Intermediate (Real Analysis, Functional Analysis, Linear Algebra)

---

## 🎯 Learning Objectives

- Prove the density of neural networks in $C(K)$ using the Hahn-Banach Theorem.
- Quantify the benefits of depth through the lens of "Sawtooth" function compositions.
- Derive Barron's Bound to explain dimension-independent approximation rates.
- Compare and contrast Multi-Layer Perceptrons (MLPs) with Kolmogorov-Arnold Networks (KANs).

---

## 📚 Course Modules

- [**Lecture**: Unified Mathematical Foundations](./LECTURE.md)
- [**Practice**: Exercises and Open Questions](./PRACTICE.md)
- [**Project**: Expressivity and Depth Separation](./PROJECT.md)

---

## 📄 Essential Reading

- **Cybenko, G. (1989)**: [Approximation by Superpositions of a Sigmoidal Function](https://link.springer.com/article/10.1007/BF02551274) - *The seminal paper on universal approximation.*
- **Barron, A. R. (1993)**: [Universal approximation bounds for superpositions of a sigmoidal function](https://ieeexplore.ieee.org/document/256545) - *Bridging the gap between neural networks and the curse of dimensionality.*
- **Telgarsky, M. (2016)**: [Benefits of depth in neural networks](https://arxiv.org/abs/1602.04485) - *A modern proof of depth separation using simple geometric constructions.*

