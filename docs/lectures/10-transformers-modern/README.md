# Topic 10: Theory of Transformers and Modern Architectures

This module explores the theoretical foundations of the Transformer architecture, which has become the backbone of modern large language models (LLMs). We move beyond simple architecture descriptions to analyze Transformers through the lenses of kernel methods, optimization theory, and mechanistic interpretability.

## 📁 Module Structure

*   [**LECTURE.md**](./LECTURE.md): Deep dive into the mathematical framework of attention, In-Context Learning (ICL) as implicit optimization, and neural scaling laws.
*   [**PRACTICE.md**](./PRACTICE.md): Theoretical exercises ranging from permutation equivariance proofs to deriving softmax variance, plus hands-on coding tasks.
*   [**PROJECT.md**](./PROJECT.md): A guide to reverse-engineering Transformer circuits or verifying the "ICL as Gradient Descent" hypothesis using synthetic datasets.

## 🎯 Learning Objectives

1.  **Mathematical Foundation**: Understand the self-attention mechanism as a learned, non-parametric kernel smoother.
2.  **Emergent Behavior**: Formalize the hypothesis of In-Context Learning as an implicit implementation of optimization algorithms.
3.  **Efficiency and Scaling**: Analyze the theoretical limits of attention (rank collapse) and the empirical power laws governing model scaling.
4.  **Interpretability**: Learn the methodology for identifying functional "circuits" within high-dimensional weight matrices.

## 📚 Key References

*   **Attention Foundation**: Vaswani et al. (2017), *Attention is All You Need*.
*   **Scaling Laws**: Kaplan et al. (2020), *Scaling Laws for Neural Language Models*.
*   **Transformer Circuits**: Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*.
*   **ICL Theory**: Von Oswald et al. (2023), *Transformers learn in-context by gradient descent*.

