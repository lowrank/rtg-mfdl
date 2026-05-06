# Topic 05: Information Theory in Deep Learning — 10-Hour Intensity Curriculum

This curriculum provides an exhaustive, 10-hour deep-dive into the information-theoretic foundations of modern Artificial Intelligence. We move from the rigorous calculus of differential entropy to the cutting-edge controversies of the Information Bottleneck and the Riemannian geometry of probability distributions.

---

## Curriculum Structure

### [05-1: Foundations and Differential Entropy](./05-1-foundations-differential-entropy.md)
*The Calculus of Uncertainty.*

- **Core Concepts:** Shannon Entropy, Differential Entropy, and the quantization limit.
- **Rigorous Proofs:** Entropy Power Inequality (Stam-Blachman-Dembo), Concentration of Surprisal in high-dim Gaussians, and the AWGN Channel Coding Theorem.
- **Key Insight:** Information is a physical quantity. Learning is the process of selective forgetting (DPI).
- **Practical Engineering:** Handling estimation bias and normalization in high-dimensional representations.

### [05-2: The Information Bottleneck Controversy](./05-2-information-bottleneck-controversy.md)
*Does Deep Learning Compress?*

- **The Theory:** Gaussian Information Bottleneck and the rigorous derivation of phase transitions from covariance eigenvalues.
- **The Controversy:** Saxe et al.'s proof of infinite information in deterministic ReLU networks and the refutation of the "compression phase."
- **Modern Synthesis:** Understanding "Functional Compression" (invariance) vs. "Information-Theoretic Compression."
- **Demos:** Visualizing the "Information Plane" for Tanh vs. ReLU networks.

### [05-3: Rate-Distortion Theory and VAEs](./05-3-rate-distortion-theory-vaes.md)
*The Architecture of Lossy Compression.*

- **Core Theory:** The Rate-Distortion function $R(D)$ and the Blahut-Arimoto convergence proof.
- **VAE Link:** Formal proof connecting the VAE ELBO Lagrangian to the Shannon Rate-Distortion objective.
- **Disentanglement:** The $\beta$-VAE conjecture and the information-theoretic pressure for independent factors.
- **Demos:** Implementing Blahut-Arimoto and tracking empirical R-D curves in neural codecs.

### [05-4: Variational Mutual Information Estimation](./05-4-variational-mi-estimation.md)
*Quantifying Information in High Dimensions.*

- **Deep Proofs:** The Nguyen-Wainwright-Jordan (NWJ) f-divergence variational bound for MI.
- **Estimators:** InfoNCE and the rigorous $\log N$ bound; variance-bias trade-offs in contrastive learning.
- **Contrastive Learning:** Proof that SimCLR and CLIP are maximizing a lower bound on Mutual Information.
- **Demos:** Benchmarking MINE, InfoNCE, and NWJ estimators on 1000-dimensional synthetic datasets.

### [05-5: Information Geometry and Natural Gradients](./05-5-information-geometry-natural-gradient.md)
*Optimization on the Statistical Manifold.*

- **The Manifold:** Chentsov’s Theorem on the uniqueness of the Fisher Information Metric.
- **Boltzmann Geometry:** The dualistic structure $(\theta, \eta)$ of Restricted Boltzmann Machines.
- **Advanced Optimization:** Natural Gradient for Wasserstein space and its connection to the JKO flow/Fokker-Planck equation.
- **Demos:** K-FAC approximations for Natural Gradient in deep networks.

---

## 10-Hour Intensity Learning Path

| Hour | Activity | Focus |
| :--- | :--- | :--- |
| **1-2** | **Foundations** | Master the EPI proof and high-dim Gaussian surprisal. Understand why $h(X)$ can be negative. |
| **3-4** | **The Bottleneck** | Derive the GIB phase transitions. Replicate the Saxe et al. binning artifact demo. |
| **5-6** | **Generative Theory** | Prove BA convergence. Map the VAE loss terms to the Shannon Rate-Distortion objective. |
| **7-8** | **Practical Estimation** | Implement NWJ and InfoNCE. Analyze the variance of MINE in high dimensions. |
| **9-10**| **Geometry** | Sketch Chentsov’s proof. Derive the Natural Gradient for Wasserstein space (JKO flow). |

---

## High-Level Synthesis: Why Information Theory?

1.  **Objective Functions are Information Metrics:** Cross-Entropy is $H(P, Q)$, KL-Divergence is the basis of VAEs, and Mutual Information is the heart of Contrastive Learning.
2.  **Generalization is Compression:** A model that generalizes is one that has found a compact "description" of the data generating process (MDL Principle).
3.  **Representations are Bottlenecks:** Every layer in a neural network is a noisy channel. Information theory tells us what we *cannot* do (DPI) and what we *must* do (IB) to learn effectively.

---
*Reference: Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory. Wiley-Interscience.*

