# Undergraduate Summer Projects: Mathematical Foundations of AI / ML / DL

A curated catalog of summer research projects (8–12 weeks) for undergraduates
interested in the mathematical foundations of machine learning and deep
learning. 

The catalog is organized into 10 core topics. Each topic is structured as a 
**10-HOUR INTENSIVE CURRICULUM** consisting of:

- **📖 LECTURE Hub**: A master portal linking five high-density sub-modules (50+ total).
- **📚 Deep Dive Submodules**: Exhaustive mathematical treatments with **rigorous proofs for every theorem**, non-trivial examples, and state-of-the-art research.
- **🛠 PRACTICE**: Specialized theoretical exercises and coding implementations with hints.
- **🚀 PROJECT**: Hands-on research projects utilizing real-world data from **Kaggle**.

---

## 📁 Contents

| #  | Topic Hub | Topic Area | Directory |
|----|-----------|------------|-----------|
| 01 | [Topic 01](lectures/01-optimization/README.md) | Optimization theory for deep learning | [lectures/01-optimization/README.md](lectures/01-optimization/README.md) |
| 02 | [Topic 02](lectures/02-approximation/README.md) | Approximation theory of neural networks | [lectures/02-approximation/README.md](lectures/02-approximation/README.md) |
| 03 | [Topic 03](lectures/03-learning-theory/README.md) | Statistical learning theory & generalization | [lectures/03-learning-theory/README.md](lectures/03-learning-theory/README.md) |
| 04 | [Topic 04](lectures/04-rmt-ntk/README.md) | Random matrix theory, NTK, and RMT | [lectures/04-rmt-ntk/README.md](lectures/04-rmt-ntk/README.md) |
| 05 | [Topic 05](lectures/05-information-theory/README.md) | Information theory in deep learning | [lectures/05-information-theory/README.md](lectures/05-information-theory/README.md) |
| 06 | [Topic 06](lectures/06-geometry-topology/README.md) | Geometry, topology, and equivariance | [lectures/06-geometry-topology/README.md](lectures/06-geometry-topology/README.md) |
| 07 | [Topic 07](lectures/07-diff-eq/README.md) | ODEs, SDEs, PDEs in deep learning | [lectures/07-diff-eq/README.md](lectures/07-diff-eq/README.md) |
| 08 | [Topic 08](lectures/08-bayesian-ml/README.md) | Bayesian and probabilistic ML | [lectures/08-bayesian-ml/README.md](lectures/08-bayesian-ml/README.md) |
| 09 | [Topic 09](lectures/09-kernel-methods/README.md) | Kernel methods and RKHS | [lectures/09-kernel-methods/README.md](lectures/09-kernel-methods/README.md) |
| 10 | [Topic 10](lectures/10-transformers-modern/README.md) | Theory of transformers & modern architectures | [lectures/10-transformers-modern/README.md](lectures/10-transformers-modern/README.md) |

---

## 🎓 Prerequisite Tiers

Each project is tagged with a prerequisite tier. Use this to pick projects
matching a student's background.

### Tier 1 — Foundational
Appropriate after 1st/2nd year of a math/CS degree.

- Multivariable calculus
- Linear algebra (eigendecomposition, SVD)
- Basic probability (expectation, variance, CLT)
- Python + NumPy/PyTorch basics

### Tier 2 — Intermediate
Typically 2nd/3rd year.

- **All of Tier 1**, plus:
- Real analysis (convergence, continuity, $\varepsilon$–$\delta$)
- Intro probability theory (σ-algebras, conditional expectation)
- Convex analysis / convex optimization
- Intro statistics (MLE, hypothesis testing)

### Tier 3 — Advanced
Typically late 3rd/4th year or honors students.

- **All of Tier 2**, plus:
- Measure-theoretic probability
- Functional analysis (Hilbert spaces, bounded operators)
- Stochastic processes / SDEs
- Graduate-level ML or optimization

---

## 🎯 Recommendations by Student Profile

### Profile A: Strong in analysis & probability, modest coding
Best picks:

- **03.1 PAC-Bayes nonvacuous bounds** — theorem-prove-experiment loop
- **01.3 Implicit bias of GD on separable data** — clean theory
- **02.1 Universal approximation theorems** — rigorous proofs + toy experiments
- **09.1 Random Fourier Features** — elegant probabilistic theorem + fast code

### Profile B: Strong coder, wants math exposure

- **04.1 Neural Tangent Kernel implementation**
- **01.2 Loss landscape visualization**
- **07.2 Score-based diffusion on toy data**
- **06.1 Intrinsic dimension estimation**

### Profile C: Physics / applied math background

- **07.1 Neural ODEs**
- **07.3 Physics-Informed Neural Networks**
- **04.2 Spectra of weight matrices (RMT)**
- **07.2 Score-based diffusion models (SDE formulation)**

### Profile D: Pure math, minimal ML coding

- **02.2 Depth separation / Barron spaces**
- **03.4 VC dimension of neural networks**
- **06.3 Equivariant neural networks** (representation theory)
- **09.2 RKHS and infinite-width networks**

### Profile E: Wants exposure to "hot" modern topics

- **10.2 In-context learning as implicit gradient descent**
- **07.2 Score-based diffusion models**
- **02.3 Kolmogorov–Arnold Networks**
- **10.3 Scaling laws**

---

## ⭐ "Sweet Spot" Projects

If you had to pick five — these have the best ratio of
**tractability × mathematical depth × modern relevance**:

1. **04.1 NTK vs. finite-width dynamics** (Tier 2)
2. **03.3 Double descent + random matrix theory** (Tier 2)
3. **01.3 Implicit bias of gradient descent** (Tier 2)
4. **07.2 Score-based diffusion on 2D toy data** (Tier 2–3)
5. **03.1 PAC-Bayes nonvacuous bounds** (Tier 2)

---

## 📚 Core Textbooks Referenced Throughout

| Book | Authors | Use |
|------|---------|-----|
| *Understanding Machine Learning: From Theory to Algorithms* | Shalev-Shwartz & Ben-David (2014) | Statistical learning theory (free PDF) |
| *High-Dimensional Probability* | Vershynin (2018) | Concentration, RMT, JL |
| *Convex Optimization* | Boyd & Vandenberghe (2004) | Optimization |
| *Foundations of Machine Learning* | Mohri, Rostamizadeh, Talwalkar (2018) | Generalization bounds |
| *Deep Learning* | Goodfellow, Bengio, Courville (2016) | General reference |
| *Mathematics for Machine Learning* | Deisenroth, Faisal, Ong (2020) | Gentle prerequisite book |
| *Pattern Recognition and Machine Learning* | Bishop (2006) | Bayesian ML |
| *Gaussian Processes for Machine Learning* | Rasmussen & Williams (2006) | GPs (free PDF) |
| *Elements of Information Theory* | Cover & Thomas (2006) | Information theory |
| *The Matrix Cookbook* | Petersen & Pedersen | Linear algebra reference |

---

## 🛠️ Suggested Common Stack

- **Language**: Python 3.10+
- **Deep learning**: PyTorch (primary), JAX (for NTK / scientific ML)
- **Scientific**: NumPy, SciPy, scikit-learn, matplotlib
- **Experiment tracking**: Weights & Biases or MLflow
- **Symbolic math (optional)**: SymPy, Mathematica
- **Version control**: Git + GitHub
- **Writing**: LaTeX (Overleaf) for final report

---

## 📝 Typical Project Deliverables

1. **Literature review** (5–10 pages) summarizing 3–5 key papers
2. **Reproducible code repository** with README and tests
3. **Technical report** (15–25 pages, LaTeX) with:
   - Background & mathematical formulation
   - Theorem statements and (where applicable) proofs or proof sketches
   - Experimental methodology and results
   - Discussion and open questions
4. **Final presentation** (20–30 minutes)
5. *(Optional, ambitious)*: Workshop/conference submission (e.g., ICML/NeurIPS workshops accept student work)

---

## 🗓️ Generic 10-Week Template

| Week | Activity |
|------|----------|
| 1 | Read 2–3 foundational papers; set up coding environment |
| 2 | Finish background reading; write literature-review draft |
| 3 | Reproduce a baseline result from a key paper |
| 4 | Implement core method/theorem verification |
| 5 | Run primary experiments; begin report draft |
| 6 | Midterm checkpoint; refine research question |
| 7 | Extend experiments (new regime / dataset / theorem variant) |
| 8 | Analyze results; write experiments section |
| 9 | Polish writeup; prepare figures |
| 10 | Final report + presentation |

---

## 📄 License & Use

This catalog is intended for faculty/mentors assembling REU-style project
offerings and for students selecting summer research directions. Individual
project files can be shared standalone.
