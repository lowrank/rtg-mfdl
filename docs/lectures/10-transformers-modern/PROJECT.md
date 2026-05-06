# Topic 10: Project - The Mechanics of Transformer Intelligence

## 1. Objective
This project aims to bridge the gap between abstract Transformer theory and empirical behavior. You will choose one of two tracks to investigate either the **Optimization Hypothesis** (ICL as GD) or the **Algorithmic Hypothesis** (Mechanistic Interpretability of circuits).

---

## 2. Track A: Verifying "ICL as Implicit Gradient Descent"

### 2.1 The Goal
Mathematically verify if a Transformer trained on synthetic linear regression tasks actually learns to implement a step of Gradient Descent in its forward pass.

### 2.2 Dataset and Setup

- **Task**: Linear Regression $y = w^\top x$.
- **Prompt Structure**: $\{ (x_1, y_1), (x_2, y_2), \dots, (x_k, y_k), x_{\text{test}} \}$.
- **Data Generation**: 
- Sample $w \sim \mathcal{N}(0, I_d)$.
- Sample $x_i \sim \mathcal{N}(0, I_d)$.
- Compute $y_i = w^\top x_i$.
- **Kaggle Link**: [Synthetic Regression for ICL Analysis](https://www.kaggle.com/datasets/arunimsamudra/linear-regression-dataset) (or generate your own using a script).

### 2.3 Implementation Steps

1.  **Architecture**: Build a small Transformer with **Linear Attention** (no softmax).
2.  **Training**: Train the model on a curriculum of sequence lengths $k \in [5, 50]$.
3.  **Analysis**:
- Extract the attention weights $A = QK^\top$.
- Compare the model's prediction $\hat{y}$ with the prediction from a single-step OLS solution: $y_{OLS} = x_{\text{test}}^\top (X^\top X)^{-1} X^\top Y$.
- Measure the Correlation between the Transformer's effective weights and the GD weights $W_{GD} = \eta X^\top Y$.

### 2.4 Expected Results

- You should observe that as the model trains, the MSE approaches the OLS error.
- A visualization of the attention matrix should show high values on the "Value" tokens corresponding to the "Key" tokens that match the current query.

---

## 3. Track B: Reverse-Engineering Modular Arithmetic

### 3.1 The Goal
Identify the "Fourier Circuit" in a Transformer trained on modular addition $(a + b) \pmod p$ and observe the phenomenon of **Grokking**.

### 3.2 Dataset and Setup

- **Task**: Predict $c = (a + b) \pmod p$ given the string "a b =".
- **Parameters**: Use $p = 113$. Split the $p^2$ possible equations into 30% training and 70% validation.
- **Kaggle/Reference**: [Grokking Dataset](https://www.kaggle.com/code/shreydan/grokking-modular-arithmetic-with-transformers).

### 3.3 Implementation Steps

1.  **Training**: Train a 1-layer Transformer (use `TransformerLens` or `nanoGPT`). Use a high weight decay (critical for grokking).
2.  **Observation**: Plot training vs. validation accuracy. Note the "Grokking point" where validation accuracy suddenly spikes long after training accuracy hits 100%.
3.  **Circuit Discovery**:
- Perform an FFT on the embedding matrix.
- Check for "Trigonometric Identities" in the weights. The model should learn to map tokens to frequencies $\omega_k = 2\pi k / p$.
- Use **Activation Patching** to see which specific neurons in the MLP are responsible for the modular jump.

### 3.4 Expected Results

- **Weight Norm**: You should see the $L_2$ norm of the weights decrease at the moment of grokking, indicating the model is switching from a "lookup table" (high norm) to a "trig formula" (low norm).
- **Embeddings**: A 2D PCA of the token embeddings should reveal a perfect circle.

---

## 4. Expected Results and Analysis Section (Common to both)

### 4.1 Comparative Metrics

| Model Configuration | Task Accuracy/MSE | Complexity (FLOPs) | Interpretation |
| :--- | :--- | :--- | :--- |
| Linear Attention (L=1) | | | "One-step Optimizer" |
| Softmax Attention (L=1)| | | "Kernel Smoother" |
| Deep Transformer (L=6) | | | "Multi-step / Algorithmic" |

### 4.2 Analysis Questions

1.  **Phase Transitions**: Did your model exhibit a sharp "Aha!" moment (Grokking) or a smooth scaling improvement?
2.  **Circuit Parsimony**: Is the learned circuit the *simplest* possible mathematical solution? How does weight decay influence this?
3.  **Generalization**: For Track A, does the model generalize to $d$ higher than seen in training? For Track B, does it generalize to $a, b$ not seen in training?

---

## 5. Deliverables

1.  **Jupyter Notebook**: Fully documented implementation using `PyTorch` and `TransformerLens`.
2.  **Visualizations**:
- Attention maps showing induction or pattern matching.
- SVD/PCA plots of embeddings.
- Training curves showing the "Grokking" gap.
3.  **Write-up**: A 2-page report connecting your empirical findings to the theorems discussed in `LECTURE.md` (e.g., Rank Collapse, NW-Kernel connection).

