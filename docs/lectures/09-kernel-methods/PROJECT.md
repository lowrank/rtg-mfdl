# Topic 09: Project - Scaling Kernels to Large Data

## 1. Objective
The goal of this project is to implement and benchmark large-scale kernel approximation techniques—**Random Fourier Features (RFF)** and the **Nyström Method**—against a standard Deep Learning baseline.

### Dataset

- [Forest Cover Type Prediction](https://www.kaggle.com/c/forest-cover-type-prediction): Multi-class classification of forest types based on cartographic variables.
- Size: $\approx 581,000$ samples.
- Task: Classify into one of 7 forest cover types.

---

## 2. Implementation Steps

### Step 1: Data Preparation

1. Download the dataset from Kaggle.
2. Normalize all continuous features to zero mean and unit variance.
3. Split the data into Training ($70\%$), Validation ($15\%$), and Test ($15\%$).

### Step 2: Exact Kernel (Small Scale)

1. Take a subset of $N=5000$ points.
2. Train a standard Kernel Ridge Classifier using the exact RBF kernel.
3. Record the training time and accuracy.

### Step 3: Random Fourier Features (RFF)

1. Implement an RBF feature extractor:
   - Sample $\omega \sim \mathcal{N}(0, 2\gamma I)$.
   - $\phi(x) = \sqrt{\frac{2}{m}} \cos(\omega^\top x + b)$.

2. Train a **Linear SGD Classifier** on the features $\phi(x)$ for $m \in \{100, 500, 1000, 5000\}$.
3. Benchmark the training time and accuracy on the *full* dataset.

### Step 4: Nyström Method

1. Select $m$ landmark points from the training set (use uniform sampling).
2. Compute the low-rank approximation $\hat{K}$.
3. Solve the resulting system using the Nyström formula.
4. Benchmark for the same values of $m$.

---

## 3. Analysis & Expected Results

### Performance Metrics

| Model | m | Accuracy | Training Time | Memory |
| :--- | :--- | :--- | :--- | :--- |
| Exact (Small) | N/A | | | |
| RFF + Linear | 1000 | | | |
| RFF + Linear | 5000 | | | |
| Nyström | 1000 | | | |
| Nyström | 5000 | | | |
| MLP Baseline | N/A | | | |

### Expected Analysis Section
In your report, address the following:

1. **Accuracy vs. m**: Plot how the accuracy improves as $m$ increases. Is there a "diminishing returns" point?
2. **Approximation Quality**: Compute the MSE between the exact Gram matrix and the approximations (on the $5000$ point subset). Which method is more accurate for a fixed $m$?
3. **Efficiency**: Discuss the memory trade-off. RFF requires $O(m \times d)$ space for weights, while Nyström requires storing the landmarks $O(m \times d)$ and the $m \times m$ matrix.
4. **Conclusion**: When would you recommend using a kernel approximation over a modern Deep Neural Network (MLP)?

---

## 4. Deliverables

1. **Source Code**: A Python script or Jupyter Notebook (`kernel_bench.py`).
2. **Project Report**: Including the plots and analysis mentioned above.
3. **Kaggle Submission**: (Optional) Submit your RFF-based predictions to the Kaggle leaderboard and report your rank.

### Tips

- Use `sklearn.kernel_approximation` for optimized implementations, but try to write your own RFF sampler first to understand the math.
- For Nyström, the landmark points can be chosen via K-means for better coverage of the data manifold.

