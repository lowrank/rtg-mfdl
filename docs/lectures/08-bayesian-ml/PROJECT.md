# Project 08: Uncertainty and Probabilistic Inference

## 1. Project Overview
This project focuses on quantifying uncertainty in deep learning models using either Bayesian sampling (HMC) or distribution-free methods (Conformal Prediction).

---

## 2. Option A: Uncertainty in Medical Cost Prediction (Bayesian)

### Goal
Build a robust regression model with rigorous uncertainty bounds using **Hamiltonian Monte Carlo (HMC)**.

### Dataset

- [Medical Insurance Cost Prediction](https://www.kaggle.com/datasets/mirichoi0218/insurance): Predict individual medical costs billed by health insurance.
- Alternative: [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set).

### Implementation Steps

1. **Model Definition**: Define a Bayesian Neural Network with 1-2 hidden layers. Use a Gaussian prior for the weights.
2. **Hamiltonian Monte Carlo**:
   - Use `Pyro` or `Hamiltorch` (PyTorch) to implement the HMC sampler.
   - Run the chain for at least 1000 samples after burn-in.

3. **Diagnostics**: Plot the trace of the weights and check the R-hat ($\hat{R}$) statistic to ensure convergence.
4. **Prediction**:
   - For a test input $x^*$, compute the predictive distribution $p(y^*|x^*, D) \approx \frac{1}{M} \sum_m p(y^*|x^*, w^{(m)})$.
   - Calculate the mean and the $95\%$ credible interval.

### Expected Results

- The BNN should provide wider uncertainty intervals for regions with sparse data.
- **Analysis**: Compare the "Calibration" of the BNN against a standard MLP. A well-calibrated model's $95\%$ interval should contain the true value exactly $95\%$ of the time.

---

## 3. Option B: Conformal Prediction for Safe Classification (Frequentist)

### Goal
Create a classification system that outputs a *set* of labels, guaranteed to contain the true label with $99\%$ confidence.

### Dataset

- [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): Detect fraudulent transactions.
- Alternative: [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) or [CIFAR-10].

### Implementation Steps

1. **Base Model**: Train a standard classifier (e.g., Logistic Regression, Random Forest, or a CNN).
2. **Data Splitting**: Split the training data into a training set ($80\%$) and a calibration set ($20\%$).
3. **Non-conformity Scores**:
   - For the calibration set, compute $S_i = 1 - \hat{f}(x_i)_{y_i}$, where $\hat{f}(x_i)_{y_i}$ is the predicted probability of the *true* class.

4. **Threshold Calculation**: Calculate $\hat{q}$, the $(1-\alpha)$ quantile of $S_i$ (e.g., $\alpha=0.01$ for $99\%$ coverage).
5. **Set Generation**: For a new $x$, include all classes $y$ such that $1 - \hat{f}(x)_y \le \hat{q}$.

### Expected Results

- For ambiguous images (e.g., a digit that looks like both 1 and 7), the model should return a set $\{1, 7\}$ rather than a single incorrect guess.
- **Analysis**: Plot the "Average Set Size" vs. the confidence level $(1-\alpha)$. Show how the set size increases as you demand higher confidence.

---

## 4. Analysis & Deliverables

### Technical Report Requirements

1. **Uncertainty Plot**: For regression, show a plot of $y$ vs. $x$ with the $95\%$ uncertainty band. For classification, show examples of "Multi-label sets".
2. **Reliability Diagram**: Plot "Observed Coverage" vs. "Nominal Coverage". The points should lie on the $y=x$ line.
3. **Comparison**: Contrast Option A (Bayesian) and Option B (Conformal). 
   - Which one is more computationally expensive? 
   - Which one makes more assumptions about the data?

4. **Kaggle Link**: [Uncertainty Estimation Datasets](https://www.kaggle.com/search?q=uncertainty).

### Tips

- **HMC**: Start with a small network. Sampling from a deep ResNet with HMC is extremely difficult.
- **Conformal**: Ensure your calibration set is truly representative of your test set (exchangeability).

