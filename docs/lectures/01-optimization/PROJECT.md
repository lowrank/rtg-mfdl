# Topic 01: Project — Optimization for Advanced Regression

## 1. Objective
Apply advanced optimization techniques (custom optimizers and second-order approximations) to a high-dimensional regression task: the **Kaggle 'House Prices: Advanced Regression Techniques'** dataset. You will investigate how second-order information (Gauss-Newton) affects convergence speed and stability compared to standard first-order methods (Adam, SGD).

## 2. Dataset Setup

- **Source**: [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Task**: Predict `SalePrice` based on 79 explanatory variables.
- **Challenges**: High dimensionality, categorical features, non-linear relationships.

### Data Exploration Tips

1.  **Skewness**: The target `SalePrice` is often right-skewed. Apply a $\log(1+x)$ transformation to normalize it.
2.  **Missing Values**: Many features have `NaN` values. Some signify "None" (e.g., `PoolQC`), while others are truly missing data.
3.  **Multicollinearity**: Features like `GarageCars` and `GarageArea` are highly correlated. Consider dropping one or using PCA.

---

## 3. Step-by-Step Implementation Guide

### Step 1: Preprocessing Pipeline
Use `pandas` and `scikit-learn` to prepare the data for PyTorch.

- Fill numerical NaNs with the median.
- Use `OneHotEncoder` for categorical variables (handle "unknown" categories).
- **Crucial**: Use `StandardScaler` on all input features. Second-order methods are sensitive to feature scaling.

### Step 2: The Model
Implement a 3-layer MLP in PyTorch.

```python
import torch.nn as nn

class HousePriceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

```

### Step 3: Custom Gauss-Newton Optimizer
For a residual vector $\mathbf{r}(w) = \mathbf{y}_{pred} - \mathbf{y}_{true}$, the Gauss-Newton update is:


$$
\Delta w = -(\mathbf{J}^T \mathbf{J})^{-1} \mathbf{J}^T \mathbf{r}
$$


where $\mathbf{J} = \frac{\partial \mathbf{y}_{pred}}{\partial w}$.

**Implementation Detail**: Use `torch.func.jacrev` to compute the Jacobian for a mini-batch. Note that $\mathbf{J}^\top \mathbf{J}$ is an $M \times M$ matrix where $M$ is the number of parameters. For small networks, this is feasible. For larger ones, use block-diagonal approximations.

### Step 4: Training and Monitoring
Run three training loops:

1.  **SGD** (Learning rate $10^{-2}$).
2.  **Adam** (Learning rate $10^{-3}$).
3.  **Gauss-Newton** ($\lambda = 0.1$).

---

## 4. Expected Results and Analysis

### What to look for?

1.  **Loss Curves**: The Gauss-Newton optimizer should show much sharper initial drops in loss compared to SGD. Does it reach a lower plateau?
2.  **Hessian Trace**: Use the **Hutchinson Trace Estimator** (sample $v \sim \mathcal{N}(0, I)$, compute $v^\top \mathbf{H} v$ via two backprops) to monitor the curvature. Second-order methods often "ride" along narrow valleys.
3.  **Overfitting**: Does the faster convergence of Gauss-Newton lead to worse generalization on the validation set? Compare the gap $|Loss_{train} - Loss_{val}|$.

### Table: Performance Comparison

| Optimizer | Final RMSE (Log) | Epochs to Converge | Max Eigenvalue $\lambda_{max}$ |
| :--- | :--- | :--- | :--- |
| SGD | | | |
| Adam | | | |
| Gauss-Newton | | | |

---

## 5. Extensions (Optional)

- **Damping Schedule**: Implement a Levenberg-Marquardt schedule for $\lambda$. Increase $\lambda$ if the loss increases; decrease it if the loss decreases.
- **K-FAC**: Implement a Kronecker-factored version of the second-order update to handle wider layers more efficiently.
