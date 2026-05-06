# Topic 03 Project: Empirical Generalization Bounds on 'Spaceship Titanic'

## 1. Objective
The goal of this project is to apply statistical learning theory to a real-world dataset. Instead of just aiming for high accuracy, you will calculate **Empirical Generalization Bounds** to provide a theoretical guarantee on your model's performance. You will explore the trade-offs between optimization, stability, and complexity.

## 2. Dataset Setup

- **Source**: [Kaggle Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic)
- **Task**: Binary classification (Predict if a passenger was transported to another dimension).
- **Features**: Numerical (Age, Spending) and Categorical (HomePlanet, Cabin, Destination).

### Data Exploration and Preprocessing Tips

1.  **Missing Values**: Use a sophisticated imputer (like `IterativeImputer` or `KNNImputer`) rather than simple mean filling.
2.  **Feature Engineering**: The `Cabin` feature can be split into `Deck`, `Num`, and `Side`. These are highly predictive.
3.  **Theoretical Constraint**: To calculate Rademacher and PAC-Bayes bounds, your inputs must be bounded. Use `StandardScaler` and then clip values to $[-L, L]$.

---

## 3. Step-by-Step Implementation Guide

### Step 1: Baseline Training
Train a 3-layer MLP using binary cross-entropy loss.

- Record the training loss and validation accuracy.
- Save the model weights $w^*$.

### Step 2: Estimating Rademacher Complexity
How well can your model fit pure noise?

1.  Replace the training labels $y_i$ with random Rademacher variables $\sigma_i \in \{1, -1\}$.
2.  Train the model to maximize correlation: $\max_w \frac{1}{n} \sum \sigma_i \Phi(x_i; w)$.
3.  Repeat for 5 trials. The average correlation is your estimate $\hat{\mathcal{R}}_S(\mathcal{H})$.
4.  **Analysis**: Compare this to the training accuracy on real labels. If the model fits noise almost as well as real data, it is likely over-parameterized.

### Step 3: PAC-Bayes Certificate
Provide a "guarantee" that the test error will not exceed a certain threshold.

1.  **Define Posterior $Q$**: $\mathcal{N}(w^*, \sigma^2 I)$. Optimize $\sigma$ to minimize the bound.
2.  **Define Prior $P$**: $\mathcal{N}(0, \lambda I)$, where $\lambda$ is related to your weight initialization.
3.  **Compute McAllester's Bound**:


$$
\epsilon = \sqrt{\frac{KL(Q || P) + \ln(2\sqrt{n}/\delta)}{2(n-1)}}
$$


4.  Calculate $R(Q) = \mathbb{E}_{h \sim Q} [\hat{R}_n(h)]$ by sampling model weights and averaging the training loss.
5.  **Target**: Can you get a bound $\epsilon < 0.5$ (non-vacuous)?

### Step 4: Double Descent Visualization

1.  Vary the number of hidden neurons $H$ from 1 to 1000.
2.  For each $H$, train the model until the training loss is near zero.
3.  Plot Train and Test Error vs. $H$.
4.  Identify the "peak" where the model starts to interpolate and observe if the test error decreases again.

---

## 4. Expected Results and Analysis

### What to look for?

1.  **Complexity vs. Regularization**: Show how Weight Decay ($\lambda$) affects the estimated Rademacher complexity. Does higher $\lambda$ consistently lead to lower complexity?
2.  **Bound Tightness**: PAC-Bayes bounds on neural networks are notoriously loose. Discuss why your bound might be vacuous ($>1$) and how "Sharpness-Aware Minimization" might help.
3.  **Stability**: Perform a "Leave-one-out" experiment. Does removing a single outlier passenger significantly change the model's decision boundary?

### Results Table

| Model Config | Train Error | Val Error | Estimated Rademacher | PAC-Bayes Bound ($\epsilon$) |
| :--- | :--- | :--- | :--- | :--- |
| MLP (Small) | | | | |
| MLP (Large) | | | | |
| MLP + SAM | | | | |

---

## 5. Deliverables

- A Jupyter notebook demonstrating the bound calculations.
- A plot of the "Double Descent" curve for this specific dataset.
- A short discussion on whether the theoretical bounds (Rademacher/PAC-Bayes) correctly predicted the relative performance of different architectures.
