# Topic 03: Statistical Learning Theory — Practice and Problem Set

## 1. Detailed Examples

### 1.1 Rademacher Complexity of Linear Classes
Consider the class of linear functions $\mathcal{H} = \{x \mapsto w \cdot x : \|w\|_2 \le B\}$ where inputs are bounded as $\|x_i\|_2 \le L$.

**Derivation**:

$$
\hat{\mathcal{R}}_S(\mathcal{H}) = \mathbb{E}_\sigma \left[ \sup_{\|w\| \le B} \frac{1}{n} \sum_{i=1}^n \sigma_i (w \cdot x_i) \right] = \frac{1}{n} \mathbb{E}_\sigma \left[ \sup_{\|w\| \le B} w \cdot \left( \sum_{i=1}^n \sigma_i x_i \right) \right]
$$

By the Cauchy-Schwarz inequality, $w \cdot v \le \|w\| \|v\|$. The supremum is attained when $w$ is in the direction of $v = \sum \sigma_i x_i$:

$$
\hat{\mathcal{R}}_S(\mathcal{H}) = \frac{B}{n} \mathbb{E}_\sigma \left\| \sum_{i=1}^n \sigma_i x_i \right\|_2
$$

Using Jensen's Inequality ($\mathbb{E}[X] \le \sqrt{\mathbb{E}[X^2]}$):

$$
\mathbb{E}_\sigma \left\| \sum \sigma_i x_i \right\| \le \sqrt{\mathbb{E}_\sigma \left\| \sum \sigma_i x_i \right\|^2} = \sqrt{\mathbb{E}_\sigma \sum_{i,j} \sigma_i \sigma_j (x_i \cdot x_j)}
$$

Since $\mathbb{E}[\sigma_i \sigma_j] = 0$ for $i \neq j$ and $1$ for $i=j$:

$$
\hat{\mathcal{R}}_S(\mathcal{H}) \le \frac{B}{n} \sqrt{\sum_{i=1}^n \|x_i\|^2} \le \frac{B}{n} \sqrt{nL^2} = \frac{BL}{\sqrt{n}}
$$

**Conclusion**: The complexity of linear models is independent of the input dimension $d$.

---

## 2. Problem Set (Theoretical)

### Exercise 1: Proving Hoeffding's Inequality

1.  Show that for a random variable $X$ with $\mathbb{E}[X]=0$ and $X \in [a, b]$, the moment generating function satisfies $\mathbb{E}[e^{\lambda X}] \le e^{\lambda^2 (b-a)^2 / 8}$ (Hoeffding's Lemma).
2.  Use the Chernoff bound $P(S_n \ge t) \le e^{-\lambda t} \prod \mathbb{E}[e^{\lambda X_i}]$ and optimize for $\lambda$ to complete the proof of Hoeffding's inequality.

### Exercise 2: The Sauer-Shelah Lemma
Let $\mathcal{H}$ be a hypothesis class with VC-dimension $d$. The growth function $m_{\mathcal{H}}(n)$ is the maximum number of ways $n$ points can be shattered by $\mathcal{H}$.

1.  Prove that for $n \le d$, $m_{\mathcal{H}}(n) = 2^n$.
2.  Prove the Sauer-Shelah Lemma by induction: $m_{\mathcal{H}}(n) \le \sum_{i=0}^d \binom{n}{i}$.
3.  Show that for $n > d$, $\sum_{i=0}^d \binom{n}{i} \le (en/d)^d$.

### Exercise 3: Stability of Ridge Regression
Consider Ridge Regression $w = (X^T X + n\lambda I)^{-1} X^T y$.

1.  Prove that the algorithm is $\beta$-uniformly stable with $\beta = O(1/\lambda n)$.
2.  How does the stability change as $\lambda \to 0$ in the over-parameterized regime ($d > n$)?

### Exercise 4: Massart's Finite Class Lemma
Let $\mathcal{A} \subset \mathbb{R}^n$ be a finite set of vectors with $|\mathcal{A}| = N$. Let $R = \max_{v \in \mathcal{A}} \|v\|_2$.
Show that $\mathbb{E}_\sigma [\sup_{v \in \mathcal{A}} \sum_{i=1}^n \sigma_i v_i] \le R\sqrt{2 \ln N}$.
*Hint*: Use the Chernoff method and the fact that $\mathbb{E}[e^{\lambda \sum \sigma_i v_i}] \le e^{\lambda^2 \|v\|^2 / 2}$.

### Exercise 5: Rademacher Complexity of MLPs
Consider a 2-layer ReLU network $f(x) = \sum_{j=1}^H a_j \text{ReLU}(w_j^\top x)$ with constraint $\|a\|_1 \le B_1$ and $\|w_j\|_2 \le B_2$.
Prove that $\hat{\mathcal{R}}_S(\mathcal{H}) \le \frac{2 B_1 B_2 L}{\sqrt{n}}$.
*Hint*: Use the contraction property of Rademacher complexity: $\hat{\mathcal{R}}_S(\phi \circ \mathcal{H}) \le 2\hat{\mathcal{R}}_S(\mathcal{H})$ for 1-Lipschitz $\phi$.

### Exercise 6: VC-Dimension of Linear Classifiers
Prove that the VC-dimension of linear classifiers in $\mathbb{R}^d$ is $d+1$.

1.  Provide a set of $d+1$ points that can be shattered.
2.  Use Radon's Theorem to show that no set of $d+2$ points can be shattered.

### Exercise 7: PAC-Bayes Change of Measure
Prove the fundamental identity (Donsker-Varadhan representation):

$$
\text{KL}(Q \parallel P) = \sup_{f} \left\{ \mathbb{E}_{Q}[f] - \log \mathbb{E}_{P}[e^{f}] \right\}
$$

*Hint*: Use the definition of KL divergence and Jensen's inequality (or Fenchel duality).

### Exercise 8: Benign Overfitting in Linear Regression
Consider $y = w^\top x + \epsilon$. In the regime $d \gg n$, the model interpolates the data.

1.  Under what conditions on the eigenvalues of the covariance matrix $\Sigma = \mathbb{E}[xx^\top]$ is the generalization error $R(\hat{w}) \to \sigma_\epsilon^2$ as $n, d \to \infty$?
2.  Discuss the "Effective Rank" of $\Sigma$.

---

## 3. Coding Practice

### Task 1: Visualizing Double Descent
Train a series of linear models with increasing polynomial degrees (or a random feature model with increasing width) on a noisy sine wave.

- **Dataset**: 20 points sampled from $y = \sin(x) + \epsilon$.
- **Model**: Linear regression with $d$ random features.
- **Plot**: Training and Test error vs. $d$. Identify the "interpolation threshold" where $d=n$.

### Task 2: Estimating Rademacher Complexity
Implement a function to estimate $\hat{\mathcal{R}}_S(\mathcal{H})$ for a small CNN on CIFAR-10.

- **Logic**: Train the model to fit random labels. The resulting training accuracy (appropriately scaled) is an empirical estimate of the complexity.
- **Experiment**: How does the complexity change if you add Dropout or Weight Decay?

### Task 3: PAC-Bayes Bound Calculation
For a simple Logistic Regression model, calculate the McAllester bound.

- **Setup**: Use $\mathcal{N}(0, I)$ as the prior. After training, use the weights as the mean for a posterior $\mathcal{N}(w, \sigma^2 I)$.
- **Analysis**: Plot the value of the bound $\epsilon$ as a function of the training sample size $n$. Is the bound non-vacuous for $n=1000$?

---

## 4. Solutions and Hints

### Solution 4 (Exercise 4)
For any $\lambda > 0$, $e^{\lambda \mathbb{E}[\sup Z]} \le \mathbb{E}[e^{\lambda \sup Z}] \le \sum_{v \in \mathcal{A}} \mathbb{E}[e^{\lambda \sigma \cdot v}] \le N e^{\lambda^2 R^2 / 2}$.
Taking log and dividing by $\lambda$: $\mathbb{E}[\sup Z] \le \frac{\ln N}{\lambda} + \frac{\lambda R^2}{2}$.
Minimizing over $\lambda$ gives $\lambda = \frac{\sqrt{2 \ln N}}{R}$, yielding the result $R\sqrt{2 \ln N}$.

### Hint 5 (Exercise 5)
$\hat{\mathcal{R}}_S(\mathcal{H}) = \mathbb{E}_\sigma [ \sup \sum_{i=1}^n \sigma_i \sum_{j=1}^H a_j \text{ReLU}(w_j^\top x_i) ] = \mathbb{E}_\sigma [ \sup \sum_{j=1}^H a_j (\sum_{i=1}^n \sigma_i \text{ReLU}(w_j^\top x_i)) ]$.
By $\|a\|_1 \le B_1$, this is $\le B_1 \mathbb{E}_\sigma [ \sup_{w: \|w\| \le B_2} \sum_{i=1}^n \sigma_i \text{ReLU}(w^\top x_i) ]$.
Apply the contraction lemma and the linear class complexity result.

---

## 5. Open Research Questions

1. **Implicit Regularization in Transformers**: Can we find a complexity measure that explains why Transformers generalize despite having billions of parameters and being trained to zero loss?
2. **Generalization of SAM**: Does Sharpness-Aware Minimization lead to a lower Rademacher complexity, or is its benefit purely explained by PAC-Bayes bounds?
3. **Out-of-Distribution (OOD)**: Can SLT be extended to the setting where the test distribution is unknown but related to the training distribution via a "latent symmetry"?
