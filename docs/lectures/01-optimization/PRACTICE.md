# Topic 01: Optimization Theory — Practice and Problem Set

## 1. Detailed Example: Newton's Method on a Non-Convex Surface

Consider the **Rosenbrock Function**, a classic non-convex test case for optimization:

$$
f(x, y) = (1 - x)^2 + 100(y - x^2)^2
$$

The global minimum is at $(1, 1)$ where $f(1, 1) = 0$.

### 1.1. Gradient and Hessian Calculation
The gradient $\nabla f(x, y)$ is:

$$
\frac{\partial f}{\partial x} = -2(1 - x) - 400x(y - x^2)
$$

$$
\frac{\partial f}{\partial y} = 200(y - x^2)
$$

The Hessian $\mathbf{H}(x, y)$ is:

$$
\mathbf{H} = \begin{pmatrix} 2 - 400y + 1200x^2 & -400x \\ -400x & 200 \end{pmatrix}
$$

### 1.2. Newton Step from $(-1, 2)$

1. **Evaluate Gradient**: $\nabla f(-1, 2) = (-2(2) - 400(-1)(2 - 1), 200(2 - 1))^\top = (396, 200)^\top$.
2. **Evaluate Hessian**: $\mathbf{H}(-1, 2) = \begin{pmatrix} 2 - 800 + 1200 & 400 \\ 400 & 200 \end{pmatrix} = \begin{pmatrix} 402 & 400 \\ 400 & 200 \end{pmatrix}$.
3. **Invert Hessian**: $\det(\mathbf{H}) = 402(200) - 400^2 = 80400 - 160000 = -79600$.

$$
\mathbf{H}^{-1} = \frac{1}{-79600} \begin{pmatrix} 200 & -400 \\ -400 & 402 \end{pmatrix}
$$

4. **Compute Update**: $\Delta w = -\mathbf{H}^{-1} \nabla f \approx (0.001, -0.998)^\top$.
   Note that because the Hessian is not positive definite at this point ($\det < 0$), the Newton step might move towards a saddle point or maximum unless damped (Levenberg-Marquardt).

---

## 2. Problem Set (Theoretical)

### Exercise 1: Strong Convexity and PL Condition
Prove that if a function $f$ is $\mu$-strongly convex, it satisfies the PL condition with parameter $\mu$.
*Hint*: Use the definition $f(y) \ge f(x) + \nabla f(x)^\top (y-x) + \frac{\mu}{2}\|y-x\|^2$ and minimize both sides with respect to $y$.

### Exercise 2: Mirror Descent and KL Divergence
Show that for the potential $\psi(x) = \sum x_i \log x_i$ on the simplex $\Delta_d$, the Bregman divergence $D_\psi(x, y)$ is exactly the KL Divergence $D_{KL}(x \| y)$.

### Exercise 3: Convergence of SGD
Consider $f(w) = \frac{1}{2} w^2$. Suppose we use SGD with $w_{t+1} = w_t - \eta (w_t + \xi_t)$ where $\xi_t \sim \mathcal{N}(0, \sigma^2)$.
Find the variance of $w_t$ as $t \to \infty$ for a constant $\eta$.

### Exercise 4: K-FAC Approximation
Explain why the approximation $\mathbb{E}[A \otimes G] \approx \mathbb{E}[A] \otimes \mathbb{E}[G]$ is used in K-FAC. What independence assumption does this imply between forward activations and backward gradients?

### Exercise 5: Linear Convergence under SVRG
Briefly derive why SVRG can use a constant learning rate while SGD must use a decaying one to converge to the exact optimum.

### Exercise 6: Nesterov Accelerated Gradient (NAG)
Consider the NAG updates:

$$
y_{t+1} = x_t - \eta \nabla f(x_t)
$$

1.

Show that for a quadratic $f(x) = \frac{1}{2} x^\top \mathbf{A} x$, this is equivalent to a second-order linear recurrence. What is the optimal $\gamma$ for a condition number $\kappa = L/\mu$?

### Exercise 7: Gradient Flow and Neural Tangent Kernel (NTK)
In the limit of $\eta \to 0$, GD becomes a differential equation: $\dot{w}(t) = -\nabla f(w(t))$.
If $f(w) = \frac{1}{2}\|y - \Phi(w)\|^2$, show that the evolution of predictions $\hat{y}(t) = \Phi(w(t))$ follows:

$$
\dot{\hat{y}}(t) = -\mathbf{K}(t)(\hat{y}(t) - y)
$$

where $\mathbf{K}(t) = \nabla \Phi(w(t)) \nabla \Phi(w(t))^\top$.

### Exercise 8: Implicit Bias of GD
For a linear model $y = w^\top x$ on separable data, show that GD with a small learning rate converges to the **Max-Margin (SVM)** solution. Use the fact that the direction $w/\|w\|$ must eventually align with the gradient of the loss at the support vectors.

---

## 3. Coding Practice

### Task 1: Implementing a Custom Line Search
Implement the **Backtracking Line Search** (Armijo Condition) for a non-convex function.

- **Goal**: Find $\eta$ such that $f(x - \eta \nabla f(x)) \le f(x) - c \eta \|\nabla f(x)\|^2$.
- **Function to test**: $f(x, y) = x^4 + y^4 - 4xy + 1$.

### Task 2: Visualizing the "Edge of Stability"
Write a script to monitor the maximum eigenvalue of the Hessian during GD on a 2-layer ReLU network.

- **Dataset**: A small subset of MNIST (e.g., digits 0 and 1).
- **Observation**: Increase the learning rate $\eta$ until the sharpness $\lambda_{max}$ hovers just above $2/\eta$. Plot the loss and $\lambda_{max}$ over iterations.

### Task 3: Natural Gradient vs. GD
Implement NGD for a simple Logistic Regression model.

- **Hint**: The Fisher Information Matrix for Logistic Regression is $\mathbf{F} = \mathbf{X}^\top \mathbf{S} \mathbf{X}$, where $\mathbf{S}$ is a diagonal matrix of variances $p_i(1-p_i)$.
- **Comparison**: Compare the number of iterations to converge for GD and NGD on the "Breast Cancer" dataset from `sklearn`.

---

## 4. Solutions and Hints

### Hint 1 (Exercise 1)
Minimizing $g(y) = f(x) + \nabla f(x)^\top (y-x) + \frac{\mu}{2}\|y-x\|^2$ gives $y^* = x - \frac{1}{\mu}\nabla f(x)$.
Substituting $y^*$ back: $f(y^*) \ge f(x) - \frac{1}{2\mu}\|\nabla f(x)\|^2$.
Since $f^* \le f(y^*)$, we have $f^* \ge f(x) - \frac{1}{2\mu}\|\nabla f(x)\|^2$.

### Solution 3 (Exercise 3)
The update is $w_{t+1} = (1-\eta)w_t - \eta \xi_t$.
Squaring and taking expectation: $\mathbb{E}[w_{t+1}^2] = (1-\eta)^2 \mathbb{E}[w_t^2] + \eta^2 \sigma^2$.
As $t \to \infty$, $\mathbb{E}[w^2] = (1-\eta)^2 \mathbb{E}[w^2] + \eta^2 \sigma^2$.
Solving for $\mathbb{E}[w^2]$: $\text{Var}(w) = \frac{\eta \sigma^2}{2-\eta} \approx \frac{\eta \sigma^2}{2}$.

### Hint 7 (Exercise 7)
Use the chain rule: $\dot{\hat{y}} = \frac{\partial \Phi}{\partial w} \dot{w}$. Substitute $\dot{w} = -\nabla_w f = -\nabla_w \Phi(w)^\top \nabla_{\hat{y}} f$.
Since $\nabla_{\hat{y}} f = (\hat{y} - y)$, the result follows.

---

## 5. Open Research Questions

1. **The Edge of Stability (EoS)**: Why do neural networks often train in a regime where the sharpness (max eigenvalue of Hessian) is slightly larger than $2/\eta$, contradicting classical stability theory?
2. **Implicit Bias of Adaptive Methods**: While GD on separable data leads to the max-margin solution, what is the specific geometric bias of Adam or Lion? Does it favor $L_\infty$ margins?
3. **Flatness and Generalization**: Is the "flatness" of a minimum a causal driver of generalization, or simply a correlated artifact of normalization and architecture?
