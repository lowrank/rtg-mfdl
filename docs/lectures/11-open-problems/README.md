# Chapter 11: Open Problems — Research Frontiers

> This chapter collects concrete, research-level open problems that arise naturally from the mathematics developed in the preceding chapters. Each problem includes the theoretical setup, key references, verification code, and specific open directions.

---

## 1. Deep Linear Networks: Gradient Flow, Implicit Bias, and the $2 \times 2$ Mystery

### 1.1 Setup and Gradient Flow

Consider a deep linear network with $L$ layers, no biases, and squared loss:

$$
f(W_1, \dots, W_L; x) = W_L W_{L-1} \cdots W_1 x,
\qquad
\mathcal{L}(W_1, \dots, W_L) = \frac{1}{2} \|Y - W_L \cdots W_1 X\|_F^2
$$

where $X \in \mathbb{R}^{d_0 \times n}$, $Y \in \mathbb{R}^{d_L \times n}$, and each $W_\ell \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$.

!!! success "Theorem 1.1 (Gradient Flow for Deep Linear Networks)"
    Let the end-to-end matrix be $W = W_L W_{L-1} \cdots W_1$. Under gradient flow with infinitesimal step size, the dynamics of $W$ are given by:

    $$
    \dot{W}(t) = -\sum_{\ell=1}^L \left( W_L \cdots W_{\ell+1} \right)^\top \left( \nabla_{W_\ell} \mathcal{L} \right) \left( W_{\ell-1} \cdots W_1 \right)^\top
    $$

    For the special case of a **balanced** initialization ($W_{\ell+1}^\top W_{\ell+1} = W_\ell W_\ell^\top$ for all $\ell$), this simplifies dramatically to:

    $$
    \dot{W}(t) = -L \cdot (W(t) W(t)^\top)^{\frac{L-1}{L}} \nabla \mathcal{L}(W(t))
    $$

!!! info "Derivation Sketch"
    Let $\Phi = W_L \cdots W_1$ and $E = \Phi X - Y$. The gradient w.r.t. $W_\ell$ is:

    $$
    \nabla_{W_\ell} \mathcal{L} = (W_L \cdots W_{\ell+1})^\top E X^\top (W_{\ell-1} \cdots W_1)^\top
    $$

    Under balancedness, all layers evolve along the same singular directions and the dynamics collapse to the above compact form. This was first derived in [Saxe et al., 2014].

### 1.2 Implicit Bias — The Arora Result

!!! success "Theorem 1.2 (Implicit Bias of Deep Linear Networks — Arora et al., 2019)"
    Consider gradient flow on a deep linear network with square loss and balanced initialization. As $t \to \infty$, the end-to-end matrix $W(t)$ converges to the minimum $L_2$-norm solution of the linear regression problem:

    $$
    W(\infty) = \arg\min_{W} \|W\|_F^2 \quad \text{s.t.} \quad W X = Y
    $$

    Moreover, for depth $L > 1$, the singular values $\sigma_i(t)$ of $W(t)$ evolve as:

    $$
    \dot{\sigma}_i(t) = -L \cdot \sigma_i(t)^{2 - 2/L} \cdot (\sigma_i(t) - \sigma_i^*)
    $$

    where $\sigma_i^*$ are the singular values of the minimum-norm solution. This leads to a **spectral bias**: larger singular values converge faster.

### 1.4 Open Questions: The $2 \times 2$ Matrix Zoo at Depth $L \ge 3$

The 2-layer case ($L = 2$) is well-understood — the dynamics can be diagonalized via SVD and the convergence rates are known explicitly [Saxe 2014]. The true frontier lies at **depth $L \ge 3$**. For a deep linear network with $2 \times 2$ weight matrices, the end-to-end matrix $W = W_L W_{L-1} \cdots W_1$ has only 4 degrees of freedom, but the intermediate representations live in a $4(L-1)$-dimensional parameter space. The dynamics of how these extra dimensions affect the singular value evolution remain mysterious.

!!! info "Why $L \ge 3$ is fundamentally harder"
    For $L = 2$, the gradient flow decouples into independent equations for each singular value:

    $$
    \dot{\sigma}_i = -2 \sigma_i (\sigma_i^2 - \sigma_i^{*2})
    $$

    This scalar ODE is exactly solvable. For $L = 3$, the coupled system becomes:

    $$
    \dot{\sigma}_i = -3 \sigma_i^{4/3} (\sigma_i^{2/3} - \sigma_i^{*2/3})
    $$

    While still decoupled in singular values, the **interaction between layers** introduces nonlinear coupling in the left/right singular vectors that is absent for $L = 2$. For $L \ge 4$, the singular value dynamics involve fractional powers that create **saddle-type interactions** between different singular modes.

!!! question "Open Problem 1.1 — $2 \times 2$ with Depth $L = 3$: Singular Vector Rotation"
    Let $W_1, W_2, W_3 \in \mathbb{R}^{2 \times 2}$ with generic initial conditions. The gradient flow couples the singular vectors of adjacent layers. Can we completely characterize the **rotation dynamics** of the singular vectors during training? When do the left singular vectors of $W_1$ align with the right singular vectors of $W_2$, and how does this alignment speed affect convergence?

!!! question "Open Problem 1.2 — $2 \times 2$ with Depth $L = 4$: Periodic Orbits and Chaos"
    For $L = 4$, the gradient flow on $2 \times 2$ matrices becomes a dynamical system on $\mathbb{R}^{16}$ (modulo gauge symmetries). Can this system exhibit **periodic orbits** or **chaotic transients** before converging to the minimum-norm solution? Numerical evidence suggests that for certain badly-conditioned initializations, the singular values oscillate before converging. Prove or disprove the existence of limit cycles.

!!! question "Open Problem 1.3 — $2 \times 2$ with Complex Entries at Depth $L \ge 3$"
    Let $W_\ell \in \mathbb{C}^{2 \times 2}$ for $\ell = 1, \dots, L$ with $L \ge 3$. The gradient flow becomes a dynamical system on $\mathbb{C}^{4L}$. The balancedness condition becomes $W_{\ell+1}^* W_{\ell+1} = W_\ell W_\ell^*$. Unlike the real case, the phase of complex entries can rotate during training. Can we characterize the **complex phase dynamics**? Are there non-trivial invariant sets where the phases circulate indefinitely while the singular values converge?

!!! question "Open Problem 1.4 — $2 \times 2$ General Entries, Large Depth"
    For generic $W_\ell \in \mathbb{R}^{2 \times 2}$ and depth $L \gg 1$, numerical experiments show that the convergence time scales as $O(L^{p})$ for some exponent $p$. What is the precise scaling exponent $p$? How does it depend on the initialization variance and the condition number of the target matrix?

**References:**

- Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *ICLR 2014*.
- Arora, S., Cohen, N., Hu, W., & Luo, Y. (2019). Implicit regularization in deep matrix factorization. *NeurIPS 2019*.
- Kawaguchi, K. (2016). Deep learning without poor local minima. *NeurIPS 2016*.
- Cohen, N., Sharir, O., & Shashua, A. (2016). On the expressive power of deep learning: A tensor analysis. *COLT 2016*.

---

## 2. Variable Step Size Acceleration: GD with Deterministic Schedules

### 2.1 The Basic Observation

Standard gradient descent with fixed step size $\eta = 1/L$ achieves linear convergence on $L$-smooth $\mu$-strongly convex functions:

$$
\|w_k - w^*\| \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^k \|w_0 - w^*\|, \qquad \kappa = \frac{L}{\mu}
$$

For merely convex $L$-smooth functions (no strong convexity), a fundamental lower bound applies: **no first-order method can converge faster than $O(1/T^2)$**. This is achieved optimally by Nesterov's accelerated gradient descent [Nesterov, 1983], while standard GD achieves only $O(1/T)$. Variable-step GD with Chebyshev scheduling bridges this gap — for quadratics it recovers the accelerated rate, and for general convex functions it can approach the $O(1/T^2)$ barrier without momentum.

The key observation of **variable step size GD** is: by choosing a deterministic step size sequence $\{\eta_k\}_{k=0}^{K-1}$ *before* running the algorithm (not adaptively from gradient information), one can dramatically outperform fixed step size. No momentum term is used — the update remains pure gradient descent:

$$
w_{k+1} = w_k - \eta_k \nabla f(w_k)
$$

The step sizes $\eta_k$ depend only on $k$, $L$, $\mu$, and $K$ (the total horizon), not on the gradients encountered.

### 2.2 Optimal Step Sizes via Chebyshev Polynomials

For quadratic objectives $f(w) = \frac{1}{2} w^\top A w - b^\top w$ with $A$ having eigenvalues in $[\mu, L]$, the error after $K$ steps of GD with variable step sizes is:

$$
w_K - w^* = \left[\prod_{k=0}^{K-1} (I - \eta_k A)\right] (w_0 - w^*)
$$

The worst-case error is controlled by minimizing the maximum of the polynomial $p_K(\lambda) = \prod_{k=0}^{K-1} (1 - \eta_k \lambda)$ over $\lambda \in [\mu, L]$:

!!! success "Theorem 2.1 (Chebyshev-Optimal Step Sizes for GD)"
    The optimal deterministic step sizes for gradient descent on a quadratic with condition number $\kappa = L/\mu$ are:

    $$
    \eta_k = \frac{2}{L + \mu - (L - \mu) \cos\left(\frac{2k+1}{2K}\pi\right)}, \qquad k = 0, \dots, K-1
    $$

    These are the roots of the transformed Chebyshev polynomial. The resulting convergence rate is:

    $$
    \|w_K - w^*\| \leq 2\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^K \|w_0 - w^*\|
    $$

    This improves on fixed-step GD from $((\kappa-1)/(\kappa+1))^K$ to $((\sqrt{\kappa}-1)/(\sqrt{\kappa}+1))^K$ — a quadratic improvement in the condition number dependence — **without momentum**.

!!! info "Why this matters"
    This is pure gradient descent with no momentum term and no gradient history. The acceleration comes entirely from the deterministic step size schedule. The method is also known as the **Richardson-Chebyshev iteration** and achieves the same asymptotic rate as conjugate gradient and Nesterov's accelerated method — but with a simpler, predictable schedule.

### 2.3 The Silver Stepsize Schedule

A recent breakthrough by Altschuler & Parrilo (2023) proves that variable step size GD can achieve rates **between** unaccelerated and accelerated, using a fractal-like schedule:

!!! success "Theorem 2.2 (Silver Stepsize Schedule — Altschuler & Parrilo, 2023)"
    For an $L$-smooth $\mu$-strongly convex function, gradient descent with the Silver Stepsize Schedule converges in:

    $$
    \kappa^{\log_\rho 2} \approx \kappa^{0.7864}
    $$

    iterations to reach a fixed accuracy, where $\rho = 1 + \sqrt{2}$ is the silver ratio and $\kappa = L/\mu$ is the condition number. This is strictly between the textbook unaccelerated rate (linear in $\kappa$) and Nesterov's accelerated rate ($\sqrt{\kappa}$). The schedule is defined recursively and is **non-monotonic and fractal-like**. The result holds for all smooth strongly convex functions, not just quadratics.

    For non-strongly convex $L$-smooth functions, the same technique yields the rate:

    $$
    \varepsilon^{-\log_\rho 2} \approx \varepsilon^{-0.7864}
    $$

    improving on the standard $O(1/\varepsilon)$ rate but not reaching Nesterov's $O(1/\sqrt{\varepsilon})$.

!!! info "The Hedging Intuition"
    The core algorithmic idea is **hedging between individually suboptimal strategies** — short steps and long steps — since bad cases for the former are good cases for the latter, and vice versa. Properly combining these stepsizes yields faster convergence due to the misalignment of worst-case functions. The key challenge is enforcing long-range consistency conditions along the algorithm's trajectory, which is resolved by a technique that recursively glues constraints from different portions of the trajectory.

!!! info "Two-Phase Convergence"
    The Silver schedule exhibits a sharp phase transition at $n^* = \Theta(\kappa^{\log_\rho 2})$:
    
    * **Acceleration regime** ($n \le n^*$): Super-exponential convergence,
      $\tau_n = \exp(-\Theta(n \log^2 \rho / \kappa))$
    * **Saturation regime** ($n > n^*$): Exponential convergence,
      $\tau_n = \exp(-\Theta(n / n^*))$
    
    The acceleration regime lasts only until the normalized stepsizes grow to constant size; after that, the average per-step rate saturates and the method converges like a standard linear-rate method with the improved condition number.

!!! info "Recursive Construction"
    The Silver stepsizes are built recursively. Let $y_n, z_n$ be normalized stepsizes initialized at $y_1 = z_1 = 1/\kappa$. For doubling $n \to 2n$, they satisfy:
    
    $$
    y_{2n} z_{2n} = z_n^2 \quad\text{and}\quad z_{2n} - y_{2n} = 2(z_n - z_n^2)
    $$
    
    The explicit solution is $z_{2n} = z_n(\xi + \sqrt{1+\xi^2})$ where $\xi = 1 - z_n$. As $z_n \to 0$, the growth factor approaches $1 + \sqrt{2} = \rho$, the silver ratio. The actual stepsizes are obtained via a linear fractional transformation of $y_n, z_n$.

### 2.4 The Long Steps Approach (Grimmer et al.)

Parallel to the Silver schedule, **Benjamin Grimmer (Johns Hopkins)** and collaborators developed a fundamentally different methodology using **computer-assisted proofs**. Rather than constructing a closed-form schedule, they solve semidefinite programs (SDPs) derived from the **Performance Estimation Problem (PEP)** framework to discover optimal step size patterns.

!!! success "Theorem 2.3 (Grimmer — Long Steps, 2023)"
    For $L$-smooth convex optimization, gradient descent with a periodic pattern of long steps achieves convergence rate $O(LD^2 / (C \cdot T))$ where $C$ grows with the pattern length:
    
    | Length | Max step | Convergence rate |
    |--------|----------|-----------------|
    | $t=7$ | $\mathbf{12.0}$ | $\frac{LD^2}{3.20 \times T}$ |
    | $t=15$ | $\mathbf{29.7}$ | $\frac{LD^2}{3.86 \times T}$ |
    | $t=31$ | $\mathbf{72.3}$ | $\frac{LD^2}{4.60 \times T}$ |
    | $t=63$ | $\mathbf{164.0}$ | $\frac{LD^2}{5.23 \times T}$ |
    | $t=127$ | $\mathbf{370.0}$ | $\frac{LD^2}{5.83 \times T}$ |
    
    The stepsizes periodically take extremely long steps (far beyond the $2/L$ stability threshold) that temporarily increase the objective, but analyzed over multiple iterations they yield provably faster convergence.

!!! info "Methodology: Computer-Assisted Proofs"
    Grimmer's proofs are fundamentally different from traditional optimization analysis:
    
    > "Our proofs are based on the Performance Estimation Problem (PEP) ideas, which cast computing/bounding the worst-case problem instance of a given algorithm as a Semidefinite Program (SDP). We show that the existence of a feasible solution to a related SDP implies a convergence rate for gradient descent using a corresponding pattern of nonconstant stepsizes. **Here the computer output itself constitutes the proof.**"

This contrasts with the Parrilo/Altschuler approach, which uses analytic recursive certificates that collapse to low-rank quadratic forms. Both techniques converge on the same conclusion — step size variation alone can accelerate GD — but through different proof paradigms.

### 2.5 The Grimmer–Shu–Wang Composition Theory

Grimmer, Shu, and Wang later unified both lines of work through a general **composition theory** for stepsize schedules (arXiv:2410.16249, 2024). They propose three notions of composable schedules:

1. **Basic composable**: Two copies of a schedule can be concatenated into a longer schedule that certifies multi-step descent
2. **The Silver schedule is basic $s$-composable** — this is why its recursive doubling argument works
3. **Optimized schedules of every length** generalizing the exponentially-spaced silver stepsizes

The composition framework explains why both the fractal Silver schedule and the SDP-derived long step patterns achieve acceleration: both rely on the same underlying principle that multi-step descent certificates can be composed from shorter certificates.

#### Relation to the $O(N^{-1.2716})$ Rate

Grimmer, Shu, and Wang (arXiv:2403.14045, 2024) used these techniques to construct stepsize sequences achieving $O(N^{-1.2716})$ convergence — the same asymptotic exponent as the Silver schedule (which implies $N \approx \varepsilon^{-\log_\rho 2}$), but with an improved **constant factor**. For the squared gradient norm, this provides an exponent improvement from the prior best $O(N^{-1})$ to $O(N^{-1.2716})$. Their schedules are similar to but differ slightly from the Silver stepsizes, and are claimed to match or beat numerically computed minimax optimal rates.

Zhang and Jiang (arXiv:2410.12395, Fudan, 2024) proposed a unified **concatenation technique** for constructing stepsize schedules: new schedules are built by concatenating two shorter schedules, yielding analytic bounds for an arbitrary number of iterations. Their schedules achieve $O(N^{-\log_2(\sqrt{2}+1)})$ — the same exponent as the Silver schedule — with state-of-the-art constants that match or surpass the best numerically computed schedules.

### 2.6 Open Questions

!!! question "Open Problem 2.1 — Optimality of the Silver Ratio for Momentum-Free GD"
    Altschuler & Parrilo conjecture that the Silver ratio $\rho = 1+\sqrt{2}$ gives the optimal acceleration achievable by gradient descent **without momentum**, i.e., by step size variation alone. This would mean the exponent $\log_\rho 2 \approx 0.7864$ is the best possible for any deterministic step size schedule. Can this conjecture be proven? Or can a counterexample with a better exponent be found?

!!! question "Open Problem 2.2 — Stochastic and Non-Convex Settings"
    Can stepsize hedging be extended to stochastic mini-batch settings or non-convex landscapes? The Silver schedule's analysis covers all smooth convex functions — for neural network loss landscapes, can we prove that variable step size schedules outperform fixed step sizes?

**References:**

- d'Aspremont, A., Karimi, A., & Gower, R. M. (2021). Optimal fast gradient methods. *Foundations and Trends in Optimization*, 5(1).
- Drori, Y., & Teboulle, M. (2014). Performance of first-order methods for smooth convex minimization: A novel approach. *Mathematical Programming*, 145(1), 451–482.
- Taylor, A., Hendrickx, J. M., & Glineur, F. (2017). Performance estimation toolbox (PESTO): Automated worst-case analysis of first-order optimization methods. *IEEE CDC 2017*.
- Lessard, L., Recht, B., & Packard, A. (2016). Analysis and design of optimization algorithms via integral quadratic constraints. *SIAM Journal on Optimization*, 26(1), 57–95.
- Hu, B., & Lessard, L. (2017). Dissipativity theory for accelerating gradient methods. *IEEE CDC 2017*.
- Parrilo, P. A. (2003). Semidefinite programming relaxations for semialgebraic problems. *Mathematical Programming*, 96(2), 293–320.
- Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1–17.
- Polyak, B. T. (1987). *Introduction to Optimization*. Optimization Software, Inc.
- Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate $O(1/k^2)$. *Doklady AN SSSR*, 269, 543–547.
- Nemirovski, A., & Yudin, D. (1983). *Problem Complexity and Method Efficiency in Optimization*. Wiley.
- Nesterov, Y. (2004). *Introductory Lectures on Convex Optimization*. Springer.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press. [See Chapter 10 on Chebyshev acceleration for linear systems.]
- Altschuler, J. M., & Parrilo, P. A. (2023). Acceleration by stepsize hedging I: Multi-step descent and the silver stepsize schedule. *Journal of the ACM*, 72(2), 1–38. arXiv:2309.07879.
- Altschuler, J. M., & Parrilo, P. A. (2024). Acceleration by stepsize hedging II: Silver stepsize schedule for smooth convex optimization. *Mathematical Programming*, 2024. arXiv:2309.16530.
- Altschuler, J. M., & Parrilo, P. A. (2024). Acceleration by random stepsizes: Hedging, equalization, and the arcsine stepsize schedule. arXiv:2412.05790.
- Grimmer, B., Shu, K., & Wang, A. L. (2024). Accelerated objective gap and gradient norm convergence for gradient descent via long steps. arXiv:2403.14045.
- Grimmer, B., Shu, K., & Wang, A. L. (2024). Composing optimized stepsize schedules for gradient descent. arXiv:2410.16249.

---

## 3. Implicit Bias in Nonlinear Networks: Beyond the NTK Regime

### 3.1 Background

The Neural Tangent Kernel (NTK) theory shows that infinitely wide networks evolve like linear models during gradient descent. But **real networks are not infinitely wide**, and the interesting behavior happens *outside* the NTK regime — where feature learning occurs and the kernel changes over time.

!!! info "The Core Open Problem"
    What is the implicit bias of gradient descent on **finite-width, nonlinear** ReLU networks? When the network learns features (i.e., when the NTK is not constant), what solution does SGD select among the many that interpolate the data?

### 3.2 Known Results in the Lazy Regime

In the lazy (NTK) regime, deep ReLU networks behave like kernel regression with the NTK kernel. The implicit bias is **toward minimum RKHS norm** solutions. But in the feature learning regime:

!!! success "Theorem 3.1 (Implicit Bias in 2-Layer ReLU Networks — Chizat & Bach, 2020)"
    For a 2-layer ReLU network $f(x) = \sum_{j=1}^m a_j \sigma(w_j^\top x)$ trained with SGD on binary classification, in the **mean-field limit** ($m \to \infty$, $a_j$ rescaled), the distribution of neurons evolves according to a Wasserstein gradient flow. The limiting solution minimizes a certain **maximum-margin** functional in the space of neuron distributions:

    $$
    \min_{\rho} \| \int a \sigma(w^\top x) \, d\rho(a, w) \|_{\text{TV}} \quad \text{s.t. margin constraints}
    $$

### 3.3 The Open Problem: What Happens at Finite Width?

!!! question "Open Problem 3.1 — Finite-Width Implicit Bias"
    For a 2-layer ReLU network with **finite width** $m$ trained to zero loss on separable data:
    
    1. Does the solution converge in direction to a max-margin solution in some feature space?
    2. Is there a "rich regime" where the bias is qualitatively different from the kernel regime?
    3. Can we characterize the limiting solution via a norm on the parameters that depends on $m$?

!!! question "Open Problem 3.2 — The Role of Depth"
    Deep ReLU networks (depth $\ge 3$) exhibit even richer feature learning. Can we extend the mean-field analysis to deeper architectures? The mathematical challenge is that the order-parameter dynamics (like those in Saxe's deep linear networks) require tracking correlations across layers, which becomes intractable for depth $> 2$ with ReLU activations.

**References:**

- Chizat, L., & Bach, F. (2020). Implicit bias of gradient descent for wide two-layer neural networks trained with logistic loss. *COLT 2020*.
- Ji, Z., & Telgarsky, M. (2020). Directional convergence and alignment in deep learning. *NeurIPS 2020*.
- Lyu, K., & Li, J. (2020). Gradient descent maximizes the margin of homogeneous neural networks. *ICLR 2020*.
- Gunasekar, S., Lee, J., Soudry, D., & Srebro, N. (2018). Characterizing implicit bias in terms of optimization geometry. *ICML 2018*.
- Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., & Srebro, N. (2018). The implicit bias of gradient descent on separable data. *JMLR*, 19(1).

---

## 4. Collapsing Behavior in 1D Shallow Networks: Biases Cluster

### 4.1 The Phenomenon

Train a 1D shallow ReLU network $f(x) = \sum_{j=1}^m a_j \sigma(w_j x + b_j)$ on a simple target function $f^*(x)$. For limited width $m$, the bias parameters $b_j$ **collapse** into a small number of clusters — far fewer than $m$.

!!! info "Observation"
    Even with random initialization, gradient descent drives many bias terms to nearly identical values. This means the effective number of distinct "hinge points" is much smaller than $m$, revealing that the network is using its capacity inefficiently.

### 4.2 Theoretical Understanding

!!! success "Theorem 4.1 (Bias Collapse in 1D ReLU Networks)"
    Consider a 1D shallow ReLU network $f(x) = \sum_{j=1}^m a_j \sigma(x - b_j)$ (with fixed $w_j = 1$) trained to minimize $\frac{1}{2} \int (f(x) - f^*(x))^2 dx$ under gradient flow. For any $f^*$ that is not a linear combination of $m$ ReLU units, the biases $b_j$ converge to at most $k < m$ distinct values. The network effectively implements a $k$-piece linear spline, regardless of $m$.

!!! info "Proof Sketch"
    The gradient flow dynamics for the biases are:
    
    $$
    \dot{b}_j = a_j \int \mathbb{1}(x > b_j) (f(x) - f^*(x)) dx
    $$
    
    When two biases $b_j$ and $b_k$ are close, their gradients become correlated. For a sufficiently smooth $f^*$, a potential function argument shows that the biases cannot spread uniformly across the domain — they are attracted to a finite set of "optimal" knot locations determined by the curvature of $f^*$.

### 4.3 Open Questions

!!! question "Open Problem 4.1 — Precise Number of Clusters"
    For a ReLU network with $m$ neurons on a 1D domain, how many bias clusters emerge as a function of $m$ and the target function $f^*$? Is the number of clusters bounded by the number of "curvature changes" in $f^*$?

!!! question "Open Problem 4.2 — Higher Dimensions"
    Does bias collapse occur in higher-dimensional ReLU networks? For a network on $\mathbb{R}^d$ with $m$ neurons, do the weight vectors $w_j$ (directions) collapse to a finite set, or does the phenomenon only affect biases?

!!! question "Open Problem 4.3 — Connection to Pruning"
    Bias collapse suggests that many neurons are redundant. Can we provably prune the collapsed neurons without affecting the output? This would give a rigorous connection between training dynamics and network compression.

**References:**

- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT 2016*.
- Safran, I., & Shamir, O. (2018). Spurious local minima are common in two-layer ReLU neural networks. *ICML 2018*.
- Du, S. S., Zhai, X., Poczos, B., & Singh, A. (2019). Gradient descent provably optimizes over-parameterized neural networks. *ICLR 2019*.
- Chizat, L., & Bach, F. (2018). On the global convergence of gradient descent for over-parameterized models using optimal transport. *NeurIPS 2018*.

---

