# Topic 02: Project — Expressivity and Symbolic Recovery in High Dimensions

## 1. Project Objective
The goal of this project is to empirically validate the **Depth Efficiency** and **Symbolic Discovery** theories discussed in this module. You will implement a deep compositional network (Sawtooth) and a Kolmogorov-Arnold Network (KAN) to solve complex approximation tasks.

---

## 2. Part 1: The Sawtooth Challenge (Depth Efficiency)

### 2.1 Task Description
Implement the Telgarsky iterated sawtooth function $f_L(x)$ for $L=8$ (which has 256 segments).

1. **Target:** Generate data points from the true $f_8(x)$ function.
2. **Deep Model:** Construct a 16-layer ReLU network with width 3.
3. **Shallow Model:** Construct a 1-layer ReLU network with width 1000.
4. **Experiment:** Train both models to fit the 256-segment sawtooth. 

### 2.2 Analysis Requirements

- **Convergence Plot:** Show the training loss vs. epochs for both models.
- **Complexity Analysis:** Compare the final approximation visually. Does the shallow network "blur" the sharp peaks?
- **Parameter Efficiency:** Calculate the ratio of parameters to accuracy for both models.

---

## 3. Part 2: Symbolic Discovery with KANs

### 3.1 Task Description
Generate data from a hidden physical law: $f(x, y) = \exp(\sin(x) + y^2)$.

1. **Model:** Initialize a KAN with architecture `[2, 5, 1]`.
2. **Train:** Fit the KAN using B-splines.
3. **Symbolic Recovery:** Use the `suggest_symbolic()` tool in `pykan` (or your own implementation) to identify the underlying functions.

### 3.2 Analysis Requirements

- **Interpretability:** Plot the learned univariate functions on each edge. Do they look like $\sin(x)$, $y^2$, and $\exp(u)$?
- **Stability:** Perform "Grid Extension" (refining the spline knots) and observe the error decay. Does it follow the predicted $O(G^{-(k+1)})$ rate?

---

## 4. Part 3: Spectral Bias and High-Frequency Noise

### 4.1 Task Description
Train an MLP on a signal $y = \sin(x) + \sin(50x)$.

1. **Standard Training:** Observe how long it takes to learn each frequency.
2. **Fourier Features:** Use the Positional Encoding trick (mapping $x \to [\sin(Bx), \cos(Bx)]$) to accelerate the learning of the 50Hz signal.

### 4.2 Analysis Requirements

- **FFT of Residuals:** Compute the FFT of the error at epochs 10, 100, and 1000.
- **Bypassing Bias:** Quantify the speed-up achieved by Fourier Features in learning high-frequency components.

---

## 5. Submission Guidelines
Your final report should include:

1. **Code:** A Jupyter notebook or Python script containing all implementations.
2. **Proofs:** A short write-up (PDF) explaining why the deep network was expected to outperform the shallow one for Part 1.
3. **Visualizations:** High-quality plots for all three parts.
4. **Discussion:** Reflect on the trade-off between representational power (approximation) and optimization (how hard it was to train).
