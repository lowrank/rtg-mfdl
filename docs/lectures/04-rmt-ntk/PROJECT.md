# Topic 04 Project: NTK Kernel Regression vs. Finite-Width Networks

## 1. Objective
Compare the "Lazy Training" limit (NTK) with the "Feature Learning" regime of finite-width neural networks. You will investigate how width affects the stability of the NTK and the final performance on a real-world classification task.

## 2. Dataset
Kaggle: **Natural Language Processing with Disaster Tweets**

- [Link to Dataset](https://www.kaggle.com/c/nlp-getting-started)
- **Task**: Classify if a tweet is about a real disaster or not.

## 3. Detailed Step-by-Step Implementation

### Phase 1: Data Preparation

1.  **Text Processing**: Clean the tweets (remove URLs, special characters).
2.  **Embeddings**: Use a pre-trained sentence transformer (e.g., `sentence-transformers/all-MiniLM-L6-v2`) to convert each tweet into a 384-dimensional vector.
3.  **Dimensionality Reduction**: Apply PCA to reduce the embedding to $d=128$ dimensions.
4.  **Normalization**: Ensure all input vectors are unit norm: $x_i \leftarrow x_i / \|x_i\|$.

### Phase 2: Analytical NTK Regression

1.  **Kernel Construction**: Implement the analytical NTK for a 2-layer ReLU network:


$$
\Theta(x, z) = \frac{\|x\| \|z\|}{2\pi} \left[ (\sin \phi + (\pi - \phi) \cos \phi) + \cos \phi (\pi - \phi) \right]
$$


where $\cos \phi = x \cdot z / (\|x\| \|z\|)$.

2.  **Kernel Ridge Regression**:
- Compute the $N \times N$ kernel matrix $K$ for the training set.
- Solve $\alpha = (K + \lambda I)^{-1} y$.
- Predict on the test set: $\hat{y} = K_{test} \alpha$.
3.  **Hyperparameter Tuning**: Tune $\lambda$ using cross-validation.

### Phase 3: Finite-Width MLP Training

1.  **Architecture**: Build a 2-layer MLP with hidden width $m$.
2.  **Parameterization**: Use **NTK Parameterization**:
    - Initialize $W^{(1)} \sim \mathcal{N}(0, 1)$, $b^{(1)} = 0$, $W^{(2)} \sim \mathcal{N}(0, 1)$.
    - Forward pass: $h = \sigma(\frac{1}{\sqrt{d}} W^{(1)} x)$, $y = \frac{1}{\sqrt{m}} W^{(2)} h$.
3.  **Experiments**: Train models with widths $m \in \{16, 64, 256, 1024, 4096\}$.
4.  **Kernel Tracking**: At each epoch, compute the empirical NTK:

    $$
    \hat{\Theta}(t) = \nabla_\theta f(X; \theta(t)) \nabla_\theta f(X; \theta(t))^\top
    $$


### Phase 4: Comparative Analysis

1.  Measure the **Frobenius distance** between the empirical NTK at $t=0$ and $t=T$: $\Delta\Theta = \|\hat{\Theta}_T - \hat{\Theta}_0\|_F / \|\hat{\Theta}_0\|_F$.
2.  Compare the test accuracy of the Kernel Regression vs. the finite-width MLPs.

---

## 4. Expected Results and Analysis

### Expected Observations

1.  **Consistency**: As width $m$ increases, the test accuracy of the MLP should converge to the accuracy of the Analytical NTK.
2.  **Lazy Training**: For large $m$ (e.g., 4096), the relative change in the kernel $\Delta\Theta$ should be very small ($< 5\%$).
3.  **Small-Width Advantage?**: You might find that small-width networks (e.g., $m=64$) actually outperform the NTK if they are trained with a higher learning rate, as they can "learn features" that the fixed NTK cannot.

### Analysis Questions

- Does the "disaster tweet" dataset benefit from feature learning, or is it simple enough that the fixed NTK is sufficient?
- How does the dimensionality of the input ($d$) affect the speed of convergence to the NTK limit?

---

## 5. Kaggle Tips

- Use the `nlp-getting-started` Kaggle competition for automated evaluation.
- Since the dataset is small (~7,600 samples), the $N \times N$ kernel matrix (~60MB) will easily fit in memory.
- Use `jax` or `neural-tangents` library if you want to compute the empirical NTK efficiently.
