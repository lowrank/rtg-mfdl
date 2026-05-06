# Topic 05 Project: VAEs, Rate-Distortion, and MI Estimation

## 1. Objective
Deeply investigate the information-theoretic properties of generative models. You will build a $\beta$-VAE and analyze its performance through the lens of Rate-Distortion Theory, then use MINE to probe the internal representations.

## 2. Dataset
Kaggle: **Dogs vs. Cats** or **Fashion-MNIST**

- [Dogs vs. Cats Link](https://www.kaggle.com/c/dogs-vs-cats)
- [Fashion-MNIST Link](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

## 3. Detailed Step-by-Step Implementation

### Phase 1: The $\beta$-VAE Framework

1.  **Architecture**: Implement a Variational Autoencoder with a latent dimension $z \in \mathbb{R}^{32}$.
2.  **Loss Function**: $\mathcal{L} = \text{MSE}(x, \hat{x}) + \beta D_{KL}(q(z|x) \| p(z))$.
3.  **The Sweep**: Train the VAE for $\beta \in \{0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$.
4.  **Metric Collection**: After training, calculate the average Reconstruction Error ($D$) and KL-Divergence ($R$) on the test set.

### Phase 2: Rate-Distortion Analysis

1.  **The R-D Curve**: Plot $R$ vs. $D$ for the different values of $\beta$.
2.  **Theoretical Comparison**: For a Gaussian source with the same variance as your dataset, calculate the theoretical $R(D)$ curve. How close is your neural compressor to the theoretical limit?
3.  **Visual Proof**: Display reconstructions and latent space interpolations for $\beta=0.1$ (high rate) vs. $\beta=10.0$ (low rate).

### Phase 3: Probing with MINE

1.  **Task**: Train a separate MLP classifier on the VAE latents $z$ to predict the class label (e.g., Cat vs. Dog).
2.  **MI Estimation**: Implement MINE to estimate the Mutual Information $I(x; z)$ and $I(z; y)$ where $y$ is the class label.
3.  **Comparison**: Does increasing $\beta$ (more compression) lead to a decrease in $I(z; y)$? Is there a "sweet spot" for $\beta$ that maximizes $I(z; y) / I(x; z)$?

---

## 4. Expected Results and Analysis

### Expected Observations

1.  **R-D Trade-off**: As $\beta$ increases, $R$ will decrease and $D$ will increase. The curve should be roughly convex.
2.  **Disentanglement**: At higher $\beta$, you should see more "meaningful" latent variables (e.g., one dimension might control the ear shape, another the fur color).
3.  **MI Ceiling**: MINE estimates might plateau or become unstable for high MI values. You may need to use a large batch size (~1024) for reliable estimation.

### Analysis Questions

- Does the VAE act as a "sufficient statistic" for the class labels?
- How does the "Rate" (KL term) relate to the number of bits actually used to encode the image?
- Can you identify any "redundant" latent dimensions that have $D_{KL} \approx 0$?

---

## 5. Kaggle Tips and Resources

- For Dogs vs. Cats, resize images to $64 \times 64$ to speed up VAE training.
- Use `torchvision` transformations for data augmentation.
- Implement the "reparameterization trick" carefully to ensure gradients flow correctly through the sampler.

