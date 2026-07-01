# 03 — K-Nearest Neighbours Score Estimation

## The core problem

To reverse the diffusion process we need:

$$s(z, t) = \nabla_z \log p_t(z)$$

This requires knowing the density $p_t$. Fitting a parametric density and differentiating it requires training a neural network — which is what standard diffusion models do.

**The paper's key insight**: we can skip neural network training for the score and use K-Nearest Neighbours instead.

## Derivation

Start from the **Kernel Density Estimate** with Gaussian kernel $K_\sigma$:

$$\hat{p}(z) = \frac{1}{M} \sum_{i=1}^{M} K_\sigma(z - z_i), \quad K_\sigma(u) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-u^2/(2\sigma^2)}$$

The score of the KDE is:

$$\nabla_z \log \hat{p}(z) = \frac{\sum_i K_\sigma(z - z_i) \cdot (z_i - z)/\sigma^2}{\sum_i K_\sigma(z - z_i)}$$

This is a **weighted mean** of $(z_i - z)$, with Gaussian weights centred at $z$.

**KNN approximation**: if we use only the $K$ nearest neighbours to $z$, the far points contribute negligibly (their Gaussian weight is tiny). So:

$$\nabla_z \log \hat{p}(z) \approx \frac{\mu_{\text{KNN}} - z}{\text{var}_{\text{KNN}}}$$

where $\mu_{\text{KNN}}$ and $\text{var}_{\text{KNN}}$ are the mean and variance of the $K$ nearest neighbours.

## Choosing K

| $K$ | Bias | Variance | Best for |
|---|---|---|---|
| Small (5-10) | Low | High | Dense regions, large samples |
| Large (50-100) | Higher | Low | Tails, small samples |

In 1-D with $N=50000$ samples, $K = 50$ is a good balance.

## Conditioning on x_t

The paper conditions on the SDE position $x$. So we need:

$$s(z, t \mid x) = \nabla_z \log p_t(z \mid x_t = x)$$

We estimate this by searching for KNN in the **joint space** $(x, Z_t)$ with appropriate normalisation.

### Normalisation

The joint space has two coordinates with very different scales:
- $x$: spans $[0, 6]$ with std $\approx 0.9$
- $Z_t$: spans $[-3, 3]$ at $t=0.5$ with std $\approx 0.96$

We normalise each coordinate by its **actual standard deviation at diffusion time $t$**:

```python
x_norm   = x_ref / x_ref.std()          # scale x
z_t_ref  = alpha(t) * dx_ref + sigma(t) * noise
z_norm   = z_t_ref / z_t_ref.std()      # scale z_t (changes with t!)
```

**Critical mistake to avoid**: using `dx.std()` (≈0.022) as the z-scale instead of `z_t.std()` (≈1 at $t=0.5$). This inflates z-coordinates by ×45 and destroys the KNN search.

## Score magnitude near t = 0

As $t \to 0$, the distribution $p_t(z)$ becomes very narrow (std $\to$ 0.022):

$$s(z, t \to 0) \approx -\frac{z}{\sigma_0^2} \approx -\frac{z}{0.022^2} = -\frac{z}{0.0005} \approx -2000z$$

This causes numerical instability in the ODE integrator. We stop the ODE at $t_{\min} = 0.1$ and apply the **Tweedie formula** instead.

## Tweedie MMSE estimator

For any $t$, the minimum mean-squared error estimator of $Z_0$ given $Z_t$ is:

$$E[Z_0 \mid Z_t] = \frac{Z_t + \sigma^2(t) \cdot s(Z_t, t)}{\alpha(t)}$$

This is the **Tweedie formula** (Efron 2011, rediscovered in diffusion models).

We use this as a one-shot denoising step at $t = t_{\min}$, avoiding the stiff ODE region.
