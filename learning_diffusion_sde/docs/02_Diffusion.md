# 02 — The Diffusion Model (Forward Process)

## What is a diffusion model?

A diffusion model is a generative model that learns to reverse a **noise-adding process**. It has two phases:

```
FORWARD  (data → noise):   Z_0  →  Z_{0.25}  →  Z_{0.5}  →  Z_{0.75}  →  Z_1 ≈ N(0,1)
REVERSE  (noise → data):   Z_1  →  Z_{0.75}  →  Z_{0.5}  →  Z_{0.25}  →  Z_0 ≈ ΔX
```

## What is Z_0?

In our pipeline, $Z_0 = \Delta X$ is the **step increment** from the SDE training data.

The diffusion model learns to:
1. Gradually add noise to $\Delta X$ until it looks like standard Gaussian noise
2. Reverse this process to generate new samples of $\Delta X$

## The VP-SDE noise schedule

We use the **Variance-Preserving (VP)** schedule from Song et al. (2021):

$$Z_t = \alpha(t) \cdot Z_0 + \sigma(t) \cdot \varepsilon, \quad \varepsilon \sim N(0,1)$$

where:

$$\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$$

$$\log \alpha(t) = -\frac{1}{2}\left(\beta_{\min} t + \frac{\beta_{\max} - \beta_{\min}}{2} t^2\right)$$

$$\alpha(t) = e^{\log\alpha(t)}, \quad \sigma(t) = \sqrt{1 - \alpha(t)^2}$$

### Key values at extreme times:

| $t$ | $\alpha(t)$ | $\sigma(t)$ | Interpretation |
|---|---|---|---|
| $0$ | $1$ | $0$ | Pure data: $Z_0 = \Delta X$ |
| $0.5$ | $0.28$ | $0.96$ | Mostly noise |
| $1$ | $0.007$ | $\approx 1$ | Pure noise: $Z_1 \approx N(0,1)$ |

### Why "variance-preserving"?

$$\text{Var}[Z_t] = \alpha^2(t) \cdot \text{Var}[Z_0] + \sigma^2(t) \approx 1 \quad \forall t$$

(approximately, since $\text{Var}[Z_0] \ll 1$ for our narrow $\Delta X$ distribution)

This keeps $Z_t$ in the same scale for all $t$, which is numerically stable.

## The score function

The **score** at diffusion time $t$ is:

$$s(z, t) = \nabla_z \log p_t(z)$$

where $p_t$ is the marginal density of $Z_t$.

For the forward process: $p_t(z | z_0) = N(z; \alpha(t) z_0, \sigma^2(t))$

The conditional score is:
$$\nabla_z \log p_t(z | z_0) = -\frac{z - \alpha(t) z_0}{\sigma^2(t)} = -\frac{\varepsilon}{\sigma(t)}$$

The **unconditional score** (integrating over the data distribution) is what we need for sampling — estimated by KNN in script 04.

## Why not just sample from a Gaussian?

For particles **in the interior** ($x$ far from both walls), $\Delta X \approx N(0, dt)$ and we could sample directly.

Near the **boundaries**, the distribution is **truncated**:
- Near $x = 0$: large negative $\Delta X$ would push the particle outside → absorbed, not included
- Near $x = 6$: large positive $\Delta X$ similarly removed

The diffusion model learns this truncated, asymmetric distribution implicitly through the score function.

## Connection to the KDE score

The unconditional score of a KDE with Gaussian kernel $K_\sigma$ is:

$$\nabla_z \log \hat{p}(z) = \frac{\sum_i K_\sigma(z - z_i)(z_i - z)/\sigma^2}{\sum_i K_\sigma(z - z_i)} \approx \frac{\mu_{\text{KNN}}(z) - z}{\text{var}_{\text{KNN}}(z)}$$

This is exactly the KNN estimator used in script 04.
