# 04 — The Probability-Flow ODE

## Two ways to reverse a diffusion

Given the forward SDE: $dZ = -\tfrac{1}{2}\beta(t) Z\, dt + \sqrt{\beta(t)}\, dW_t$

There are two ways to run it backwards:

### 1. Reverse-time SDE (stochastic)

$$dZ = \left[-\tfrac{1}{2}\beta Z - \beta \cdot \nabla_z \log p_t(Z)\right] dt + \sqrt{\beta}\, d\bar{W}_t$$

This is a stochastic process — different random paths each run.

### 2. Probability-Flow ODE (deterministic)

$$\frac{dZ}{dt} = -\tfrac{1}{2}\beta(t)\left[Z + \nabla_z \log p_t(Z)\right]$$

Both have the **same marginal distributions** $p_t$ at every $t$.

We use the ODE for its advantages:
- **Deterministic**: given $Z_1$, the ODE uniquely determines $Z_0$
- **Invertible**: the map $Z_1 \leftrightarrow Z_0$ is bijective
- **Fewer evaluations**: ODE solvers take larger adaptive steps

## Euler discretisation (reverse direction)

Moving from $t = 1$ to $t = 0$ with step $-\Delta t$:

$$Z_{t - \Delta t} = Z_t + \Delta t \cdot \frac{1}{2}\beta(t)\left[Z_t + s(Z_t, t)\right]$$

where $s(Z_t, t) = \nabla_z \log p_t(Z_t)$ is the score (estimated by KNN).

### Why is there no minus sign?

We are integrating **forward in $\tau = -t$** (i.e., $\tau$ increases as $t$ decreases). Writing $Z(\tau)$ with $\tau = 1 - t$:

$$\frac{dZ}{d\tau} = +\tfrac{1}{2}\beta(1-\tau)\left[Z + s\right]$$

So the step is **positive** drift in the direction of $(Z + \text{score})$.

## Why the ODE is stiff near t = 0

At $t \approx 0$: $\alpha(0) = 1$, $\sigma(0) = 0$, so $p_0 = $ data distribution.

For our narrow data distribution with $\sigma_0 = 0.022$:

$$s(z, 0) = -\frac{z}{\sigma_0^2} \approx -2000z$$

The drift at $t = 0$ is:

$$\frac{1}{2}\beta(0) (z + (-2000z)) = \frac{1}{2} \times 0.1 \times (-1999z) \approx -100z$$

A fixed step $\Delta t = 0.02$ gives: $Z_{\text{new}} = Z + 0.02 \times (-100Z) = -Z$.

This is **numerical instability** — the solution oscillates between positive and negative.

### Fix: Tweedie correction at $t_{\min}$

We stop the ODE at $t_{\min} = 0.1$ and use:

$$\hat{Z}_0 = \frac{Z_{t_{\min}} + \sigma^2(t_{\min}) \cdot s(Z_{t_{\min}}, t_{\min})}{\alpha(t_{\min})}$$

At $t = 0.1$: $\alpha = 0.947$, $\sigma = 0.322$, so $\sigma^2 = 0.104$.

This single formula gives the MMSE estimate of $Z_0$ from $Z_{t_{\min}}$ and is exact for Gaussian conditionals.

## ASCII diagram of the reverse ODE

```
t=1          t=0.75       t=0.5        t=0.1        t=0
 │            │            │            │            │
Z₁ ─────────►Z₀.₇₅ ───────►Z₀.₅ ───────►Z₀.₁ ──┐   Z₀
N(0,1)                                          │ Tweedie
                                                └───────► ΔX
         ←── Euler ODE steps ────────────────────►
```

## What stops the ODE from diverging in practice?

Two safeguards:
1. **Score clipping**: we clip $s$ to $[-5, 5]$ to prevent runaway steps
2. **Tweedie correction**: avoids integrating through the stiff $t \in [0, t_{\min}]$ region

Even with these safeguards, the ODE is an **approximation** when the score is estimated by KNN. A neural network score estimator (as used in the original paper) would give much better results.

## The training-free promise

The paper's title is "training-free conditional diffusion models." The word "training-free" refers specifically to not training a **score network** $s_\theta(z, t)$. Instead, the score is estimated by KNN from the data.

The neural network $G(x, z)$ trained in scripts 06-07 is NOT a score network — it is a fast approximator of the ODE output.
