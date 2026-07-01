# 01 — The Stochastic Differential Equation

## What we are studying

We study a particle undergoing pure **Brownian motion** (Wiener process) confined to a bounded domain:

$$dX_t = dW_t, \quad X_0 = 1, \quad 0 < X_t < 6 \quad \text{(absorbing boundaries)}$$

- $dW_t$ is a **Wiener increment**: the change in a Brownian motion over time $dt$, distributed as $N(0, dt)$.
- The domain is $[0, 6]$. When a particle reaches either wall, it is **permanently removed** (absorbed).
- We start 200,000 particles at $X_0 = 1$ and evolve to $T = 3$.

## Why this SDE?

1. **Simplicity**: zero drift, unit diffusion. No parameters to tune.
2. **Non-trivial**: despite the simple equation, the absorbing boundaries make the **conditional distribution at T non-Gaussian**.
3. **Analytical solution**: the exact solution exists (Fourier series), so we can verify numerical methods.

## Euler-Maruyama discretisation

We cannot integrate $dW_t$ exactly, so we use the Euler-Maruyama scheme:

$$X_{n+1} = X_n + \sqrt{dt} \cdot Z_n, \quad Z_n \sim N(0,1)$$

This is **exact** for additive-noise SDEs (no discretisation error in the distribution — only floating-point rounding).

With $dt = 5 \times 10^{-4}$ and $T = 3$: **6000 steps**.

## Absorbing boundary treatment

After every Euler step, we check:

```
alive[i] = (X_i > 0) AND (X_i < 6)
```

Dead particles stay dead — they never contribute to future steps.

## Analytical solution

The survival PDF $p(x, T)$ satisfies the **heat equation** (Fokker-Planck for zero drift):

$$\frac{\partial p}{\partial t} = \frac{1}{2} \frac{\partial^2 p}{\partial x^2}, \quad p(0,t) = p(L,t) = 0, \quad p(x,0) = \delta(x - x_0)$$

Solved by separation of variables:

$$p(x, T) = \frac{2}{L} \sum_{n=1}^{\infty} \sin\!\left(\frac{n\pi x_0}{L}\right) \sin\!\left(\frac{n\pi x}{L}\right) \exp\!\left(-\frac{n^2 \pi^2 T}{2L^2}\right)$$

The **conditional** PDF (given survival to T) is $p(x,T) / P_{\text{surv}}$ where $P_{\text{surv}} = \int_0^L p(x,T)\,dx$.

## Key diagnostics to check

| Quantity | Expected | How to check |
|---|---|---|
| Survival rate | $\approx 43\%$ | Compare `len(survivors)/N_PARTICLES` to Fourier $P_{\text{surv}}$ |
| Distribution shape | Approximately bell-shaped, bimodal near walls | Histogram |
| Theory vs simulation | Exact overlap | Red curve on histogram |

## What the ground truth teaches us

The surviving particle distribution is:
- **Skewed left** (started at $x_0=1$, left wall is closer)
- **Truncated near both walls** (boundary absorption removes tails)
- **Not Gaussian** (a Gaussian would need unbounded support)

This non-Gaussianity is exactly what the generative model must capture.
