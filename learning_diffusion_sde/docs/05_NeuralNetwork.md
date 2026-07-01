# 05 — The Neural Network G(x, z) → ΔX

## Why a neural network?

The ODE+KNN approach (scripts 04-05) is:
- **Correct**: produces samples from approximately the right distribution
- **Slow**: ~1 sample/second (30 KDTree queries per ODE step)
- **Memory-heavy**: requires the full reference dataset in RAM

For rollout we need **millions of increments** (200,000 particles × 6,000 steps = 1.2 billion evaluations). The ODE is impractical.

**Solution**: train a fast neural network $G(x, z) \to \Delta X$ that **imitates** the ODE mapping.

## Architecture: FlowNet

```
Input:  [x_t / 6.0,   z]         (2-dimensional)
           │              │
      Linear(2 → 256)
      LayerNorm(256)
      SiLU()
           │
      Linear(256 → 256)
      LayerNorm(256)
      SiLU()
           │
      Linear(256 → 256)
      LayerNorm(256)
      SiLU()
           │
      Linear(256 → 256)
      LayerNorm(256)
      SiLU()
           │
      Linear(256 → 1)
           │
Output: ΔX                        (1-dimensional)
```

**Design choices**:
- **LayerNorm** instead of BatchNorm: works at any batch size, no running statistics to accumulate
- **SiLU** (Swish) activation: $f(x) = x \cdot \sigma(x)$. Smoother than ReLU, slightly better at approximating smooth functions
- **No output activation**: $\Delta X$ can be any real number
- **4 hidden layers × 256 units** = 200,449 parameters

## What the network must learn

The target function is approximately:

$$G(x, z) \approx \mu(x) + \sigma(x) \cdot z$$

where $\mu(x) = E[\Delta X \mid x_t = x]$ and $\sigma(x) = \text{Std}[\Delta X \mid x_t = x]$.

For **interior** positions: $\mu(x) \approx 0$, $\sigma(x) \approx \sqrt{dt}$ (pure Brownian motion)

For **boundary-adjacent** positions: the function becomes **non-linear** because:
- Near $x = 0$: $\mu(x) > 0$ (positive bias, wall repels), $\sigma(x) < \sqrt{dt}$ (distribution truncated)
- Near $x = 6$: $\mu(x) < 0$ (negative bias)

The non-linearity near boundaries is small but critical for correct rollout behaviour.

## Training objective

$$\mathcal{L}(\theta) = \mathbb{E}_{(x, z, \Delta X) \sim \text{labels}}\left[(G_\theta(x, z) - \Delta X)^2\right]$$

**Why MSE and not NLL?**  
The labels are deterministic functions of $(x, z)$ — for each $(x, z)$ pair there is exactly one label $\Delta X$ (from the reparameterisation). MSE minimises the squared error to this deterministic target.

**Why not use the VP-SDE loss?**  
The standard DDPM loss trains a network to predict the noise $\varepsilon$ at each diffusion time $t$. Our approach is different: we generate $(x, z, \Delta X)$ label triples and train to memorise this mapping directly. This is simpler and sufficient for our 1-D problem.

## Training details

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 → 0.00001 (cosine schedule) |
| Batch size | 2048 |
| Epochs | 100 |
| Train/Val split | 80/20 |
| Gradient clip | max_norm = 1.0 |

**Result**: val_MSE = 0.000004, relative MSE = 0.75% (excellent)

## Interpreting the relative MSE

$$\text{Relative MSE} = \frac{\text{val MSE}}{\text{Var}[\Delta X]} = \frac{0.000004}{0.000502} \approx 0.0075$$

- Relative MSE < 0.01 means the network explains 99.25% of the variance in $\Delta X$
- The residual 0.75% is from the KNN approximation error in label generation
- This is excellent for the purpose of trajectory rollout

## Rollout procedure

At inference time:
```python
for step in range(N_STEPS):
    z ~ N(0, 1)                   # fresh noise
    ΔX = G(x_t, z)               # network forward pass (~1ms for 200k particles)
    x_new = x_t + ΔX
    if x_new outside [0, 6]:
        particle absorbed
    else:
        x_t = x_new
```

The network is called once per step on all alive particles (batch size up to 200k). On CPU this runs at ~5 steps/second for 200k particles.
