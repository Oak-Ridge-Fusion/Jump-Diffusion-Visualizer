# 06 — Rollout and the Three Ablations

## The rollout algorithm

Given trained network $G(x, z)$ and starting position $x_0$:

```
For each particle i = 1, ..., N:
    x = x_0
    For each step n = 0, ..., N_STEPS - 1:
        z ~ N(0, 1)            # independent fresh noise
        ΔX = G(x, z)           # neural network forward pass
        x ← x + ΔX
        if x ≤ 0 or x ≥ 6:
            particle absorbed; break
    if particle survived:
        record x as survivor
```

## Why sample fresh z at every step?

The Brownian increment $dW_t$ is **independent** across time steps. By sampling independent $z$ at each step we replicate this independence.

If we reused the same $z$ across steps, the particle would have correlated increments — not Brownian motion.

## The three training conditions (Figure 2 ablations)

All three conditions use the SAME:
- Network architecture (FlowNet, 200k parameters)
- Training procedure (MSE, Adam, 100 epochs)
- Rollout procedure
- Boundary check at rollout

The ONLY difference is **which (x, ΔX) pairs are used for training**:

### Condition 1: Our Method ✓

**Include**: transitions where particle was **inside at both the start and end** of the step.

**Effect**: Near $x = 0$, only steps with $\Delta X > -x$ are included (particle survived). The training distribution is right-biased near the left wall.

**Result**: The network learns the boundary correction → distribution matches ground truth (approximately).

### Condition 2: All Trajectories Trained ✗

**Include**: ALL transitions, including those where the particle **crossed the boundary** during the step.

**Effect**: Near $x = 0$, we now include steps with $\Delta X = -5$ (particle exited). The distribution is now symmetric near the wall.

**Result**: The network learns the uncorrected, symmetric distribution → doesn't respect the boundary → different distribution at T.

### Condition 3: Only Confined Trained ✗

**Include**: ONLY transitions from particles that **survived all the way to T=3**.

**Effect**: These are specially-selected particles that never wandered near either wall. Their step distributions are sampled from the "interior" regime, not the boundary regime.

**Result**: The network never sees near-boundary transitions → can't handle particles that approach the walls → distribution is too narrow/concentrated.

## Mathematical interpretation

### Condition 1 (Our Method)

Training pairs are $(x_t, \Delta X)$ where both $x_t$ and $x_t + \Delta X$ are in $(0, L)$.

The training distribution is: $p(\Delta X \mid x_t, \text{survived step})$

At rollout: $G(x, z)$ samples from this conditioned distribution. Particles near walls get boundary-corrected increments → correct long-run behaviour.

### Condition 2 (All Trajectories)

Training pairs include all $(x_t, \Delta X)$ regardless of absorption.

The training distribution is: $p(\Delta X \mid x_t)$ = the **unconditional** step distribution.

For Brownian motion: $\Delta X \mid x_t \sim N(0, dt)$ for ALL $x_t$ (boundary effect only appears when conditioning on survival).

At rollout: $G(x, z)$ samples from the symmetric Gaussian → no boundary correction → many particles drift into the walls immediately → survivors have a different distribution.

### Condition 3 (Only Confined)

Training pairs are from the **long-surviving** subset.

Mathematically, this is sampling from the **Doob h-transform** of the process:

$$p^h(\Delta X \mid x_t) = \frac{h(x_t + \Delta X)}{h(x_t)} p(\Delta X \mid x_t)$$

where $h(x) = \sin(\pi x / L)$ is the ground state of the killed Brownian motion.

The h-transform SUPPRESSES movements toward the boundary and ENHANCES movements toward the centre — exactly opposite to what we want at individual steps.

## Why does the choice of training data matter so much?

The three conditions highlight a fundamental principle:

> **A generative model is only as good as its training distribution.**

The network $G(x, z)$ is a universal function approximator. It will fit whatever distribution you show it. If you show it:
- Clean (surviving) transitions → it learns the boundary effect
- All transitions → it learns to ignore the boundary
- Survivor-selected transitions → it learns to avoid boundaries (too aggressively)

This is why **data curation is as important as model architecture**.
