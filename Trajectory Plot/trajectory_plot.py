import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters 
# -----------------------------

x_min = 0.0
x_max = 6.0

sigma = 0.7          # Brownian diffusion
lam = 1.0            # Jump rate
jump_mean = 0.0
jump_std = 0.7

T = 1.0
n_sub = 200
dt = T / n_sub
sqrt_dt = np.sqrt(dt)

N = 1000             # number of electrons

rng = np.random.default_rng(42)

# ---------------------------------
# Allocate trajectory array
# ---------------------------------

# rows = electrons
# columns = time

X = np.zeros((N, n_sub + 1))

alive = np.ones(N, dtype=bool)

# random initial positions

X[:, 0] = rng.uniform(x_min, x_max, size=N)

# ---------------------------------
# Euler-Maruyama simulation
# ---------------------------------

for k in range(1, n_sub + 1):

    # previous position
    X[:, k] = X[:, k - 1]

    # only update living particles
    idx = np.where(alive)[0]

    if len(idx) == 0:
        break

    # -------------------------
    # Brownian motion
    # -------------------------

    dW = rng.normal(
        0.0,
        sqrt_dt,
        size=len(idx)
    )

    X[idx, k] += sigma * dW

    # -------------------------
    # Poisson jumps
    # -------------------------

    jump_probability = 1.0 - np.exp(-lam * dt)

    jump_mask = rng.random(len(idx)) < jump_probability

    if jump_mask.any():

        jumps = rng.normal(
            jump_mean,
            jump_std,
            size=jump_mask.sum()
        )

        X[idx[jump_mask], k] += jumps

    # -------------------------
    # Absorbing boundaries
    # -------------------------

    exited = (
        (X[:, k] < x_min) |
        (X[:, k] > x_max)
    ) & alive

    alive[exited] = False

    # stop drawing after exit

    X[exited, k:] = np.nan

# ---------------------------------
# Time vector
# ---------------------------------

time = np.linspace(0, T, n_sub + 1)

# ---------------------------------
# Plot survivors only
# ---------------------------------

plt.figure(figsize=(10,6))

for i in range(N):

    if alive[i]:

        plt.plot(
            time,
            X[i],
            linewidth=1,
            alpha=0.5,
            color="blue"
        )

plt.axhline(x_min,color="red",linestyle="--")
plt.axhline(x_max,color="red",linestyle="--")

plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Surviving Electron Trajectories")

plt.show()