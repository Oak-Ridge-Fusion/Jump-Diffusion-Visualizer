import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================

L = 6.0                     # Domain [0,L]
T = 3.0                     # Final time
dt = 5e-4                   # Euler step
N_particles = 200000        # Number of particles
initial_position = 1.0

steps = int(T / dt)

np.random.seed(0)

# ============================================================
# Initialize particles
# ============================================================

x = np.ones(N_particles) * initial_position

# True = particle still inside domain
alive = np.ones(N_particles, dtype=bool)

# ============================================================
# Euler-Maruyama Simulation
# ============================================================

sqrt_dt = np.sqrt(dt)

for _ in range(steps):

    alive_index = np.where(alive)[0]

    if len(alive_index) == 0:
        break

    # Brownian increment
    dW = np.random.randn(len(alive_index)) * sqrt_dt

    x[alive_index] += dW

    # Absorbing boundaries
    escaped = (x[alive_index] < 0) | (x[alive_index] > L)

    alive[alive_index[escaped]] = False

# ============================================================
# Surviving particles
# ============================================================

survivors = x[alive]

print("Particles remaining:", len(survivors))
print("Confinement rate:", len(survivors)/N_particles)

# ============================================================
# Plot Histogram
# ============================================================

plt.figure(figsize=(6,5))

plt.hist(
    survivors,
    bins=40,
    range=(0,L),
    color='blue',
    edgecolor='black',
    alpha=0.8
)

plt.xlim(0,6)
plt.ylim(0,8000)

plt.xlabel("Position $X_T$")

plt.ylabel("Count")
plt.title("Ground Truth")







X


plt.grid(alpha=0.3)


plt.tight_layout()


plt.show()