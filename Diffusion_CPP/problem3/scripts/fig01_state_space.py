"""
fig01_state_space.py
=====================
Figure 1: the (p, xi) state space of the 2D runaway-electron model.

THIS IS AN EXPLANATORY FIGURE, NOT A SIMULATION.  It exists to orient the
reader in the domain before any dynamics are shown: what the two coordinates
mean, which walls absorb a particle (removing it from the simulation) and
which walls reflect it, and roughly what a trajectory approaching each kind
of wall looks like.  The four illustrative curves below are hand-built
smooth/random paths (numpy cumulative noise, lightly filtered) -- NOT output
of common_2.simulate_re2d_step.  Figure 3/4 will show the real, physics-true
trajectories once the SDE itself has been introduced.

Physics being illustrated
--------------------------
State:  X = (p, xi)
  p  = electron momentum, normalized by m_e c (so p is dimensionless).
  xi = cos(pitch angle) = cos(theta), i.e. the cosine of the angle between
       the electron's momentum and the local magnetic field. xi=+1 means the
       particle moves exactly along the field; xi=-1 exactly against it.

Domain and boundary conditions (see common_2.py docstring, Section on
"Domain"):
  p in [p_min, p_max]:
    p = p_min  -- ABSORBING.  Falling below p_min means collisional drag has
                  overwhelmed the accelerating electric field; the electron
                  rejoins the thermal bulk distribution.  This is the
                  "thermalization" exit.
    p = p_max  -- ABSORBING.  Crossing p_max is treated as a confirmed
                  runaway electron (the state of interest in the avalanche
                  literature). This is the "runaway" exit.
  xi in [-1, 1]:
    xi = +-1   -- REFLECTING.  Physically this is just the pitch angle
                  passing through 0 or pi; there is nothing singular about
                  the state itself, only the (p, xi) parametrization. The
                  pitch-angle diffusion coefficient s_xi ~ sqrt(1 - xi^2)
                  vanishes at +-1, so the stochastic term switches off
                  exactly at the wall -- consistent with a reflecting
                  boundary in the coarse integrator.

Four illustrative trajectory types are sketched:
  1. Thermalized  (blue)   -- net drift toward low p, exits at p_min.
  2. Runaway      (red)    -- net drift toward high p, exits at p_max.
  3. Reflecting bounce (green) -- wanders in xi, bounces off xi=+1 once,
                                  and survives inside the domain.
  4. Knock-on jump (magenta) -- mostly smooth like the others, but with one
                                  abrupt, discontinuous jump in p (dashed
                                  segment) representing a single large-angle
                                  collision, then continues smoothly.

Output: problem3/figures/fig01_state_space.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import Config

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def smooth_path(x0, n, drift, wander, seed, clip=None):
    """A schematic smooth path: cumulative Gaussian noise with a constant
    drift, lightly box-filtered so it reads as a continuous curve rather
    than a jagged random walk. Purely illustrative (see module docstring).
    """
    rng = np.random.default_rng(seed)
    steps = drift + wander * rng.standard_normal(n)
    raw = x0 + np.cumsum(steps)
    kernel = np.ones(5) / 5.0
    padded = np.concatenate([np.full(4, raw[0]), raw])
    sm = np.convolve(padded, kernel, mode="valid")
    if clip is not None:
        sm = np.clip(sm, clip[0], clip[1])
    return sm[: n]


def make_thermalized_path(p_min, p_max, n=140):
    """Starts mid-domain, net drag-dominated drift toward p_min, wanders a
    bit in xi, and is cut off the first time it reaches p_min."""
    p = smooth_path(5.5, n, drift=-0.045, wander=0.10, seed=1)
    xi = smooth_path(0.1, n, drift=0.0, wander=0.02, seed=2, clip=(-1, 1))
    hit = np.where(p <= p_min)[0]
    if len(hit):
        k = hit[0] + 1
        p, xi = p[:k], xi[:k]
        p[-1] = p_min
    return p, xi


def make_runaway_path(p_min, p_max, n=140):
    """Starts mid-domain, net field-dominated drift toward p_max (the
    electric field wins over drag once the particle is fast enough), cut
    off at p_max."""
    p = smooth_path(6.0, n, drift=0.05, wander=0.10, seed=3)
    xi = smooth_path(0.55, n, drift=0.0015, wander=0.015, seed=4, clip=(-1, 1))
    hit = np.where(p >= p_max)[0]
    if len(hit):
        k = hit[0] + 1
        p, xi = p[:k], xi[:k]
        p[-1] = p_max
    return p, xi


def make_reflecting_path(p_min, p_max, n=110):
    """Wanders in xi with no strong p drift, bounces off xi=+1 exactly once
    (mirror reflection), and survives to the end of the illustration window
    without reaching either p wall."""
    p = smooth_path(3.2, n, drift=0.006, wander=0.05, seed=5)
    p = np.clip(p, p_min + 0.3, p_max - 0.3)
    xi_raw = smooth_path(0.55, n, drift=0.011, wander=0.03, seed=6)
    xi = xi_raw.copy()
    over = xi > 1.0
    if over.any():
        first = np.where(over)[0][0]
        xi[first - 1] = 1.0  # touch the wall exactly, no overshoot pre-mirror
        # mirror everything from the crossing back into [-1, 1]
        xi[first:] = 2.0 - xi[first:]
        xi = np.clip(xi, -1.0, 1.0)
    return p, xi, (np.where(over)[0][0] - 1 if over.any() else None)


def make_jump_path(p_min, p_max, n=130, jump_at=55, jump_size=2.6):
    """A smooth path like the others, but with one instantaneous jump in p
    inserted at index `jump_at`: this is the schematic stand-in for a single
    knock-on (large-angle Moller) collision, which moves p by an O(1) amount
    in zero time -- a discontinuity, not a diffusive step."""
    p = smooth_path(4.0, n, drift=0.012, wander=0.08, seed=7)
    xi = smooth_path(-0.2, n, drift=0.0, wander=0.02, seed=8, clip=(-1, 1))
    p[jump_at:] -= jump_size  # the knock-on: primary loses energy, p drops
    p = np.clip(p, p_min + 0.15, p_max - 0.15)
    return p, xi, jump_at


def main():
    cfg = Config()
    p_min, p_max = cfg.p_min, cfg.p_max

    fig, ax = plt.subplots(figsize=(8.0, 6.4))

    # ---- exterior (absorbed) regions, shaded ----
    pad = 0.6 * (p_max - p_min) / 9.0 + 0.4
    ax.axvspan(p_min - pad, p_min, color="#4c72b0", alpha=0.12, zorder=0)
    ax.axvspan(p_max, p_max + pad, color="#c44e52", alpha=0.12, zorder=0)
    ax.set_xlim(p_min - pad, p_max + pad)
    ax.set_ylim(-1.25, 1.25)

    # ---- domain box ----
    ax.axhspan(-1, 1, xmin=0, xmax=1, facecolor="none", edgecolor="none")
    ax.plot([p_min, p_max], [1, 1], color="0.25", lw=1.0, zorder=1)
    ax.plot([p_min, p_max], [-1, -1], color="0.25", lw=1.0, zorder=1)

    # ---- absorbing boundaries: p_min, p_max ----
    ax.axvline(p_min, color="#4c72b0", lw=3.0, zorder=2)
    ax.axvline(p_max, color="#c44e52", lw=3.0, zorder=2)
    for (x0, dx, col) in [(p_min, -1, "#4c72b0"), (p_max, 1, "#c44e52")]:
        ax.annotate("", xy=(x0 + dx * 1.0, 0.0), xytext=(x0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
    ax.text(p_min - pad * 0.55, 0.15, "ABSORBING\n(thermalized,\n$p<p_{\\min}$)",
            color="#4c72b0", fontsize=9.5, ha="center", va="bottom",
            fontweight="bold")
    ax.text(p_max + pad * 0.55, 0.15, "ABSORBING\n(runaway,\n$p>p_{\\max}$)",
            color="#c44e52", fontsize=9.5, ha="center", va="bottom",
            fontweight="bold")

    # ---- reflecting boundaries: xi = +-1 ----
    ax.axhline(1.0, color="#55a868", lw=2.2, ls="--", zorder=2)
    ax.axhline(-1.0, color="#55a868", lw=2.2, ls="--", zorder=2)
    ax.text(p_min + 0.5 * (p_max - p_min), 1.135, "REFLECTING  ($\\xi=+1$)",
            color="#55a868", fontsize=9.5, ha="center", fontweight="bold")
    ax.text(p_min + 0.5 * (p_max - p_min), -1.21, "REFLECTING  ($\\xi=-1$)",
            color="#55a868", fontsize=9.5, ha="center", fontweight="bold")

    # ---- illustrative trajectories ----
    p_t, xi_t = make_thermalized_path(p_min, p_max)
    ax.plot(p_t, xi_t, color="#4c72b0", lw=1.8, zorder=3)
    ax.plot(p_t[0], xi_t[0], "o", color="0.2", ms=5, zorder=4)
    ax.plot(p_t[-1], xi_t[-1], "X", color="#4c72b0", ms=9, zorder=4)

    p_r, xi_r = make_runaway_path(p_min, p_max)
    ax.plot(p_r, xi_r, color="#c44e52", lw=1.8, zorder=3)
    ax.plot(p_r[0], xi_r[0], "o", color="0.2", ms=5, zorder=4)
    ax.plot(p_r[-1], xi_r[-1], "X", color="#c44e52", ms=9, zorder=4)

    p_b, xi_b, bounce_idx = make_reflecting_path(p_min, p_max)
    ax.plot(p_b, xi_b, color="#55a868", lw=1.8, zorder=3)
    ax.plot(p_b[0], xi_b[0], "o", color="0.2", ms=5, zorder=4)
    ax.plot(p_b[-1], xi_b[-1], "*", color="#55a868", ms=12, zorder=4)
    if bounce_idx is not None:
        ax.plot(p_b[bounce_idx], 1.0, "d", color="#55a868", ms=7, zorder=5)

    p_j, xi_j, jidx = make_jump_path(p_min, p_max)
    ax.plot(p_j[:jidx + 1], xi_j[:jidx + 1], color="#8172b2", lw=1.8, zorder=3)
    ax.plot(p_j[jidx:], xi_j[jidx:], color="#8172b2", lw=1.8, zorder=3)
    ax.plot([p_j[jidx - 1], p_j[jidx]], [xi_j[jidx - 1], xi_j[jidx]],
            color="#8172b2", lw=1.4, ls=":", zorder=3)
    ax.plot(p_j[0], xi_j[0], "o", color="0.2", ms=5, zorder=4)
    ax.plot(p_j[-1], xi_j[-1], "*", color="#8172b2", ms=12, zorder=4)
    ax.plot(p_j[jidx], xi_j[jidx], "P", color="#8172b2", ms=10, zorder=5)

    # ---- legend (proxy artists) ----
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#4c72b0", lw=2, marker="X",
               label="thermalized exit ($p \\to p_{\\min}$)"),
        Line2D([0], [0], color="#c44e52", lw=2, marker="X",
               label="runaway exit ($p \\to p_{\\max}$)"),
        Line2D([0], [0], color="#55a868", lw=2, marker="*",
               label="reflecting bounce, survives"),
        Line2D([0], [0], color="#8172b2", lw=2, marker="P",
               label="knock-on jump, survives"),
        Line2D([0], [0], color="0.2", lw=0, marker="o", label="start"),
    ]
    ax.legend(handles=legend_elems, loc="lower center",
              bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=9, frameon=False)

    ax.set_xlabel(r"$p$  (momentum, units of $m_e c$)", fontsize=11)
    ax.set_ylabel(r"$\xi = \cos\theta$  (pitch-angle cosine)", fontsize=11)
    ax.set_title("Figure 1 — State space of the 2D runaway-electron model\n"
                 "(schematic: domain, boundary conditions, illustrative "
                 "trajectory types)", fontsize=11.5)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(FIG_DIR, "fig01_state_space.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"domain: p in [{p_min}, {p_max}], xi in [-1, 1]")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
