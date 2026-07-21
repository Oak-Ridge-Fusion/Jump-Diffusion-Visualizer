"""
fig02_drift_field.py
=====================
Figure 2: the deterministic drift vector field (b_p, b_xi) of the 2D
runaway-electron model, computed exactly from common_2.re2d_coeffs -- this
is a real evaluation of the ground-truth physics, not a schematic.

Physics
-------
re2d_coeffs(p, xi, cfg) returns the drift components of

  dp  = [ E xi - Fdrag(p) - gamma p (1-xi^2)/tau_syn ] dt   + (jumps only)
  dxi = [ E (1-xi^2)/p + xi (1-xi^2)/(tau_syn gamma) - xi nu_c ] dt
        + sqrt( nu_c (1-xi^2) ) dW

Three competing physical effects set the direction of b_p:
  * E xi            -- electric-field acceleration along the field line;
                        pushes p up when xi>0 (moving with the field).
  * -Fdrag(p)        -- collisional drag, ~(1+p^2)/p^2 -> ~1/p^2 at large p
                        (never truly negligible, but weaker than the
                        constant-in-p field term once p is large enough).
  * -gamma p (1-xi^2)/tau_syn -- synchrotron radiation reaction; always a
                        loss term, strongest away from xi=+-1.

Note dp itself has NO Brownian term in this model -- b_p above is the WHOLE
p-dynamics apart from jumps. So the sign of b_p alone answers the central
runaway question at a given (p, xi): "does this electron accelerate toward
p_max (runaway) or decelerate toward p_min (thermalization) on average?"
The zero-crossing curve of b_p (its nullcline) is exactly a deterministic
approximation to the "runaway separatrix" p_c described in the note --
diffusion and jumps let real trajectories cross it stochastically, but the
nullcline is where the drift itself changes sign.

We also draw the b_xi=0 nullcline (pitch-angle equilibrium curve): where the
electric-field pitch-torque balances collisional pitch scattering.

Arrow convention: because p (range ~9) and xi (range 2) are different
physical quantities on very different scales, arrows are drawn as UNIT
vectors in range-normalized coordinates (equal on-screen length everywhere)
so direction is legible everywhere in the domain; the true drift magnitude
and sign of b_p is instead shown by arrow COLOR (diverging colormap) and by
the nullcline contour.

Output: problem3/figures/fig02_drift_field.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import Config, re2d_coeffs

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    cfg = Config()
    p_min, p_max = cfg.p_min, cfg.p_max

    # ---- evaluate the real physics on a grid ----
    n_p, n_xi = 26, 19
    p = np.linspace(p_min, p_max, n_p)
    xi = np.linspace(-1.0, 1.0, n_xi)
    P, XI = np.meshgrid(p, xi)                     # shape (n_xi, n_p)
    b_p, b_xi, s_xi = re2d_coeffs(P, XI, cfg)

    # ---- direction-only unit arrows in range-normalized coordinates ----
    # (so equal visual length everywhere despite p and xi living on very
    # different scales; see module docstring)
    range_p, range_xi = p_max - p_min, 2.0
    bhat_p, bhat_xi = b_p / range_p, b_xi / range_xi
    mag = np.sqrt(bhat_p ** 2 + bhat_xi ** 2) + 1e-300
    dp_grid, dxi_grid = p[1] - p[0], xi[1] - xi[0]
    arrow_len = 0.85          # fraction of one grid cell
    U = (bhat_p / mag) * dp_grid * arrow_len
    V = (bhat_xi / mag) * dxi_grid * arrow_len

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    vmax = float(np.percentile(np.abs(b_p), 98))
    q = ax.quiver(P, XI, U, V, b_p, cmap="RdBu_r", clim=(-vmax, vmax),
                  angles="xy", scale_units="xy", scale=1.0,
                  pivot="mid", width=0.0045)
    cb = fig.colorbar(q, ax=ax, pad=0.02)
    cb.set_label(r"$b_p$   ($<0$: drag/synchrotron win $\to$ thermalization;"
                 r"  $>0$: field wins $\to$ runaway)", fontsize=9.5)

    # ---- nullclines: where the deterministic drift vanishes ----
    ax.contour(P, XI, b_p, levels=[0.0], colors="black", linewidths=2.4,
               zorder=5)
    ax.contour(P, XI, b_xi, levels=[0.0], colors="0.35", linestyles="--",
               linewidths=1.6, zorder=4)

    legend_elems = [
        Line2D([0], [0], color="black", lw=2.4,
               label=r"$b_p=0$  (deterministic separatrix-like curve)"),
        Line2D([0], [0], color="0.35", lw=1.6, ls="--",
               label=r"$b_\xi=0$  (pitch-angle equilibrium curve)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9,
              framealpha=0.9)

    ax.set_xlim(p_min, p_max)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel(r"$p$  (momentum, units of $m_e c$)", fontsize=11)
    ax.set_ylabel(r"$\xi=\cos\theta$  (pitch-angle cosine)", fontsize=11)
    ax.set_title(
        "Figure 2 — Deterministic drift field $(b_p,\\,b_\\xi)$ from "
        "re2d_coeffs\n"
        f"$\\hat E={cfg.E_hat:g}$, $Z={cfg.Z:g}$, "
        f"$\\tau_{{\\rm syn}}={cfg.tau_syn:g}$   "
        "(arrow = direction only; color/contour = true $b_p$)",
        fontsize=11)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig02_drift_field.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"grid: {n_p} x {n_xi} over p in [{p_min},{p_max}], xi in [-1,1]")
    print(f"b_p   range: [{b_p.min():.4f}, {b_p.max():.4f}]")
    print(f"b_xi  range: [{b_xi.min():.4f}, {b_xi.max():.4f}]")
    print(f"s_xi  range: [{s_xi.min():.4f}, {s_xi.max():.4f}]  "
          f"(diffusion amplitude -- shown in Figure 3, not here)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
