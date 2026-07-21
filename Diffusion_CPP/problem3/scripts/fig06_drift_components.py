"""
fig06_drift_components.py
===========================
Figure 6: the momentum-drift b_p(p) at several fixed pitch angles xi, and
its decomposition into the three competing physical terms (common_2.py,
Sec. "Physics: drift and diffusion coefficients"):

  b_p(p, xi) = E_hat * xi            <- electric-field acceleration along
                                         the field line (constant in p;
                                         changes SIGN with xi)
             - drag(p)                <- collisional friction,
                                         drag = f(p) * (1+p^2)/p^2
                                         -> ~1/p^2 at large p (this 1/p^2
                                         falloff is WHY runaway exists: drag
                                         becomes negligible at high p while
                                         the field term does not)
             - gamma p (1-xi^2)/tau_syn  <- synchrotron radiation reaction
                                         (a loss term, vanishes at xi=+-1
                                         where there is no perpendicular
                                         momentum to radiate away)

Panel A: b_p(p) for several fixed xi, colored by xi. Each curve's zero
crossing (if any) is exactly a point on the Figure 2 separatrix at that xi
-- below the crossing thermalization wins, above it runaway wins. For
xi<=0 the field term is <=0, so nothing here ever accelerates the electron:
b_p<0 for all p (curve never crosses zero) and the electron can only
thermalize on average, regardless of drag.

Panel B: the three terms plotted SEPARATELY for one representative xi (near
the Figure 2 separatrix, where the competition is closest), showing how
their sum produces the total b_p(p) curve and its zero crossing p_c(xi).

Output: problem3/figures/fig06_drift_components.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import Config, gamma_of_p, lnL_reduction, re2d_coeffs

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def drift_terms(p, xi, cfg):
    """Re-derive the three additive pieces of b_p separately (re2d_coeffs
    only returns their sum). Same formulas as common_2.re2d_coeffs -- kept
    here, not in common_2.py, purely so this figure can plot the pieces."""
    gamma = gamma_of_p(p)
    fac = lnL_reduction(gamma, cfg)
    field = cfg.E_hat * xi * np.ones_like(p)
    drag = fac * (1.0 + p * p) / (p * p)
    if cfg.tau_syn > 0:
        synchrotron = -gamma * p * (1.0 - xi * xi) / cfg.tau_syn
    else:
        synchrotron = np.zeros_like(p)
    return field, -drag, synchrotron


def find_zero_crossing(p, y):
    """First sign change of y(p) (linear interpolation); None if none."""
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return p[i] - y[i] * (p[i + 1] - p[i]) / (y[i + 1] - y[i])


def main():
    cfg = Config()
    p = np.linspace(cfg.p_min, cfg.p_max, 400)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # ---- Panel A: b_p(p) at several fixed xi ----
    xi_values = [-0.5, 0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap("coolwarm")
    print("b_p(p) zero-crossings (deterministic separatrix points p_c(xi)):")
    for xi0 in xi_values:
        b_p, _, _ = re2d_coeffs(p, xi0 * np.ones_like(p), cfg)
        color = cmap(norm(xi0))
        axA.plot(p, b_p, color=color, lw=2.0, label=fr"$\xi={xi0:g}$")
        pc = find_zero_crossing(p, b_p)
        if pc is not None:
            axA.plot(pc, 0.0, "o", color=color, ms=7, mec="black", mew=0.8,
                      zorder=5)
            print(f"  xi={xi0:+.2f}: p_c = {pc:.3f}")
        else:
            print(f"  xi={xi0:+.2f}: no crossing "
                  f"(b_p {'< 0' if b_p[0] < 0 else '> 0'} everywhere -- "
                  f"{'always thermalizes' if b_p[0] < 0 else 'always runs away'})")

    axA.axhline(0.0, color="0.3", lw=1.0, ls=":")
    axA.set_xlabel(r"$p$  (momentum, units of $m_e c$)")
    axA.set_ylabel(r"$b_p(p,\xi)$")
    axA.legend(fontsize=8.5, loc="lower right", ncol=2)
    axA.set_title(r"$b_p(p)$ at fixed $\xi$: dots = zero crossing "
                  r"$p_c(\xi)$ "
                  "(the Fig. 2 separatrix)\n"
                  r"$\xi\leq 0$: field never helps "
                  r"$\Rightarrow$ always thermalizes",
                  fontsize=10.5)

    # ---- Panel B: decompose b_p into its three terms, at one xi near the
    # separatrix, where the competition is tightest ----
    xi_star = 0.6
    field, neg_drag, synchrotron = drift_terms(p, xi_star, cfg)
    total = field + neg_drag + synchrotron
    total_check, _, _ = re2d_coeffs(p, xi_star * np.ones_like(p), cfg)
    assert np.allclose(total, total_check, atol=1e-10), \
        "decomposition must reproduce re2d_coeffs exactly"

    axB.plot(p, field, color="#4c72b0", lw=1.8, ls="--",
              label=r"$+E\,\xi$  (field, constant in $p$)")
    axB.plot(p, neg_drag, color="#c44e52", lw=1.8, ls="--",
              label=r"$-F_{\rm drag}(p)$   ($\sim -1/p^2$ at large $p$)")
    axB.plot(p, synchrotron, color="#8172b2", lw=1.8, ls="--",
              label=r"$-\gamma p(1-\xi^2)/\tau_{\rm syn}$  (synchrotron)")
    axB.plot(p, total, color="black", lw=2.6,
              label=r"$b_p$  (sum, thick)")
    axB.axhline(0.0, color="0.3", lw=1.0, ls=":")
    pc_star = find_zero_crossing(p, total)
    if pc_star is not None:
        axB.axvline(pc_star, color="0.3", lw=1.0, ls=":")
        axB.plot(pc_star, 0.0, "o", color="black", ms=8, zorder=5)
        axB.annotate(fr"$p_c({xi_star:g})={pc_star:.2f}$",
                      xy=(pc_star, 0.0), xytext=(pc_star + 0.5, field[0] * 0.5),
                      fontsize=9.5,
                      arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8))

    axB.set_xlabel(r"$p$  (momentum, units of $m_e c$)")
    axB.set_ylabel(r"drift contribution")
    axB.legend(fontsize=8.5, loc="upper right")
    axB.set_title(fr"Decomposition of $b_p(p)$ at $\xi={xi_star:g}$: "
                  "field vs. drag vs. synchrotron\n"
                  r"drag falls $\sim 1/p^2$, so the (constant) field term "
                  "eventually wins", fontsize=10.5)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig06_drift_components.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPanel B check at xi={xi_star:g}: "
          f"max|decomposed sum - re2d_coeffs| = "
          f"{np.max(np.abs(total - total_check)):.2e}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
