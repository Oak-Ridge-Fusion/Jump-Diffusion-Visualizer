"""
fig05_knock_rate.py
=====================
Figure 5: the knock-on collision rate lambda(p), from common_2.knock_rate.

Physics
-------
lambda(p) is the rate of the compound-Poisson jump process used in Figures
4: the frequency of large-angle ("knock-on") Moller collisions transferring
more than eps_min of kinetic energy. In tau_c units (see common_2.py
docstring, Embreus 2018):

  lambda(p) = R * beta(p) * sigma_hat(gamma(p)) / (4 lnLambda)

  beta(p)      = p / gamma            (relativistic speed, -> 1 as p -> inf)
  sigma_hat    = moller_sigma_hat(gamma):  the Moller cross-section
                 (in units of 2 pi r0^2), integrated over the open range of
                 transferred energy eps in [eps_min, (gamma-1)/2]
  R            = knock_density_ratio  (n_tot/n_e, =1 by default)
  lnLambda     = Coulomb logarithm (=15 by default)

Panel A shows lambda(p) itself, decomposed against its two p-dependent
factors (beta, sigma_hat), each normalized to its own domain maximum so
their SHAPES can be compared on one axis. This answers a natural first
question: given beta(p) rises monotonically with p (faster particles) while
sigma_hat(gamma) falls (the momentum-transfer cross-section shrinks for a
more energetic beam), does lambda rise, fall, or stay flat? The two effects
turn out to nearly cancel, so lambda(p) is close to flat across the whole
domain -- collisions are neither much rarer nor much more frequent for a
runaway candidate than for a electron near p_min.

Panel B shows the sensitivity of lambda(p) to eps_min, the energy-transfer
threshold that defines the FP/jump split (below eps_min, an energy transfer
is treated as part of the continuous Fokker-Planck drag/diffusion; above
it, as a discrete jump -- common_2.py Sec. "Jump part"). Raising eps_min
shrinks the open integration range [eps_min, (gamma-1)/2] (and can close
the channel entirely at low p, where (gamma-1)/2 < eps_min), so lambda drops
-- this is the same knob that would, in the ML pipeline, trade off how much
of the large-angle physics the jump head has to learn vs. how much is
folded into the (already-reduced) Fokker-Planck drag via lnLambda_bar.

Output: problem3/figures/fig05_knock_rate.png
"""

import os
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import Config, gamma_of_p, moller_sigma_hat, knock_rate

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    cfg = Config()
    p = np.linspace(cfg.p_min, cfg.p_max, 400)
    gamma = gamma_of_p(p)
    beta = p / gamma
    sigma_hat = moller_sigma_hat(gamma, cfg)
    lam = knock_rate(p, cfg)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # ---- Panel A: lambda(p) and its two shape-factors ----
    axA.plot(p, lam, color="black", lw=2.4, label=r"$\lambda(p)$  (actual rate)")
    axA.set_xlabel(r"$p$  (momentum, units of $m_e c$)")
    axA.set_ylabel(r"$\lambda(p)$   [$\tau_c^{-1}$]", fontsize=11)
    axA.set_ylim(0, max(lam) * 1.35)

    axA2 = axA.twinx()
    axA2.plot(p, beta / beta.max(), color="#4c72b0", lw=1.6, ls="--",
              label=r"$\beta(p)/\beta_{\max}$  (speed, shape only)")
    axA2.plot(p, sigma_hat / sigma_hat.max(), color="#c44e52", lw=1.6,
              ls="--", label=r"$\hat\sigma(\gamma)/\hat\sigma_{\max}$  "
              "(Moller cross-section, shape only)")
    axA2.set_ylabel("normalized shape factors (arb. units)", fontsize=9.5)
    axA2.set_ylim(0, 1.35)

    lines = axA.get_lines() + axA2.get_lines()
    axA.legend(lines, [l.get_label() for l in lines], fontsize=8.5,
               loc="upper center")
    axA.set_title(r"$\lambda = R\,\beta\,\hat\sigma / (4\ln\Lambda)$: "
                  r"$\beta\uparrow$ and $\hat\sigma\downarrow$ nearly "
                  "cancel\n"
                  f"(default: eps_min={cfg.eps_min:g}, "
                  f"lnLambda={cfg.lnLambda:g}, R={cfg.knock_density_ratio:g})",
                  fontsize=10.5)

    # ---- Panel B: sensitivity to eps_min (the FP/jump split threshold) ----
    eps_values = [0.01, 0.02, 0.05, 0.10, 0.20]
    cmap = plt.get_cmap("viridis")
    for i, eps in enumerate(eps_values):
        cfg_i = replace(cfg, eps_min=eps)
        lam_i = knock_rate(p, cfg_i)
        style = "-" if abs(eps - cfg.eps_min) < 1e-12 else "--"
        lw = 2.4 if abs(eps - cfg.eps_min) < 1e-12 else 1.4
        axB.plot(p, lam_i, color=cmap(i / (len(eps_values) - 1)), lw=lw,
                  ls=style,
                  label=fr"$\epsilon_{{\min}}={eps:g}$"
                  + ("  (default)" if style == "-" else ""))
    axB.set_xlabel(r"$p$  (momentum, units of $m_e c$)")
    axB.set_ylabel(r"$\lambda(p)$   [$\tau_c^{-1}$]", fontsize=11)
    axB.legend(fontsize=8.5, loc="upper right")
    axB.set_title(r"Sensitivity to $\epsilon_{\min}$: raising the FP/jump "
                  "split\nthreshold suppresses the jump rate "
                  "(and can close the channel\nat low $p$)", fontsize=10.5)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig05_knock_rate.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"lambda(p) at p_min, mid, p_max: "
          f"{lam[0]:.4f}, {lam[len(lam)//2]:.4f}, {lam[-1]:.4f}  "
          f"[tau_c^-1]  (variation across the domain: "
          f"{(lam.max()-lam.min())/lam.mean()*100:.1f}% of the mean)")
    for eps in eps_values:
        cfg_i = replace(cfg, eps_min=eps)
        lam_i = knock_rate(p, cfg_i)
        closed = np.sum(lam_i == 0.0)
        print(f"  eps_min={eps:5.2f}: lambda range "
              f"[{lam_i.min():.4f}, {lam_i.max():.4f}], "
              f"channel closed at {closed}/{len(p)} grid points")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
