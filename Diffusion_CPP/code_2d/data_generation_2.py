"""
data_generation.py  --  LOGIC PIECE 1 of 3
==========================================
Generate large-Delta-t transition data from the ground-truth 2D
runaway-electron jump-diffusion (see common.py for the model + references).

Outputs (in Config.data_dir):
  data_pairs.npz : survivor pairs (X_t, X_{t+Dt}), X = (p, xi) -- the flow-map
                   training set (survivors never leave the domain, so the flow
                   map is never evaluated out-of-domain over long rollouts)
  data_exit.npz  : exit labels (X_t, side) for ALL starts, side in
                   {0: stayed, 1: exited p_min (thermalized),
                    2: exited p_max (runaway)} -- the exit-head training set

Additionally, `n_exit_extra` STRATIFIED exit labels are generated with the
starts concentrated in the two boundary bands (where the one-step exit
fronts are steep and uniform sampling is too sparse) -> data_exit_extra.npz.

Run:  python data_generation.py         (full: pairs + exit + extra exit)
      python data_generation.py exit    (ONLY the stratified extra exit
                                         labels; data_pairs.npz and
                                         data_exit.npz are left untouched --
                                         use this to improve the exit head
                                         without regenerating flow data)
"""

import json
import os
import sys

import numpy as np

from common import Config, knock_rate, simulate_re2d_step, set_seed


def gen_exit_extra(cfg, rng):
    """Stratified exit-label starts: 50% in the thermalization band
    [p_min, p_min+3], 30% in the runaway band [p_max-1.2, p_max], 20%
    uniform; xi uniform everywhere (the classifier needs negatives too)."""
    n = cfg.n_exit_extra
    n1 = int(0.5 * n); n2 = int(0.3 * n); n3 = n - n1 - n2
    p0 = np.concatenate([
        rng.uniform(cfg.p_min, min(cfg.p_min + 3.0, cfg.p_max), n1),
        rng.uniform(max(cfg.p_max - 1.2, cfg.p_min), cfg.p_max, n2),
        rng.uniform(cfg.p_min, cfg.p_max, n3)])
    xi0 = rng.uniform(-1.0, 1.0, n)
    _, _, _, side = simulate_re2d_step(p0, xi0, cfg, rng)
    x = np.stack([p0, xi0], axis=1).astype(np.float32)
    np.savez(os.path.join(cfg.data_dir, "data_exit_extra.npz"),
             x=x, side=side.astype(np.int64))
    print(f"  extra exit labels: {n} stratified "
          f"(thermal band {n1}, runaway band {n2}, uniform {n3}); "
          f"exit/step: thermal={float((side == 1).mean()):.4f}, "
          f"runaway={float((side == 2).mean()):.4f}")


def main():
    exit_only = len(sys.argv) > 1 and sys.argv[1] == "exit"
    cfg = Config()
    set_seed(cfg.seed)
    # a different stream for the extra labels so an exit-only rerun does not
    # duplicate the main run's samples
    rng = np.random.default_rng(cfg.seed + 10 if exit_only else cfg.seed)
    os.makedirs(cfg.data_dir, exist_ok=True)

    if exit_only:
        print(f"model={cfg.model} | EXIT-ONLY mode: stratified boundary-band "
              f"exit labels -> data_exit_extra.npz")
        gen_exit_extra(cfg, rng)
        print("Done.")
        return

    # jump-activity summary at a few momenta (rate in tau_c units)
    p_probe = np.array([cfg.p_min, 2.0, 5.0, cfg.p_max])
    lam = knock_rate(p_probe, cfg)
    print(f"model={cfg.model} | domain=[{cfg.p_min},{cfg.p_max}]x[-1,1] "
          f"| dt={cfg.dt} | E/Ec={cfg.E_hat} | eps_min={cfg.eps_min}")
    print("  knock-on rate lambda(p)*dt at p=" +
          ", ".join(f"{pv:g}: {lv * cfg.dt:.3f}" for pv, lv in zip(p_probe, lam)))

    n = cfg.n_full
    p0 = rng.uniform(cfg.p_min, cfg.p_max, size=n)
    xi0 = rng.uniform(-1.0, 1.0, size=n)
    p1, xi1, alive, side = simulate_re2d_step(p0, xi0, cfg, rng)

    x_all = np.stack([p0, xi0], axis=1).astype(np.float32)
    y_all = np.stack([p1, xi1], axis=1).astype(np.float32)

    # survivor flow pairs (in-domain -> in-domain)
    np.savez(os.path.join(cfg.data_dir, "data_pairs.npz"),
             x_sample=x_all[alive], y_sample=y_all[alive])
    # exit labels for ALL starts (3-class side)
    np.savez(os.path.join(cfg.data_dir, "data_exit.npz"),
             x=x_all, side=side.astype(np.int64))

    fr_lo = float((side == 1).mean())
    fr_hi = float((side == 2).mean())
    print(f"  survivors={int(alive.sum())}/{n}  "
          f"exit/step: thermal (p<p_min)={fr_lo:.4f}, runaway (p>p_max)={fr_hi:.4f}")
    incr = y_all[alive] - x_all[alive]
    print(f"  survivor increment mean={incr.mean(axis=0)} std={incr.std(axis=0)}")

    # stratified extra exit labels for the steep boundary bands
    gen_exit_extra(cfg, rng)

    with open(os.path.join(cfg.data_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
