"""
data_generation.py  --  LOGIC PIECE 1 of 3
==========================================
Generate large-Delta-t transition data from the ground-truth 1D SDE.

  Config.model = "merton" / "double_well"  (UNBOUNDED):
      one dataset of pairs (X_t, X_{t+Dt})            -> data_pairs.npz

  Config.model = "bounded"  (ABSORBING boundaries, RE-aligned):
      survivor pairs (X_t, X_{t+Dt}) for non-exiting steps  -> data_pairs.npz
      exit labels    (X_t, exited?)                          -> data_exit.npz
      (Survivors never leave [x_min,x_max], so the flow map is never evaluated
       out-of-domain -- this is what fixes the long-rollout OOD problem.)

Run:  python data_generation.py
"""

import json
import os
import sys

import numpy as np

from common_1 import (Config, reference_transition_samples,
                      simulate_bounded_step, set_seed)


def main():
    # optional CLI arg selects the model: python data_generation.py bounded_sd
    cfg = Config(model=sys.argv[1]) if len(sys.argv) > 1 else Config()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    os.makedirs(cfg.data_dir, exist_ok=True)

    if cfg.model in ("bounded", "bounded_sd"):
        n = cfg.n_full
        from common_1 import bounded_coeffs
        lam_c = bounded_coeffs(np.array([0.5 * (cfg.x_min + cfg.x_max)]), cfg)[2]
        print(f"model={cfg.model} | domain=[{cfg.x_min},{cfg.x_max}] | dt={cfg.dt} "
              f"| mean jumps/step at center={float(lam_c[0])*cfg.dt:.3f}")
        x0 = rng.uniform(cfg.x_min, cfg.x_max, size=n)
        x_end, alive = simulate_bounded_step(x0, cfg, rng)

        # survivor flow pairs (in-domain -> in-domain)
        np.savez(os.path.join(cfg.data_dir, "data_pairs.npz"),
                 x_sample=x0[alive].astype(np.float32),
                 y_sample=x_end[alive].astype(np.float32))
        # exit labels for ALL starts (1 = exited this step)
        np.savez(os.path.join(cfg.data_dir, "data_exit.npz"),
                 x=x0.astype(np.float32),
                 exited=(~alive).astype(np.float32))
        print(f"  survivors={int(alive.sum())}/{n}  "
              f"exit fraction/step={float((~alive).mean()):.4f}")
    else:
        n = cfg.n_full
        lo, hi = cfg.x_range
        print(f"model={cfg.model} | dt={cfg.dt} | mean jumps/step="
              f"{float(cfg.lam(0.0))*cfg.dt:.3f}")
        x_sample = rng.uniform(lo, hi, size=n)
        y_sample = reference_transition_samples(x_sample, cfg, rng)
        np.savez(os.path.join(cfg.data_dir, "data_pairs.npz"),
                 x_sample=x_sample.astype(np.float32),
                 y_sample=y_sample.astype(np.float32))
        incr = y_sample - x_sample
        print(f"  increment mean={incr.mean():.4f} std={incr.std():.4f}")

    with open(os.path.join(cfg.data_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
