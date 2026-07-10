# comparison_tuned/ — hyperparameter-only experiment

Clean copy of `jump1d/comparison/`. **No model architecture, no metrics, no
dataset code changed** — every file is byte-identical to the original except
`config.py`, where a handful of default hyperparameters were changed. Reads
the exact same `../artifacts_bd/data_pairs.npz` as the original benchmark
(nothing in `jump1d/` was touched or regenerated).

Not run yet — hand this off to a GPU box and run the three commands under
"How to run" below. Results will land in `comparison_tuned/results/` and
`comparison_tuned/figures/`, separate from the original `comparison/results/`
so you can diff the two side by side.

## What changed and why

Baseline numbers below are from the original run (`comparison/results/results.json`).

### Conditional RealNVP

| field | baseline | tuned | why |
|---|---|---|---|
| `nf_n_coupling` | 8 | **16** | For a scalar state (`dim=1`), every coupling layer's scale/shift is a function of the embedded `X_t` context only (there's no second half of the vector to condition on) — so the whole flow reduces to a conditional-Gaussian `p(y\|x)`. More layers can't lift that ceiling, but they let the x-dependent mean/std function fit the true conditional mean and variance more tightly. |
| `nf_hidden_dim` | 128 | **256** | Wider coupling-layer networks — more capacity to model that mean/std function. |
| `nf_context_dim` | 64 | **128** | Richer embedding of `X_t` feeding every coupling layer. |
| `nf_epochs` | 200 | **300** | The extra capacity above needs more optimization steps to converge; kept at the same batch size / LR so this isolates capacity, not optimizer settings. |

**Prediction:** should tighten Wasserstein/NLL a bit further (RealNVP's existing strength) but is unlikely to move KL/Hellinger much, since the fundamental limitation — a single conditional Gaussian can't represent the skew/multimodality near the absorbing walls — isn't something these hyperparameters can fix. That would require a different transform (e.g. spline coupling), which is out of scope here since you asked to keep this to hyperparameters only.

### Conditional DDPM

| field | baseline | tuned | why |
|---|---|---|---|
| `diff_n_timesteps` | 1000 | **200** | The main lever. DDPM's headline weakness was ~3,800x slower sampling than RealNVP (820 vs. 3.1M samples/s) because sampling requires one network call per timestep. Cutting timesteps 5x directly targets that gap. Denoiser capacity (`diff_hidden_dim`, `diff_n_res_blocks`) is left unchanged so step-count is the only thing being tested. |
| `diff_epochs` | 200 | **250** | With 5x fewer timesteps, each one has to remove a coarser slice of noise per step — a bit more training compensates for the harder per-step task. |

**Prediction:** sampling speed should improve roughly proportionally (expect somewhere in the ballpark of 5x faster, i.e. ~4,000 samples/s instead of 820/s), closing much of the gap with RealNVP. The open question this experiment actually answers is how much distributional fidelity (KL/Hellinger/Wasserstein) is given up for that — DDPM's advantage was fitting the bulk of the density very well with 1000 steps, and this is the direct trade-off knob for that.

Everything else (`batch_size`, `lr`, `weight_decay`, `grad_clip`, denoiser width/depth, RealNVP's `n_hidden_layers`/`use_actnorm`) is untouched from baseline, so any metric differences trace back to only the fields in the tables above.

## How to run (on the GPU box)

```bash
cd jump1d/comparison_tuned
pip install -r requirements.txt

python train_nf.py          # -> checkpoints/realnvp.pt
python train_diffusion.py   # -> checkpoints/diffusion.pt
python compare.py            # -> results/results.{csv,json}, figures/*.{png,pdf}
```

`config.device = "auto"` picks CUDA automatically if available (falls back to
CPU otherwise — note it does not currently auto-detect Apple `mps`, only
`cuda`). No flags are needed; the tuned values above are now the file's
defaults. To A/B against the original settings without touching this folder,
pass the old values back in on the command line, e.g.:

```bash
python train_diffusion.py --diff_n_timesteps 1000 --diff_epochs 200
```

## Comparing against the original run

```bash
python -c "
import json
base = json.load(open('../comparison/results/results.json'))
tuned = json.load(open('results/results.json'))
for b, t in zip(base, tuned):
    print(b['model'])
    for k in ('wasserstein','kl_divergence','hellinger','sampling_rate_samples_per_s','training_time_s'):
        print(f'  {k:28s} {b[k]!s:>18s} -> {t[k]!s:>18s}')
"
```
