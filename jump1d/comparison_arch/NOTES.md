# comparison_arch/ — architecture experiment (NSF + v-prediction + DDIM)

Copy of `jump1d/comparison/` (the **original baseline** hyperparameters, not
`comparison_tuned`'s) with three targeted architecture upgrades. Same dataset
(`../artifacts_bd/data_pairs.npz`), same metrics pipeline, same figures —
nothing outside this folder was touched. All capacity/epoch/timestep-count
hyperparameters are left at the original baseline values so the *only*
variable under test is the architecture change itself.

**Not run.** I verified the new code with a local, CPU-only smoke test
(tiny synthetic data, a few steps) confirming the spline transform is
invertible to ~1e-6 and both diffusion samplers/parameterizations produce
finite output — that's a correctness check, not a real experiment. No
checkpoints/figures/results exist yet. Hand this to the GPU box and run the
commands under "How to run" below.

## What changed and why

### 1. RealNVP → Neural Spline Flow (`nf_coupling_type: "affine" → "spline"`)

Diagnosis from the baseline run: `models/layers.py`'s `ConditionalAffineCoupling`
computes `y = x*exp(s(ctx)) + t(ctx)` — at `dim=1` there's no second half of
the vector to hold out, so `s`/`t` depend on `ctx` (embedded `X_t`) only,
never on the flow's own running value. Composing affine-in-`z` maps whose
coefficients don't depend on `z` is *still affine in `z`* — so any number of
stacked affine coupling layers can only represent a **conditional Gaussian**
`p(y|x)`. That's consistent with what the two prior runs showed: RealNVP
reliably won Wasserstein (good at matching the bulk mean/scale) but
consistently lost KL/Hellinger (can't capture skew or the compression near
the absorbing walls) to DDPM.

**Fix:** `models/layers.py` now also has `ConditionalSplineCoupling`
(Neural Spline Flows, Durkan et al. 2019) — same coupling-layer contract
(`forward`/`inverse` returning `(y, logdet)` / `x`), but the per-dimension
map is a monotonic piecewise rational-quadratic spline (8 bins by default,
linear/identity tails outside `±5` standardised units) instead of an affine
transform. Still exactly invertible, still exact NLL via change-of-variables
— just not capped at Gaussian shape. `models/realnvp.py`'s
`ConditionalRealNVP` now takes a `coupling_type` argument and builds either
coupling class; nothing else in the flow (ActNorm, permutations, base
distribution, training loop) changed.

**Prediction:** should meaningfully close the KL/Hellinger gap to DDPM
without touching training/sampling cost much (still one forward pass to
sample, still exact NLL) — this is the one upgrade that should show up
across every distributional metric, not just Wasserstein.

### 2. DDPM epsilon-prediction → v-prediction (`diff_parameterization: "eps" → "v"`)

Salimans & Ho, "Progressive Distillation" (2022): instead of predicting the
injected noise `eps`, the denoiser predicts `v = alpha_t*eps - sigma_t*y0`.
`eps`-prediction is poorly conditioned at the extremes of the schedule (near
`t=0`, `eps` barely affects `y_t`; near `t=T`, `y_t` is almost pure noise) —
`v`-prediction is a fixed rotation of the same information that stays
well-scaled across the whole schedule. `models/diffusion.py` now derives
`(x0_pred, eps_pred)` analytically from whichever the network predicts
(`_predict_x0_eps`), so both samplers work unchanged regardless of
`parameterization`.

### 3. Ancestral sampling → DDIM (`diff_sampler: "ancestral" → "ddim"`, `diff_ddim_steps: 50`)

Song, Meng & Ermon (2020): a deterministic (or partially stochastic, via
`diff_ddim_eta`), non-Markovian reverse process that **reuses the exact same
trained network** — no retraining needed to switch samplers — but walks a
strided subsequence of `diff_ddim_steps` timesteps instead of all
`diff_n_timesteps`. Default here: 50 steps instead of 1000, a 20x reduction
(vs. the 5x you get from just cutting `diff_n_timesteps` at training time,
which was tested in `comparison_tuned` and cost some KL/Hellinger). Because
this only changes *inference*, it can in principle be applied on top of
either parameterization — `GaussianDiffusion.sample()` dispatches on
`self.sampler` automatically.

**Prediction for #2 + #3 combined:** the open question this experiment
answers is whether v-prediction's better-conditioned training recovers
enough quality headroom that a 20x-fewer-step DDIM sampler still lands close
to (or better than) the original 1000-step ancestral DDPM — i.e. whether you
can get RealNVP-like sampling speed out of DDPM without the Wasserstein-outlier
regression that the naive `comparison_tuned` timestep cut showed on KL/Hellinger.

## What did *not* change

`nf_n_coupling`, `nf_hidden_dim`, `nf_context_dim`, `nf_epochs`,
`diff_hidden_dim`, `diff_n_res_blocks`, `diff_epochs`, `diff_n_timesteps`
(the *training*-time timestep count — still 1000, only inference is
strided down to 50 via DDIM), `batch_size`, `lr`, `weight_decay`, `grad_clip`
— all at the original baseline. Any metric difference from
`comparison/results/results.json` traces back only to the three items above.

*Not implemented this round (available if you want a follow-up):* the
boundary-aware reparameterization (diffusing in a domain-matched warped
space) that would directly target DDPM's boundary-violation/outlier issue —
skipped per your last scope decision, still the highest-ceiling option if
the DDIM/v-prediction combo doesn't fully fix the Wasserstein tail behavior.

## How to run (on the GPU box)

```bash
cd jump1d/comparison_arch
pip install -r requirements.txt

python train_nf.py           # -> checkpoints/realnvp.pt   (Neural Spline Flow)
python train_diffusion.py    # -> checkpoints/diffusion.pt (v-pred, saves both samplers' config)
python compare.py            # -> results/results.{csv,json}, figures/*.{png,pdf}
```

Same `--device cuda:0` / `cuda:1` flags as before if you want to run both
trainings in parallel on two GPUs.

To A/B the DDIM step count or eta without retraining (sampler-only change,
reuses `checkpoints/diffusion.pt`):
```bash
python evaluate.py --model diffusion --diff_ddim_steps 20
python evaluate.py --model diffusion --diff_ddim_steps 100
python evaluate.py --model diffusion --diff_sampler ancestral   # back to the original sampler, same v-pred weights
```

## Comparing against the original baseline

```bash
python -c "
import json
base = json.load(open('../comparison/results/results.json'))
arch = json.load(open('results/results.json'))
for b, t in zip(base, arch):
    print(b['model'])
    for k in ('wasserstein','kl_divergence','hellinger','sampling_rate_samples_per_s','training_time_s'):
        print(f'  {k:28s} {b[k]!s:>18s} -> {t[k]!s:>18s}')
"
```
