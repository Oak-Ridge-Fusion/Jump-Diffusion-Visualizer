# Learning Diffusion SDE — Educational Implementation

An educational, step-by-step implementation of **training-free conditional diffusion models** for learning stochastic flow maps of SDEs in bounded domains.

Based on the paper:
> *Generative AI Models for Learning Flow Maps of Stochastic Dynamical Systems in Bounded Domains*

---

## Goal

Reproduce **Figure 2** of the paper: four histograms comparing different training data strategies for learning the flow map of a Brownian motion with absorbing boundaries.

```
Ground Truth | Our Method | All Trajectories | Only Confined
```

---

## The SDE

$$dX_t = dW_t, \quad X_0 = 1, \quad 0 < X_t < 6 \quad \text{(absorbing boundaries)}$$

Pure Brownian motion with absorbing walls. Simple equation, non-trivial terminal distribution.

---

## Project Structure

```
learning_diffusion_sde/
├── config.py                    # Central configuration (T, dt, N, etc.)
├── utils.py                     # Shared helpers (plotting, SDE steps, VP schedule)
├── model.py                     # FlowNet architecture (shared across scripts)
│
├── 01_simulate_ground_truth.py  # SDE simulation → ground truth histogram
├── 02_build_dataset.py          # Generate (x_t, ΔX) training pairs
├── 03_forward_diffusion.py      # VP-SDE noise schedule, Z_t distributions
├── 04_knn_score_estimation.py   # KNN approximation of ∇log p_t(z)
├── 05_probability_flow_ode.py   # Reverse ODE: noise → increment
├── 06_generate_labels.py        # (x, z, ΔX) label pairs via reparameterisation
├── 07_train_network.py          # Train G(x,z) → ΔX with MSE
├── 08_rollout.py                # Generate 'Our Method' histogram
├── 09_compare_results.py        # Side-by-side GT vs Our Method
├── 10_all_trajectories_trained.py  # Ablation: include boundary crossings
├── 11_only_confined.py          # Ablation: only surviving particles
├── 12_reproduce_figure2.py      # Final 4-panel Figure 2
│
├── data/                        # Generated .npz files
├── models/                      # Trained .pt weights
├── plots/                       # All generated figures
└── docs/
    ├── 01_SDE.md                # SDE mathematics
    ├── 02_Diffusion.md          # VP-SDE noise schedule
    ├── 03_KNNScore.md           # KNN score estimation
    ├── 04_ProbabilityFlowODE.md # Reverse ODE derivation
    ├── 05_NeuralNetwork.md      # FlowNet architecture
    └── 06_Rollout.md            # Rollout and ablations
```

---

## Run Order

```bash
cd learning_diffusion_sde
python 01_simulate_ground_truth.py    # ~20s
python 02_build_dataset.py            # ~40s
python 03_forward_diffusion.py        # ~10s
python 04_knn_score_estimation.py     # ~2min
python 05_probability_flow_ode.py     # ~5min
python 06_generate_labels.py          # ~15s
python 07_train_network.py            # ~2min (MPS/GPU) or ~10min (CPU)
python 08_rollout.py                  # ~35min (200k particles × 6000 steps)
python 09_compare_results.py          # ~5s
python 10_all_trajectories_trained.py # ~60min (re-trains + re-rolls)
python 11_only_confined.py            # ~60min
python 12_reproduce_figure2.py        # ~5s
```

---

## Key Concepts at Each Step

| Script | Concept | Key equation |
|---|---|---|
| 01 | Euler-Maruyama | $X_{n+1} = X_n + \sqrt{dt} \cdot Z$ |
| 02 | Training pairs | $(x_t, \Delta X)$ with both endpoints inside |
| 03 | VP-SDE | $Z_t = \alpha(t) Z_0 + \sigma(t) \varepsilon$ |
| 04 | KNN score | $s(z) \approx (\mu_{\text{KNN}} - z) / \text{var}_{\text{KNN}}$ |
| 05 | PF-ODE | $dZ/dt = -\tfrac{1}{2}\beta[Z + s(Z,t)]$ |
| 06 | Reparameterisation | $z = (\Delta X - \mu(x)) / \sigma(x)$ |
| 07 | MSE training | $\mathcal{L} = E[(G(x,z) - \Delta X)^2]$ |
| 08 | Rollout | $x_{\text{new}} = x + G(x, z \sim N(0,1))$ |

---

## Results

| Condition | Survival rate | KS vs GT |
|---|---|---|
| Ground Truth | 43.87% | — |
| Our Method | 43.34% | 0.086 |
| All Trajectories | TBD | TBD |
| Only Confined | TBD | TBD |

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.11
matplotlib >= 3.7
torch >= 2.1
tqdm >= 4.65
```

Install: `pip install -r requirements.txt`

---

## Educational Design Principles

Every script:
- Prints detailed diagnostics
- Generates plots explaining the current step
- Contains ASCII diagrams and mathematical derivations in comments
- Explains the WHY, not just the WHAT
- Can be run independently

The repository is designed to feel like a **textbook implemented in code**.
