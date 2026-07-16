"""
common.py
=========
Shared library for the 2D runaway-electron (RE) jump-diffusion flow-map
verification.  Same three-piece structure and training-free diffusion engine
as jump1d; the SDE is now the physics model of the note's Section 1.

THE MODEL (every formula is literature-grounded; see README.md for details)
---------------------------------------------------------------------------
State X = (p, xi):  p = momentum / (m_e c),  xi = cos(pitch angle).
Normalized units of the ORNL backward-Monte-Carlo RE papers
[Zhang & del-Castillo-Negrete, Phys. Plasmas 24, 092511 (2017),
 arXiv:1708.00947, Eqs. (5)-(6)]:
  time in relativistic collision times tau_c = m_e c / (e E_c),
  E in units of the Connor-Hastie critical field E_c,
  gamma = sqrt(1 + p^2),  beta = v/c = p/gamma.

Drift-diffusion part (small-angle collisions + E-field + synchrotron):
  dp  = [ E xi - Fdrag(p) - gamma p (1-xi^2)/tau_syn ] dt
  dxi = [ E (1-xi^2)/p + xi (1-xi^2)/(tau_syn gamma) - xi nu_c ] dt
        + sqrt( nu_c (1-xi^2) ) dW
with Fdrag = (1+p^2)/p^2  and  nu_c = (Z+1) gamma / p^3.

Jump part (large-angle "knock-on" collisions = compound Poisson):
  * rate  lambda(p) = (1/2) n v sigma(p): sigma(p) is the ANALYTIC Moller
    cross-section integrated over transfers eps in [eps_min, (gamma-1)/2]
    [Embreus, Stahl & Fulop, J. Plasma Phys. 84, 905840102 (2018),
     arXiv:1708.08779, Eq. (35); eps_max=(gamma-1)/2 from indistinguishability];
    in tau_c units  lambda = R * beta * sigma_hat / (4 lnLambda)  with
    sigma_hat = sigma/(2 pi r0^2) and R = n_tot/n_e (DREAM uses the TOTAL
    electron density for knock-on, arXiv:2103.16457 Eq. (22); R=1 default).
  * transferred energy eps ~ Moller spectrum, Embreus Eq. (5) (heavy-tailed,
    ~1/eps^2), sampled by rejection with a 1/eps^2 envelope.
  * primary after the event: gamma' = gamma - eps (target at rest), deflected
    by cos(theta_d) = sqrt[ (gamma'-1)(gamma+1) / ((gamma'+1)(gamma-1)) ]
    (Embreus Eq. (6) applied to the outgoing primary), random azimuth.
  * double counting removed by reducing the Fokker-Planck Coulomb logarithm:
    lnLambda_bar = lnLambda - (1/2) ln[(gamma-gamma_m)/(gamma_m-1)]
    (Embreus Eqs. (13)-(14)); applied to the drag (electron-dominated) and
    to the e-e unit of the deflection frequency ONLY:
    Fdrag = f*(1+p^2)/p^2,  nu_c = (Z + f)*gamma/p^3  with f = lnLbar/lnL
    (the e-i part has no jump channel and is not reduced).

Domain: (p, xi) in [p_min, p_max] x [-1, 1].  ABSORBING at both p ends:
  p < p_min  -> the electron falls back into the thermal bulk (the separatrix
                exit of the note; jump- and drag-driven),
  p > p_max  -> confirmed runaway exit (the quantity of interest in the
                exit-time RE literature, arXiv:2104.14561).
REFLECTING at xi = +-1 (the pitch diffusion vanishes there).

The training-free diffusion engine (KNN-once + probability-flow ODE + FN_Net
distillation) is the same as jump1d / the JCP reference code, generalized
only by dimension (the JCP 3D code uses the identical pipeline in 3D).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    model: str = "re2d"              # single model in this package
    physics_rev: int = 2             # bumped whenever the ground-truth
                                     # physics functions change, so caches
                                     # keyed on the config are invalidated
                                     # (rev 2: the lnLambda_bar reduction no
                                     # longer acts on the electron-ion part
                                     # of the pitch scattering)

    # ---- physics: drift-diffusion (normalized units, see module docstring) --
    E_hat: float = 2.0               # E / E_c  (>1 so runaway is possible)
    Z: float = 1.0                   # effective ion charge
    tau_syn: float = 100.0           # synchrotron timescale (tau_c units); <=0 disables
    lnLambda: float = 15.0           # Coulomb logarithm (only the jump physics
                                     # needs it explicitly; the FP part absorbed
                                     # lnLambda into tau_c)

    # ---- physics: knock-on jump process --------------------------------------
    jumps_on: bool = True
    eps_min: float = 0.02            # FP/Boltzmann energy-transfer cutoff, m_e c^2
                                     # units (0.02 ~ 10 keV; must stay well below
                                     # the critical energy, cf. DREAM pCutAvalanche)
    knock_density_ratio: float = 1.0 # R = n_tot/n_e for the jump channel (DREAM
                                     # uses n_tot = free+bound electrons; R>1
                                     # mimics material-injection scenarios)
    knock_lnL_correction: bool = True  # reduce the e-e Coulomb log (drag +
                                       # e-e unit of nu_c; NOT the e-i part)
                                       # (Embreus Eqs. 13-14, anti-double-count)

    # ---- domain and the LARGE coarse step ------------------------------------
    p_min: float = 1.0
    p_max: float = 10.0
    dt: float = 1.0                  # the ML big step (tau_c units)

    # ---- reference fine integrator (the "numerical method") ------------------
    n_sub: int = 400                 # sub-steps per dt (dt_sub = 0.0025).
                                     # Set by convergence_check.py: n_sub=200
                                     # (the JCP 3D code's dt=0.005) showed a
                                     # localized pitch-distribution bias at the
                                     # low-p/high-xi corner (1.3, 0.8); 400
                                     # passes at every start vs the 800 ref.
    seed: int = 0

    # ---- training-free diffusion engine (mirrors the JCP 3D code) -----------
    diff_scale: tuple = (1.5, 1.5)   # per-dim scale on the increment (p, xi)
    train_size_labels: int = 50000   # (x0,z)->increment labels to generate
    knn_k: int = 1024                # neighbours per x0
    ode_steps: int = 2000            # probability-flow ODE steps
    label_chunk: int = 10000         # chunk over labels to bound GPU memory

    # ---- distillation net (FN_Net) -------------------------------------------
    hid_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-6
    distill_iters: int = 15000

    # ---- exit head (3-class: stay / exit-low / exit-high) --------------------
    exit_hidden: int = 256
    exit_epochs: int = 300           # cosine LR decay + best-on-valid (the
                                     # corrected-physics run showed valid CE
                                     # still falling at 120 epochs)
    n_exit_extra: int = 400000       # stratified extra exit labels focused on
                                     # the two boundary bands (steep exit
                                     # fronts) -> data_exit_extra.npz

    # ---- dataset / evaluation sizes ------------------------------------------
    n_full: int = 600000             # big-step pairs for training data
    n_ref_mc: int = 200000           # particles per start in the B2 rollout
    n_b0: int = 100000               # particles per start in the B0 transition
    b1_grid: tuple = (36, 18)        # (n_p, n_xi) grid for the exit-prob maps
    b1_mc: int = 1500                # MC particles per grid point (ground truth)
    rollout_K: int = 10              # big ML steps for the B2 rollout (T = K dt)
    transition_test_x: tuple = ((2.0, 0.6), (5.0, 0.9))  # fixed (p, xi) starts

    device: str = "cuda"
    data_dir: str = "auto"           # auto -> artifacts_re2d

    def __post_init__(self):
        if self.data_dir == "auto":
            self.data_dir = "artifacts_re2d"

    def to_dict(self):
        d = asdict(self)
        d["diff_scale"] = list(self.diff_scale)
        d["b1_grid"] = list(self.b1_grid)
        d["transition_test_x"] = [list(v) for v in self.transition_test_x]
        return d


# ----------------------------------------------------------------------------
# 2. Physics: drift and diffusion coefficients
# ----------------------------------------------------------------------------
def gamma_of_p(p):
    p = np.asarray(p, dtype=np.float64)
    return np.sqrt(1.0 + p * p)


def lnL_reduction(gamma, cfg: Config):
    """Coulomb-log reduction factor on the Fokker-Planck drag and pitch
    scattering when the >eps_min transfers are handled as discrete jumps:
      lnLambda_bar / lnLambda = 1 - ln[(gamma-gamma_m)/(gamma_m-1)] / (2 lnLambda)
    [Embreus 2018, Eqs. (13)-(14)].  Only active where the jump channel is
    kinematically open (gamma - 1 > 2 eps_min); floored at 0.5 for safety."""
    if not (cfg.jumps_on and cfg.knock_lnL_correction):
        return np.ones_like(np.asarray(gamma, dtype=np.float64))
    gamma = np.asarray(gamma, dtype=np.float64)
    gm = 1.0 + cfg.eps_min
    fac = np.ones_like(gamma)
    open_ = gamma - 1.0 > 2.0 * cfg.eps_min
    ratio = np.maximum((gamma[open_] - gm) / (gm - 1.0), 1.0)
    fac[open_] = 1.0 - np.log(ratio) / (2.0 * cfg.lnLambda)
    return np.maximum(fac, 0.5)


def re2d_coeffs(p, xi, cfg: Config):
    """(b_p, b_xi, s_xi): drift components and the xi-diffusion amplitude of
    the RE test-particle model [Zhang & del-Castillo-Negrete PoP 2017,
    Eqs. (5)-(6)].  dp has no Brownian noise in this model -- the momentum
    randomness comes entirely from the jumps."""
    p = np.asarray(p, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    gamma = np.sqrt(1.0 + p * p)
    fac = lnL_reduction(gamma, cfg)
    drag = fac * (1.0 + p * p) / (p * p)
    # The knock-on channel replaces large-angle ELECTRON-ELECTRON collisions
    # only, so the lnLambda_bar reduction applies to the e-e unit ("1") of
    # the deflection frequency and to the (electron-dominated) drag -- NOT
    # to the electron-ion ("Z") part, which has no jump channel here.
    nu_c = (cfg.Z + fac) * gamma / (p ** 3)
    one_m = np.clip(1.0 - xi * xi, 0.0, 1.0)

    b_p = cfg.E_hat * xi - drag
    b_xi = cfg.E_hat * one_m / p - xi * nu_c
    if cfg.tau_syn > 0:
        b_p = b_p - gamma * p * one_m / cfg.tau_syn
        b_xi = b_xi + xi * one_m / (cfg.tau_syn * gamma)
    s_xi = np.sqrt(nu_c * one_m)
    return b_p, b_xi, s_xi


# ----------------------------------------------------------------------------
# 3. Physics: the knock-on jump process (compound Poisson)
# ----------------------------------------------------------------------------
def moller_sigma_hat(gamma, cfg: Config):
    """sigma(p) / (2 pi r0^2): the Moller cross-section integrated over BOTH
    outgoing branches gamma_1 in [gamma_m, gamma+1-gamma_m], analytic closed
    form [Embreus 2018, Eq. (35)], with gamma_m = 1 + eps_min.  Zero where the
    channel is closed (gamma - 1 <= 2 eps_min)."""
    gamma = np.asarray(gamma, dtype=np.float64)
    gm = 1.0 + cfg.eps_min
    out = np.zeros_like(gamma)
    ok = gamma - 1.0 > 2.0 * cfg.eps_min
    g = gamma[ok]
    term1 = (0.5 * (g + 1.0) - gm) * (1.0 + 2.0 * g * g / ((g - gm) * (gm - 1.0)))
    term2 = (2.0 * g - 1.0) / (g - 1.0) * np.log((g - gm) / (gm - 1.0))
    out[ok] = (term1 - term2) / (g * g - 1.0)
    return out


def knock_rate(p, cfg: Config):
    """Jump rate lambda(p) in tau_c units.
    lambda = (1/2) n_tot v sigma  ->  R * beta * sigma_hat / (4 lnLambda),
    using n_e c (2 pi r0^2) tau_c = 1/(2 lnLambda)  [tau_c as in Embreus 2018
    Sec. 1] and the factor 1/2 because Eq. (35) integrates over both outgoing
    branches of each (indistinguishable-pair) collision event."""
    if not cfg.jumps_on:
        return np.zeros_like(np.asarray(p, dtype=np.float64))
    p = np.asarray(p, dtype=np.float64)
    gamma = np.sqrt(1.0 + p * p)
    beta = p / gamma
    return cfg.knock_density_ratio * beta * moller_sigma_hat(gamma, cfg) \
        / (4.0 * cfg.lnLambda)


def _moller_diff(gamma, eps):
    """Differential Moller spectrum Sigma(eps) per unit transferred kinetic
    energy eps (compact rearrangement of Embreus 2018, Eq. (5), with
    u = eps (T - eps), T = gamma - 1):
      Sigma = [ gamma^2 T^2 / u^2 - (2 gamma^2 + 2 gamma - 1)/u + 1 ] / (gamma^2 - 1).
    Leading behavior ~ 1/eps^2 (Rutherford-like, heavy-tailed)."""
    T = gamma - 1.0
    u = eps * (T - eps)
    return (gamma * gamma * T * T / (u * u)
            - (2.0 * gamma * gamma + 2.0 * gamma - 1.0) / u
            + 1.0) / (gamma * gamma - 1.0)


def sample_eps(gamma, cfg: Config, rng):
    """Sample the transferred kinetic energy eps in [eps_min, (gamma-1)/2]
    from the Moller spectrum by rejection with the 1/eps^2 envelope
    (envelope constant 4 gamma^2/(gamma^2-1) >= Sigma * eps^2 on the support).
    Vectorized over the (already-fired) particles; gamma is an array."""
    gamma = np.asarray(gamma, dtype=np.float64)
    n = gamma.shape[0]
    eps = np.full(n, cfg.eps_min, dtype=np.float64)
    todo = np.ones(n, dtype=bool)
    for _ in range(64):
        m = int(todo.sum())
        if m == 0:
            break
        g = gamma[todo]
        T = g - 1.0
        # inverse-CDF sample of pdf ~ 1/eps^2 on [eps_min, T/2]
        u = rng.random(m)
        inv = 1.0 / cfg.eps_min - u * (1.0 / cfg.eps_min - 2.0 / T)
        e = 1.0 / inv
        acc = _moller_diff(g, e) * e * e * (g * g - 1.0) / (4.0 * g * g)
        take = rng.random(m) < acc
        idx = np.where(todo)[0]
        eps[idx[take]] = e[take]
        todo[idx[take]] = False
    return eps


def apply_knock_jump(p, xi, cfg: Config, rng):
    """One knock-on event per particle (p, xi arrays of fired particles).
    Since eps <= (gamma-1)/2, the retained electron is always the MORE
    ENERGETIC outgoing branch; the electrons are indistinguishable, so this
    is a relabeling of the leading branch as the continuing particle (not
    literally a tagged incident primary).  The retained branch has
    gamma' = gamma - eps  (stationary target, energy
    conservation gamma_in + 1 = gamma_1' + gamma_2), and is deflected by
      cos(theta_d) = sqrt[ (gamma'-1)(gamma+1) / ((gamma'+1)(gamma-1)) ]
    [Embreus 2018, Eq. (6) applied to the outgoing primary]; the azimuth of
    the deflection is uniform (gyro-phase average).  Returns (p_new, xi_new).
    NOTE: the knocked-on secondary is NOT tracked -- this follows the
    retained leading branch only (a slowing-down/exit model, not an
    avalanche/branching model).  Secondaries can carry up to half the
    primary's kinetic energy (p ~ 5.4 for a primary at p = 10), i.e. some
    omitted secondaries lie in the suprathermal or runaway region."""
    p = np.asarray(p, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    gamma = np.sqrt(1.0 + p * p)
    eps = sample_eps(gamma, cfg, rng)
    g_new = np.maximum(gamma - eps, 1.0 + 1e-12)
    p_new = np.sqrt(g_new * g_new - 1.0)

    cos_th = np.sqrt(np.clip((g_new - 1.0) * (gamma + 1.0)
                             / ((g_new + 1.0) * (gamma - 1.0)), 0.0, 1.0))
    sin_th = np.sqrt(np.clip(1.0 - cos_th * cos_th, 0.0, 1.0))
    phi = rng.uniform(0.0, 2.0 * math.pi, size=p.shape)
    sin_pitch = np.sqrt(np.clip(1.0 - xi * xi, 0.0, 1.0))
    xi_new = np.clip(xi * cos_th + sin_pitch * sin_th * np.cos(phi), -1.0, 1.0)
    return p_new, xi_new


# ----------------------------------------------------------------------------
# 4. The reference numerical method: fine jump-adapted Euler-Maruyama
# ----------------------------------------------------------------------------
def simulate_re2d_step(p0, xi0, cfg: Config, rng, n_sub=None):
    """One coarse step dt of the RE jump-diffusion on [p_min, p_max] x [-1, 1]
    with ABSORBING p-boundaries and REFLECTING xi-boundaries.  Jump thinning
    per sub-step (rate at the pre-jump state), then the drift/diffusion
    Euler-Maruyama sub-step with the pre-jump coefficients -- the same scheme
    as jump1d's simulate_bounded_step, in 2D.

    Returns (p_end, xi_end, alive, side):
      alive[i] False if particle i exited during the step;
      side[i]  0 = still inside, 1 = exited through p_min (thermalized),
               2 = exited through p_max (runaway)."""
    n_sub = cfg.n_sub if n_sub is None else n_sub
    p = np.array(p0, dtype=np.float64).copy()
    xi = np.array(xi0, dtype=np.float64).copy()
    alive = np.ones(p.shape, dtype=bool)
    side = np.zeros(p.shape, dtype=np.int64)
    dt_sub = cfg.dt / n_sub
    sqrt_dt = math.sqrt(dt_sub)

    for _ in range(n_sub):
        a = alive
        if not a.any():
            break
        # coefficients at the pre-jump state
        b_p, b_xi, s_xi = re2d_coeffs(p, xi, cfg)
        # ---- jump sub-step (thinning; rate at the pre-jump state) ----
        if cfg.jumps_on:
            lam = knock_rate(p, cfg)
            fired = a & (rng.random(p.shape) < (1.0 - np.exp(-lam * dt_sub)))
            if fired.any():
                p[fired], xi[fired] = apply_knock_jump(p[fired], xi[fired],
                                                       cfg, rng)
        # ---- drift + diffusion sub-step (only living particles move) ----
        na = int(a.sum())
        p[a] += b_p[a] * dt_sub
        xi[a] += b_xi[a] * dt_sub + s_xi[a] * sqrt_dt * rng.standard_normal(na)
        # ---- reflect xi at +-1 ----
        hi = a & (xi > 1.0)
        xi[hi] = 2.0 - xi[hi]
        lo = a & (xi < -1.0)
        xi[lo] = -2.0 - xi[lo]
        np.clip(xi, -1.0, 1.0, out=xi)
        # ---- absorb at the p boundaries ----
        out_lo = a & (p < cfg.p_min)
        out_hi = a & (p > cfg.p_max)
        side[out_lo] = 1
        side[out_hi] = 2
        alive[out_lo | out_hi] = False
    return p, xi, alive, side


def simulate_re2d_rollout(p0, xi0, cfg: Config, rng, K=None):
    """K coarse steps of the fine integrator (the classical rollout).
    Returns (p, xi, alive, side, survival, frac_lo, frac_hi):
      survival[k]  fraction alive after big step k+1,
      frac_lo/hi[k] cumulative fraction exited through p_min / p_max."""
    K = cfg.rollout_K if K is None else K
    p = np.array(p0, dtype=np.float64).copy()
    xi = np.array(xi0, dtype=np.float64).copy()
    alive = np.ones(p.shape, dtype=bool)
    side = np.zeros(p.shape, dtype=np.int64)
    survival, frac_lo, frac_hi = [], [], []
    for _ in range(K):
        idx = np.where(alive)[0]
        if len(idx):
            pe, xe, al, sd = simulate_re2d_step(p[idx], xi[idx], cfg, rng)
            p[idx] = pe
            xi[idx] = xe
            alive[idx] = al
            side[idx[~al]] = sd[~al]
        survival.append(float(alive.mean()))
        frac_lo.append(float((side == 1).mean()))
        frac_hi.append(float((side == 2).mean()))
    return p, xi, alive, side, survival, frac_lo, frac_hi


# ----------------------------------------------------------------------------
# 4b. Exit-head feature map
# ----------------------------------------------------------------------------
def exit_features(X, cfg: Config):
    """Features for the exit head: (p, xi, log-distance to each absorbing
    wall).  The one-step exit probability has steep fronts near p_min and
    p_max; in log-wall-distance those fronts become mild, learnable slopes
    (this addresses the B1 max-error at the runaway front).  X:(N,2)->(N,4)."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 2)
    d_lo = np.maximum(X[:, 0] - cfg.p_min, 1e-3)
    d_hi = np.maximum(cfg.p_max - X[:, 0], 1e-3)
    return np.stack([X[:, 0], X[:, 1], np.log(d_lo), np.log(d_hi)], axis=1)


# ----------------------------------------------------------------------------
# 5. Utilities
# ----------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)


def get_device(cfg):
    if _HAS_TORCH and cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def wasserstein1(x, y, n_q=4096):
    """1D W1 between two sample sets (quantile average).  The quantile grid
    is capped at n_q points: a midpoint quadrature of int |Q_x - Q_y| dq with
    error ~1/n_q -- far below the metric differences we report -- and ~25x
    faster than evaluating all ~1e5 sample quantiles (W1 and sliced-W1 were
    the wall-clock hotspot of inference once the ground truth is cached)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = min(max(len(x), len(y)), n_q)
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))


def sliced_w1(X, Y, rng, n_proj=64):
    """Sliced W1 between two 2D sample clouds: average 1D W1 over random
    directions, computed in coordinates standardized by the FIRST cloud's
    std (so p and xi contribute comparably)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) == 0 or len(Y) == 0:
        return float("nan")
    s = X.std(axis=0) + 1e-8
    Xs, Ys = X / s, Y / s
    total = 0.0
    for _ in range(n_proj):
        th = rng.uniform(0.0, math.pi)
        d = np.array([math.cos(th), math.sin(th)])
        total += wasserstein1(Xs @ d, Ys @ d)
    return float(total / n_proj)


# ----------------------------------------------------------------------------
# 6. Training-free diffusion engine (same as jump1d, dimension-general)
# ----------------------------------------------------------------------------
# Forward-process schedule:  alpha(0)=1, sigma2(0)=0 (data) ;  ~noise at t=1.
def cond_alpha(t, dt):  return 1.0 - t + dt
def cond_sigma2(t, dt): return t + dt
def drift_f(t, dt):     return -1.0 / cond_alpha(t, dt)
def diff_g2(t, dt):     return 1.0 - 2.0 * drift_f(t, dt) * cond_sigma2(t, dt)


def knn_neighbors(c_sample, c0, k, device):
    """Indices (len(c0), k) of nearest neighbours of c0 within c_sample (L2 in
    conditioning space). FAISS-GPU if available, else a chunked torch fallback."""
    c_sample = np.ascontiguousarray(c_sample, dtype=np.float32)
    c0 = np.ascontiguousarray(c0, dtype=np.float32)
    d = c_sample.shape[1]
    try:
        import faiss
        if device.type == "cuda" and faiss.get_num_gpus() > 0:
            index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d)
        else:
            index = faiss.IndexFlatL2(d)
        index.add(c_sample)
        _, idx = index.search(c0, k)
        return idx
    except Exception:
        if not _HAS_TORCH:
            raise
        cs = torch.tensor(c_sample, device=device)
        out = np.empty((len(c0), k), dtype=np.int64)
        bs = 1024
        for i in range(0, len(c0), bs):
            cb = torch.tensor(c0[i:i + bs], device=device)
            dist = torch.cdist(cb, cs)
            out[i:i + bs] = torch.topk(dist, k, largest=False).indices.cpu().numpy()
        return out


def ode_solve(zt, neigh, ode_steps):
    """Probability-flow ODE, batched and reusing neighbours.
    zt:(B,d) latent;  neigh:(B,k,d) neighbour TARGETS (scaled increments)."""
    device = zt.device
    t_vec = torch.linspace(1.0, 0.0, ode_steps + 1, device=device)
    for j in range(ode_steps):
        t = t_vec[j + 1]; dt = t_vec[j] - t_vec[j + 1]
        a = cond_alpha(t, dt); s2 = cond_sigma2(t, dt)
        diff = zt[:, None, :] - a * neigh                 # (B,k,d)
        logw = -0.5 * torch.sum(diff ** 2, dim=2) / s2    # (B,k)
        w = torch.softmax(logw, dim=1)                    # (B,k)
        score = torch.sum((-diff / s2) * w[:, :, None], dim=1)  # (B,d)
        zt = zt - (drift_f(t, dt) * zt - 0.5 * diff_g2(t, dt) * score) * dt
    return zt


def generate_labels(c_sample, target_sample, cfg, rng, device):
    """Make (c0, zT) -> target labels via the training-free probability-flow
    ODE.  c_sample:(N,d_c) conditioning;  target_sample:(N,d_y) the scaled
    increments to learn.  The KNN search runs in STANDARDIZED conditioning
    coordinates so that p (range ~9) and xi (range 2) contribute comparably.
    Returns c0:(B,d_c), zT:(B,d_y), y_gen:(B,d_y)."""
    N = c_sample.shape[0]
    d_y = target_sample.shape[1]
    B = min(cfg.train_size_labels, N)
    sel = rng.permutation(N)[:B]
    c0 = c_sample[sel]

    cm = c_sample.mean(axis=0, keepdims=True)
    cs = c_sample.std(axis=0, keepdims=True) + 1e-8
    idx = knn_neighbors((c_sample - cm) / cs, (c0 - cm) / cs,
                        cfg.knn_k, device)                  # (B,k)
    neigh_all = target_sample[idx]                          # (B,k,d_y)

    zT = rng.standard_normal((B, d_y)).astype(np.float32)
    y_gen = np.empty((B, d_y), dtype=np.float32)
    for s in range(0, B, cfg.label_chunk):
        e = min(s + cfg.label_chunk, B)
        zt = torch.tensor(zT[s:e], device=device)
        neigh = torch.tensor(neigh_all[s:e], dtype=torch.float32, device=device)
        y_gen[s:e] = ode_solve(zt, neigh, cfg.ode_steps).cpu().numpy()
    return c0.astype(np.float32), zT, y_gen


# ----------------------------------------------------------------------------
# 7. Neural networks
# ----------------------------------------------------------------------------
if _HAS_TORCH:

    class FN_Net(nn.Module):
        """Distilled generator G(x, z) -> increment.  Same architecture as the
        JCP train_NN.py: two hidden tanh layers.  Here input 4 = (p, xi, z1, z2),
        output 2 = the scaled (dp, dxi)."""
        def __init__(self, input_dim, output_dim, hid_size):
            super().__init__()
            self.input = nn.Linear(input_dim, hid_size)
            self.fc1 = nn.Linear(hid_size, hid_size)
            self.output = nn.Linear(hid_size, output_dim)

        def forward(self, x):
            x = torch.tanh(self.input(x))
            x = torch.tanh(self.fc1(x))
            return self.output(x)

    class ExitNet(nn.Module):
        """3-class exit head: logits for (stay, exit through p_min, exit
        through p_max) during one big step.  Same body as the JCP
        train_escape_binary EscapeModel, with a softmax-readout instead of a
        sigmoid so the EXIT SIDE (thermalization vs runaway) is predicted.
        Inputs are the 4 exit_features (p, xi, log wall distances); dropout
        is 0.1 -- heavier dropout biases the estimator toward smoothness,
        which is exactly wrong on the steep exit fronts."""
        def __init__(self, cfg: Config):
            super().__init__()
            h = cfg.exit_hidden
            self.net = nn.Sequential(
                nn.Linear(4, h), nn.LeakyReLU(0.01), nn.Dropout(0.1),
                nn.Linear(h, h), nn.LeakyReLU(0.01), nn.Dropout(0.1),
                nn.Linear(h, h), nn.LeakyReLU(0.01), nn.Dropout(0.1),
                nn.Linear(h, 3))

        def forward(self, x):
            return self.net(x)      # logits (N,3)
