"""Shared helpers for step1_reproduce.py and step2_noise.py.

Everything that the authors provide (network generation, reciprocity control, spectral
metrics, MC, KR) is called from their code via authors_code.load().  The only thing
implemented here is `mc_curve()` -- a small reservoir used in Step 2 because the authors'
compute_memory_capacity() returns only the summed MC (no per-delay R^2) and uses uniform,
non-Gaussian noise.  mc_curve() is validated against the authors' function at sigma = 0
(see validate_against_authors()).
"""
import json, os
import numpy as np
from numba import njit
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

# ---- fixed experiment constants (paper Methods 4.1-4.7 unless noted) ---------------------
N, K = 64, 6
HM_LEVEL, HM_E, HM_SZ = 6, 4.0, 4          # 2**6 = 64 nodes; E = 4.0; cluster-size parameter s = 4
T_STEPS = 3200
TRAIN_FRAC = 0.7
W_IN_SCALE = 0.05
RIDGE = 1e-5
N_TRANSIENT = 100                          # authors' default (metrics.compute_memory_capacity)
NUM_DELAYS = int(np.floor(1.4 * N))        # 89
BASE_SEED = 20260101


# ---- seeding -----------------------------------------------------------------------------
@njit(cache=True)
def seed_numba(s):
    """numba keeps its own RNG; echoes' state noise draws from it. Must be seeded explicitly."""
    np.random.seed(s)


def net_seed(kind, instance, r_index=0):
    """Deterministic integer seed for (network kind, instance, reciprocity level)."""
    kind_id = {"HM": 1, "random": 2, "rho_sweep": 3}[kind]
    return BASE_SEED + kind_id * 10_000_000 + instance * 100 + r_index


# ---- networks (all construction by authors' functions) -----------------------------------
def make_base_network(U, kind, instance):
    """kind='HM' -> hierarchical modular; kind="random" -> directed k-out random graph."""
    s = net_seed(kind, instance)
    if kind in ("HM", "rho_sweep"):
        return U.generate_hierarchical_modular(HM_LEVEL, HM_E, HM_SZ, desired_links=N * K, rand_seed=s)
    return U.generate_random_directed_graph(N, K, seed=s)


def with_reciprocity(U, base, r, kind, instance, r_index):
    """Authors' binary NRC on a copy of `base` (uses global np.random -> seed it first)."""
    np.random.seed(net_seed(kind, instance, r_index))
    A = U.adjust_reciprocity_binary(base.copy(), r)
    return A.astype(float)


def spectral_descriptors(U, A):
    """raw spectral radius, spectral gap |l1|-|l2|, normalised Henrici index d_F/||W||_F."""
    h = float(U.departure_from_normality(A))
    if not np.isfinite(h):
        # The authors' function takes sqrt(||W||_F^2 - sum|lambda|^2); for (numerically) normal
        # matrices, e.g. symmetric networks at r = 1, that radicand is a tiny NEGATIVE number -> NaN.
        # True value is 0.  Same formula with the radicand clipped at 0:
        nf2 = float(np.linalg.norm(A, "fro") ** 2)
        lam2 = float(np.sum(np.abs(np.linalg.eigvals(A)) ** 2))
        h = float(np.sqrt(max(nf2 - lam2, 0.0) / nf2))
    return dict(
        spectral_radius_raw=float(U.find_spectral_radius(A)),
        spectral_gap=float(U.compute_eigenvalue_gap(A)),
        henrici_index=h,
    )


# ---- reservoir with per-delay readout (Step 2) -------------------------------------------
@njit(cache=True)
def _harvest(W, w_in, u, sigma, seed):
    """x_t = tanh(W x_{t-1} + w_in u_t) + sigma * xi_t,  xi ~ N(0,1);  x_0 = 0, u_0 unused.
    (Same update order as echoes.harvest_states: loop starts at t=1; noise added after tanh.)"""
    np.random.seed(seed)
    T, n = u.shape[0], W.shape[0]
    X = np.zeros((T, n))
    for t in range(1, T):
        pre = W @ X[t - 1] + w_in * u[t]
        x = np.tanh(pre)
        if sigma > 0.0:
            x = x + sigma * np.random.standard_normal(n)
        X[t] = x
    return X


def scale_to_radius(A, rho):
    """Same operation as echoes.utils.set_spectral_radius."""
    return A * (rho / np.max(np.abs(np.linalg.eigvals(A))))


def mc_curve(A, rho, inst_seed, sigma, noise_seed=None, sigma_test=None, readout_input=False, ridge=RIDGE, T=T_STEPS,
             train_frac=TRAIN_FRAC, w_in_scale=W_IN_SCALE, n_transient=N_TRANSIENT):
    """Per-delay R^2_n (n = 1..1.4N) for one network. Noise (std sigma) acts in training AND test.

    u and W_in come from default_rng(inst_seed) in the same order as the authors' function
    (W_in first, then u), so that sigma = 0 reproduces compute_memory_capacity() exactly.
    Returns raw (unclipped) sklearn-style R^2 per delay; MC = sum(max(R^2, 0)) as in the authors' code."""
    rng = np.random.default_rng(inst_seed)
    n = A.shape[0]
    w_in = rng.uniform(-w_in_scale, w_in_scale, size=n)
    u = rng.standard_normal(T)
    nd = int(np.floor(1.4 * n))
    Y = np.zeros((T, nd))
    for lag in range(1, nd + 1):
        Y[lag:, lag - 1] = u[:-lag]
    split = int(np.floor(train_frac * T))
    W = scale_to_radius(A, rho)
    ns = inst_seed if noise_seed is None else noise_seed
    Xtr = _harvest(W, w_in, u[:split], sigma, ns)              # training run (own noise stream)
    st = sigma if sigma_test is None else sigma_test                # default: noise ON at test too
    Xte = _harvest(W, w_in, u[split:], st, ns + 1)             # test run: fresh state
    if readout_input:  # paper Eq.: readout on [x(t); u(t)] (authors' code fits on states only)
        Xtr = np.column_stack([Xtr, u[:split]]); Xte = np.column_stack([Xte, u[split:]])
    Xa, Ya = Xtr[n_transient:], Y[:split][n_transient:]
    G = Xa.T @ Xa + ridge * np.eye(Xa.shape[1])                          # ridge, no intercept (echoes default)
    Wout = np.linalg.solve(G, Xa.T @ Ya)
    off = n_transient + 1
    Yp = (Xte @ Wout)[off:]
    Yt = Y[split:][off:]
    # sklearn r2_score per column, vectorised: 1 - SS_res / SS_tot
    ss_res = ((Yt - Yp) ** 2).sum(0)
    ss_tot = ((Yt - Yt.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot


def mc_from_curve(r2):
    return float(np.maximum(r2, 0).sum())


# ---- statistics --------------------------------------------------------------------------
def linfit(x, y):
    res = stats.linregress(x, y)
    return dict(slope=float(res.slope), intercept=float(res.intercept),
                r2=float(res.rvalue ** 2), p=float(res.pvalue), n=int(len(x)))


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2)


def ridx(r):
    """Reciprocity-level index used in seeds. For r in {0,.1,...,1} equals Step 1's index, so
    Step 2 re-uses exactly the same networks as Step 1 at r = 0, 0.5, 1."""
    return int(round(r * 10)) if abs(r * 10 - round(r * 10)) < 1e-9 else 100 + int(round(r * 100))
