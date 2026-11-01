# REPORT

All numbers below come from `results/`. Seeds are fixed (`common.net_seed`). Python 3.12.3; versions in `requirements.txt`.

## Step 0 - code report (authors' commits: metrics repo `6ca9d1f`, NRC repo `8a9b081`)

| Need | Function (file) |
|---|---|
| (a) HM network | `generate_hierarchical_modular(mx_lvl, E, sz_cl, desired_links=None, rand_seed=None)` (metrics repo `utils.py`). The authors' own variant of Sporns' recursive algorithm that samples a fixed number of links. Used with `mx_lvl=6, E=4.0, sz_cl=4, desired_links=384`. |
| (b) k-out null | `generate_random_directed_graph(n, out_degree, seed=None)` |
| (c) binary reciprocity | `adjust_reciprocity_binary(adj_matrix, desired_reciprocity)`; `compute_reciprocity_binary(adjmatrix)`. Modifies its input in place, uses the global `np.random`. Identical in both repos. |
| (d) log-normal weights | **Not found in either repo.** Nearest: `random_weight_assignment` (uniform) and `convert_binary_to_weighted_matrix` (uniform 0.1-1). Not needed for the binary Step 1. |
| (e) strength reciprocity | `adjust_reciprocity_weighted(adj_matrix, desired_reciprocity, num_iter=2000)` (differs between repos; unused here) |
| (f) MC | `compute_memory_capacity(W, spectral_radius=0.95, sequence_length=3000, n_transient=100, w_in_scale=0.05, ridge_alpha=1e-10, noise=1e-5, train_fraction=0.7, random_state=283)` (`metrics.py`) |
| (g) KR | `compute_kernel_rank(W, spectral_radius=0.95, num_inputs=None, signal_length=500, n_transient=50, w_in_scale=0.05, svd_threshold=0.01, noise=1e-5, random_state=283)` |
| (h) reservoir | No standalone function. MC/KR wrap `echoes.ESNRegressor` / `echoes.reservoir._leaky_numba.harvest_states`. |
| descriptors | `find_spectral_radius`, `compute_eigenvalue_gap`, `departure_from_normality` (`utils.py`) |

**Behaviour of the authors' code that differs from the task/paper text** (I followed the code):
- echoes always rescales W to `spectral_radius` (contrary to a comment in `metrics.py`).
- State noise is **uniform** in (-0.5, 0.5)·`noise` (std ≈ 2.9e-5 for 1e-4), added **after** tanh, and is also active in `predict()`, i.e. in testing (checked empirically). The paper describes Gaussian noise during training.
- The MC readout uses states only (`fit_only_states=True`), not [x; u]. Per-delay R² is clipped at 0. Default ridge is 1e-10 and T = 3000; I pass 1e-5 and 3200.
- The KR docstring says ρ = 0.99 and signal length 200; the signature has 0.95 and 500.
- bctpy's `makefractalCIJ(6, E=4, s=4)` gives ~1343 links (k ≈ 21), so the paper's literal HM parameters cannot give k = 6 through bctpy alone.

**Problems patched in memory (`authors_code.py`)**: (P1) `metrics.py` does not parse as published (unclosed docstring, split `W_in =` line, `W.float()` on numpy arrays); (P2) echoes 0.0.8 calls `Ridge(normalize=...)`, removed in scikit-learn 1.2 (the authors' pinned 1.3.2 fails too); (P3, in `common.py`) numba's RNG is unseeded (seeded explicitly) and `departure_from_normality` returns NaN for symmetric r = 1 networks because of a tiny negative radicand (clipped at 0; 256 rows repaired). Nothing else in the authors' numerics was altered.

I wrote one thing myself: `common.mc_curve` (per-delay R², Gaussian noise), used only in Step 2 and `diagnose_step1.py`, because the authors' MC returns only the sum. It matches their function at σ = 0 to ~1e-7.

## Step 1 - reproduction

Setup: 50 HM instances × 11 reciprocity levels; 250 random k-out nulls (5 per instance), NRC applied to nulls as well (my assumption). ρ = 0.95 from a sweep over {0.8, 0.9, 0.95, 0.99} on 20 separate r = 0 networks (mean MC 7.59 / 8.37 / 8.73 / 8.68). Achieved reciprocity within 0.005 of target; 382-384 links (target 384).

| Fit vs r (per unit r) | slope | R² | p | n |
|---|---|---|---|---|
| HM MC, level means | -0.980 | 0.987 | 7.6e-10 | 11 |
| HM KR, level means | -0.500 | 0.951 | 3.3e-07 | 11 |
| HM MC, all instances | -0.980 | 0.426 | 3.6e-68 | 550 |
| HM KR, all instances | -0.500 | 0.147 | 9.7e-21 | 550 |
| random MC / KR, level means | -0.775 / -0.527 | 0.988 / 0.919 | | 11 |
| **paper** MC / KR | -1.26 / -0.45 | 0.995 / 0.882 | <0.001 | |

**Verdict: partial reproduction.** Linear decline with reciprocity, significance, and HM > null are reproduced. The KR slope is close (0.05 steeper). The MC slope is about 22% shallower than the paper's. The paper's units match mine (change per unit reciprocity, 0 to 1).

**Post-hoc diagnostic** (`diagnose_step1.py`, HM, same instances, my validated reservoir): MC slope was -0.91 to -1.04 across noise protocols (none; Gaussian 1e-4 train only; train+test), readout with/without u(t), and ρ ∈ {0.95, 0.99}. None approaches -1.26, so these choices do not explain the gap.

**Untested candidates:** the paper averages 5 training trials per instance (I use one, which may also explain the lower R²); the paper's "other parameters optimised for MC" and exact HM tuning of E and s; differences between the paper's and the repository's generator; number of delays or train/test details.

## Step 2 - PRELIMINARY observations

Design as specified: r ∈ {0, .25, .5, .75, 1}, σ ∈ {0, 1e-4, ..., 1e-1}, Gaussian noise after tanh in training **and** testing, same networks as Step 1 (identical at r = 0, 0.5, 1), 50 HM + 250 null networks per r. The noise stream is shared across σ within a network. State RMS without noise is ~0.033 at every r (HM 0.0341 at r = 0 to 0.0327 at r = 1), so σ corresponds to a similar signal-to-noise ratio across r.

- **MC vs σ.** MC decreased with σ for every r and both network types. Lower r gave higher MC at every σ. HM paired difference (r = 0 minus r = 1): 0.93 at σ = 0 and 0.11 at σ = 0.1 (SE ≤ 0.06). The ratio of mean MC (r = 0 / r = 1) rose from 1.12 to 1.38.
- **Fraction of noise-free MC retained** (HM) was similar across r, slightly higher at low r: 0.595 (r = 0) vs 0.571 (r = 1) at σ = 1e-3, and 0.307 vs 0.290 at σ = 1e-2. In this setting I see no sign that low reciprocity costs proportionally more MC under noise; in absolute terms low-r networks lose somewhat more because they start higher.
- **Sensitivity to small noise is large:** MC at σ = 1e-4 is already ~13% below σ = 0 although σ is ~0.3% of the state RMS. I suspect the weakly regularised readout (γ = 1e-5) but did not test it.
- **β (log σ*ₙ ~ β log n).** Only delays 1-9 have noise-free R² ≥ 0.2, so each fit uses about 7-9 delays spanning under one decade. β ≈ -2.7 to -3.2 throughout. In the main analysis HM β is -2.995 (r = 0), -3.089, -2.999, -2.875, -2.749 (r = 1); 95% bootstrap CIs are about ±0.03-0.05. The number of usable delays differs by r (7.6 to 9.5) because the longer delays are left-censored at low r. With a common window n ≤ 6, HM β runs -2.550, -2.710, -2.851, -2.937, -2.962, i.e. the **opposite** trend. I therefore draw no conclusion about the direction of the r-dependence of β.
- **Correlations (HM, across 250 networks).** Spearman of the Henrici index with MC(σ) is 0.51-0.68 and with β is -0.52, but after controlling for r (partial rank correlation) it is -0.06 to -0.04 (p > 0.3) for MC and -0.03 for β. Spectral gap: raw 0.16-0.22, partial -0.13 to -0.25 (p < 0.04) for MC. Reciprocity itself drives the Henrici/gap association; with one topology family these data cannot separate the two.

**What this does NOT show:** anything beyond a single topology family, a single size (N = 64, k = 6), binary weights, a linear memory task and one readout regularisation; nothing about nonlinear tasks, other noise types (e.g. input noise), larger networks, or biological relevance. The β results are not robust to the delay window.

## Every guess or deviation

1. HM generator = authors' `desired_links` variant (bctpy cannot reach k = 6 at E = 4, s = 4).
2. NRC also applied to random nulls; 5 nulls per HM instance.
3. ρ = 0.95 from my sweep (also the code default); KR docstring says 0.99.
4. Step 1 noise/readout/clipping/transient follow the authors' code, not the paper text (see Step 0).
5. T = 3200 and ridge 1e-5 passed explicitly (code defaults differ); one input seed per network (paper: 5 trials).
6. Primary fits on the 11 level means; all-instance fits also reported. Extra CSV columns: `achieved_reciprocity`, `n_links`, `rho_used`.
7. Step 2 uses my reservoir (Gaussian, after tanh, shared noise stream across σ). Analysis choices: R² floor 0.2, linear interpolation of R² vs log σ, first crossing, censored delays dropped, per-network β needs ≥ 5 delays, 5000-resample bootstrap over networks.
8. Python 3.12 and newer package versions than the authors' pins (their numpy 1.24.4 / numba 0.58.1 etc. have no 3.12 wheels).
9. Tested: clean-venv reduced run (3 HM instances) of both scripts. Not tested: Python 3.10/3.11, full-size run in a clean environment.
