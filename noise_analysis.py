"""Analysis helpers for Step 2 (used by step2_noise.py and analysis.ipynb).

Definitions (all choices below are mine, not the paper's -- see REPORT.md):
  * delay n is 'eligible' for a network if its noise-free R^2_n >= R2_FLOOR (otherwise '50 % of a
    value near zero' is meaningless);
  * sigma*_n = smallest tested sigma>0 where R^2_n < 0.5 * R^2_n(sigma=0); linear interpolation of
    R^2_n against log10(sigma) between the bracketing grid points;
  * a delay is left-censored if already below 50 % at the smallest sigma>0, right-censored if it
    never falls below 50 % up to the largest sigma; censored delays are dropped from the beta fit;
  * optional max_delay restricts the fit to delays n <= max_delay (sensitivity check: a common delay
    range for all r removes the r-dependent left-censoring of the longer delays);
  * beta = slope of log10(sigma*) vs log10(n) over uncensored eligible delays (>= MIN_DELAYS).
"""
import numpy as np, pandas as pd
from scipy import stats

R2_FLOOR = 0.2
MIN_DELAYS = 5


def sigma_star(r2_by_sigma, sigmas, r2_zero):
    """r2_by_sigma: R^2_n at each sigma in `sigmas` (all > 0, ascending). Returns (sigma*, status)."""
    thr = 0.5 * r2_zero
    below = np.where(r2_by_sigma < thr)[0]
    if len(below) == 0:
        return np.nan, "right"
    j = below[0]
    if j == 0:
        return np.nan, "left"
    x0, x1 = np.log10(sigmas[j - 1]), np.log10(sigmas[j])
    y0, y1 = r2_by_sigma[j - 1], r2_by_sigma[j]
    xs = x0 + (thr - y0) * (x1 - x0) / (y1 - y0)
    return 10 ** xs, "ok"


def network_beta(r2_matrix, sigmas, delays, max_delay=None):
    """r2_matrix: (n_sigma_total, n_delays) rows ordered as [0, sigmas...]. Returns dict."""
    r2_0, r2_s = r2_matrix[0], r2_matrix[1:]
    sig = np.asarray(sigmas[1:], float)
    ns, ss, stat = [], [], {"ok": 0, "left": 0, "right": 0}
    for i, n in enumerate(delays):
        if r2_0[i] < R2_FLOOR or (max_delay is not None and n > max_delay):
            continue
        s, st = sigma_star(r2_s[:, i], sig, r2_0[i])
        stat[st] += 1
        if st == "ok":
            ns.append(n); ss.append(s)
    out = dict(n_eligible=sum(stat.values()), n_ok=stat["ok"], n_left=stat["left"], n_right=stat["right"],
               beta=np.nan, beta_intercept=np.nan)
    if len(ns) >= MIN_DELAYS and len(set(np.round(np.log10(ss), 6))) > 1:
        res = stats.linregress(np.log10(ns), np.log10(ss))
        out.update(beta=res.slope, beta_intercept=res.intercept)
    return out


def bootstrap_ci(x, n_boot=5000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(x, len(x))) for _ in range(n_boot)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def compute_beta_table(df, sigmas, max_delay=None):
    """df: step2_noise.csv. One row per network (type, instance, reciprocity)."""
    r2cols = [c for c in df.columns if c.startswith("R2_")]
    delays = np.array([int(c[3:]) for c in r2cols])
    rows = []
    for (kind, inst, r), g in df.groupby(["network_type", "instance", "reciprocity"]):
        g = g.set_index("sigma").loc[sigmas]
        b = network_beta(g[r2cols].values, sigmas, delays, max_delay)
        first = g.iloc[0]
        rows.append(dict(network_type=kind, instance=inst, reciprocity=r, **b,
                         spectral_radius_raw=first.spectral_radius_raw, spectral_gap=first.spectral_gap,
                         henrici_index=first.henrici_index))
    return pd.DataFrame(rows)


def beta_summary(beta_df):
    rows = []
    for (kind, r), g in beta_df.groupby(["network_type", "reciprocity"]):
        lo, hi = bootstrap_ci(g.beta.values)
        rows.append(dict(network_type=kind, reciprocity=r, n_networks=len(g),
                         n_beta_defined=int(g.beta.notna().sum()), beta_mean=g.beta.mean(),
                         beta_sd=g.beta.std(), ci_low=lo, ci_high=hi,
                         mean_delays_used=g.n_ok.mean(), mean_left_censored=g.n_left.mean(),
                         mean_right_censored=g.n_right.mean()))
    return pd.DataFrame(rows)


def _partial_spearman(x, y, z):
    """Spearman correlation of x and y after regressing out rank(z) from both rank-transforms."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a):
        A = np.column_stack([np.ones_like(rz), rz])
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    return stats.pearsonr(resid(rx), resid(ry))


def correlations(noise_df, beta_df, sigmas):
    """Spearman (and partial, controlling for reciprocity) of MC(sigma) and beta with Henrici / gap."""
    rows = []
    key = ["network_type", "instance", "reciprocity"]
    m = noise_df.merge(beta_df[key + ["beta"]], on=key)
    for scope in ["HM", "random", "all"]:
        d = m if scope == "all" else m[m.network_type == scope]
        for target in ["MC"] + ["beta"]:
            sigs = sigmas if target == "MC" else [None]
            for s in sigs:
                dd = d[d.sigma == (s if s is not None else sigmas[0])].dropna(subset=[target])
                for pred in ["henrici_index", "spectral_gap"]:
                    if len(dd) < 5:
                        continue
                    rho, p = stats.spearmanr(dd[pred], dd[target])
                    prho, pp = _partial_spearman(dd[pred].values, dd[target].values, dd["reciprocity"].values)
                    rows.append(dict(scope=scope, target=target, sigma=s if s is not None else np.nan,
                                     predictor=pred, n=len(dd), spearman=rho, p=p,
                                     partial_spearman_given_r=prho, partial_p=pp))
    return pd.DataFrame(rows)
