#!/usr/bin/env python
"""Step 1 -- reproduce MC / KR vs link reciprocity for binary HM networks (N=64, k=6).

All network generation, reciprocity control, MC and KR are the AUTHORS' functions
(see authors_code.py and REPORT.md, Step 0).  Resumable: re-running skips finished rows.

    python step1_reproduce.py                 # full run (50 HM instances, 5 nulls each)
    python step1_reproduce.py --instances 20 --nulls 2    # reduced run
"""
import argparse, csv, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import authors_code, common as C

R_LEVELS = [round(0.1 * i, 1) for i in range(11)]
RHO_GRID = [0.8, 0.9, 0.95, 0.99]
STATE_NOISE = 1e-4          # passed to the authors' `noise=`; echoes draws it as uniform(-.5,.5)*noise
PAPER = {"MC": {"slope": -1.26, "r2": 0.995}, "KR": {"slope": -0.45, "r2": 0.882}}
COLS = ["network_type", "instance", "seed", "reciprocity", "MC", "KR", "spectral_radius_raw",
        "spectral_gap", "henrici_index", "achieved_reciprocity", "n_links", "rho_used"]


def choose_rho(U, M, n_sweep, forced):
    path = os.path.join(C.RESULTS, "step1_rho_sweep.csv")
    if forced is not None:
        print(f"[rho] using --rho {forced} (no sweep)")
        return forced
    choice = os.path.join(C.RESULTS, "step1_rho_choice.json")
    if os.path.exists(choice):
        import json
        rho = json.load(open(choice))["rho"]
        print(f"[rho] reusing earlier sweep result rho = {rho}")
        return rho
    print(f"[rho] sweeping {RHO_GRID} on {n_sweep} separate r=0 HM networks (MC only)")
    rows = []
    for i in range(n_sweep):
        base = C.make_base_network(U, "rho_sweep", i)
        A = C.with_reciprocity(U, base, 0.0, "rho_sweep", i, 0)
        for rho in RHO_GRID:
            C.seed_numba(C.net_seed("rho_sweep", i, 0))
            mc = M.compute_memory_capacity(A, spectral_radius=rho, sequence_length=C.T_STEPS,
                                           ridge_alpha=C.RIDGE, noise=STATE_NOISE,
                                           random_state=C.net_seed("rho_sweep", i, 0))
            rows.append(dict(instance=i, rho=rho, MC=mc))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    summ = df.groupby("rho").MC.agg(["mean", "std"])
    print(summ.round(3).to_string())
    rho = float(summ["mean"].idxmax())
    C.save_json({"rho": rho, "grid": RHO_GRID, "mean_MC": summ["mean"].to_dict(),
                 "note": "best mean MC on r=0 HM networks (instances disjoint from the main run)"},
                "step1_rho_choice.json")
    print(f"[rho] chosen rho = {rho}")
    return rho


def run_one(U, M, kind, inst, base, ri, r, rho):
    A = C.with_reciprocity(U, base, r, kind, inst, ri)
    seed = C.net_seed(kind, inst, 0)
    C.seed_numba(C.net_seed(kind, inst, ri))
    mc = M.compute_memory_capacity(A, spectral_radius=rho, sequence_length=C.T_STEPS,
                                   ridge_alpha=C.RIDGE, noise=STATE_NOISE, random_state=seed)
    C.seed_numba(C.net_seed(kind, inst, ri) + 1)
    kr = M.compute_kernel_rank(A, spectral_radius=rho, noise=STATE_NOISE, random_state=seed)
    d = C.spectral_descriptors(U, A)
    return dict(network_type=kind, instance=inst, seed=seed, reciprocity=r, MC=mc, KR=kr, **d,
                achieved_reciprocity=float(U.compute_reciprocity_binary(A)),
                n_links=int(A.sum()), rho_used=rho)


def fits(df):
    out = {}
    for kind in ["HM", "random"]:
        d = df[df.network_type == kind]
        for metric in ["MC", "KR"]:
            means = d.groupby("reciprocity")[metric].mean()
            out[f"{kind}_{metric}_level_means"] = C.linfit(means.index.values, means.values)
            out[f"{kind}_{metric}_all_instances"] = C.linfit(d.reciprocity.values, d[metric].values)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=50)
    ap.add_argument("--nulls", type=int, default=5, help="null networks per structured instance")
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--sweep-instances", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(C.RESULTS, exist_ok=True)
    U, M = authors_code.load()
    rho = choose_rho(U, M, a.sweep_instances, a.rho)

    out = os.path.join(C.RESULTS, "step1_mc_kr.csv")
    done = set()
    if os.path.exists(out):
        old = pd.read_csv(out)
        done = set(zip(old.network_type, old.instance, old.reciprocity.round(1)))
        print(f"[resume] {len(done)} rows already done")
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, COLS).writeheader()

    jobs = []
    for i in range(a.instances):
        jobs.append(("HM", i))
        jobs += [("random", i * a.nulls + j) for j in range(a.nulls)]
    todo = [(k, i) for k, i in jobs if any((k, i, r) not in done for r in R_LEVELS)]

    # runtime estimate from 3 timed networks
    if todo:
        t0 = time.time()
        for k, i in todo[:1]:
            base = C.make_base_network(U, k, i)
            for ri in (0, 5, 10):
                run_one(U, M, k, i, base, ri, R_LEVELS[ri], rho)
        per = (time.time() - t0) / 3
        print(f"[runtime] ~{per:.2f} s per network; {len(todo)*len(R_LEVELS)} networks to run "
              f"-> estimated {per*len(todo)*len(R_LEVELS)/60:.0f} min")

    t0 = time.time()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        for n, (k, i) in enumerate(todo):
            base = C.make_base_network(U, k, i)
            for ri, r in enumerate(R_LEVELS):
                if (k, i, r) in done:
                    continue
                w.writerow(run_one(U, M, k, i, base, ri, r, rho))
            f.flush()
            if n % 25 == 0:
                print(f"  {n+1}/{len(todo)} network families, {(time.time()-t0)/60:.1f} min elapsed", flush=True)

    df = pd.read_csv(out)
    bad = df.henrici_index.isna()
    if bad.any():   # authors' departure_from_normality returns NaN for symmetric (r = 1) networks
        print(f"[repair] recomputing Henrici index (radicand clipped at 0) for {int(bad.sum())} rows")
        for idx in df.index[bad]:
            k, i, r = df.at[idx, "network_type"], int(df.at[idx, "instance"]), float(df.at[idx, "reciprocity"])
            A = C.with_reciprocity(U, C.make_base_network(U, k, i), r, k, i, int(round(r * 10)))
            df.at[idx, "henrici_index"] = C.spectral_descriptors(U, A)["henrici_index"]
        df.to_csv(out, index=False)
    res = fits(df)
    res["rho"] = rho
    res["n_HM_instances"] = int(df[df.network_type == "HM"].instance.nunique())
    res["n_random_instances"] = int(df[df.network_type == "random"].instance.nunique())
    res["paper"] = PAPER
    C.save_json(res, "step1_fits.json")
    print("\n=== Linear fits vs link reciprocity (slope per unit r in [0,1]) ===")
    for key, v in res.items():
        if isinstance(v, dict) and "slope" in v:
            print(f"{key:28s} slope={v['slope']:+.3f}  R2={v['r2']:.3f}  p={v['p']:.2g}  n={v['n']}")
    print(f"paper: MC slope {PAPER['MC']['slope']} (R2 {PAPER['MC']['r2']}); "
          f"KR slope {PAPER['KR']['slope']} (R2 {PAPER['KR']['r2']})")


if __name__ == "__main__":
    main()
