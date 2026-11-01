#!/usr/bin/env python
"""Step 2 (PRELIMINARY extension) -- MC and per-delay R^2 vs state-noise level, by reciprocity.

Same HM / random k-out networks as Step 1 (identical seeds) at r in {0,.25,.5,.75,1}, same
spectral radius (read from results/step1_rho_choice.json), input scaling, ridge and T.
Gaussian state noise of std sigma is added to the reservoir update in BOTH training and testing.

    python step2_noise.py [--instances 50 --nulls 5]
"""
import argparse, csv, json, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import authors_code, common as C, noise_analysis as NA

R_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
SIGMAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
DELAYS = list(range(1, C.NUM_DELAYS + 1))
COLS = (["network_type", "instance", "seed", "reciprocity", "sigma", "MC", "spectral_radius_raw",
         "spectral_gap", "henrici_index", "rho_used"] + [f"R2_{n}" for n in DELAYS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=50)
    ap.add_argument("--nulls", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(C.RESULTS, exist_ok=True)
    rho = json.load(open(os.path.join(C.RESULTS, "step1_rho_choice.json")))["rho"]
    U, _ = authors_code.load()

    out = os.path.join(C.RESULTS, "step2_noise.csv")
    done = set()
    if os.path.exists(out):
        old = pd.read_csv(out, usecols=["network_type", "instance", "reciprocity", "sigma"])
        done = set(zip(old.network_type, old.instance, old.reciprocity, old.sigma))
        print(f"[resume] {len(done)} rows already done")
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, COLS).writeheader()

    jobs = []
    for i in range(a.instances):
        jobs.append(("HM", i))
        jobs += [("random", i * a.nulls + j) for j in range(a.nulls)]
    total = len(jobs) * len(R_LEVELS) * len(SIGMAS)
    t0 = time.time()
    C.mc_curve(np.ones((64, 64)) - np.eye(64), rho, 1, 0.0)  # trigger numba compile before timing
    print(f"[runtime] {total} reservoir runs at ~12 ms each -> ~{total*0.012/60:.0f} min expected")

    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        for n, (kind, inst) in enumerate(jobs):
            base = C.make_base_network(U, kind, inst)
            inst_seed = C.net_seed(kind, inst, 0)
            for r in R_LEVELS:
                ri = C.ridx(r)
                A = C.with_reciprocity(U, base, r, kind, inst, ri)
                desc = C.spectral_descriptors(U, A)
                for s in SIGMAS:
                    if (kind, inst, r, s) in done:
                        continue
                    c = C.mc_curve(A, rho, inst_seed, s, noise_seed=C.net_seed(kind, inst, ri) + 17)
                    row = dict(network_type=kind, instance=inst, seed=inst_seed, reciprocity=r, sigma=s,
                               MC=C.mc_from_curve(c), rho_used=rho, **desc)
                    row.update({f"R2_{d}": float(v) for d, v in zip(DELAYS, c)})
                    w.writerow(row)
            f.flush()
            if n % 50 == 0:
                print(f"  {n+1}/{len(jobs)} network families, {(time.time()-t0)/60:.1f} min", flush=True)

    df = pd.read_csv(out)
    beta = NA.compute_beta_table(df, SIGMAS)
    beta.to_csv(os.path.join(C.RESULTS, "step2_beta_per_network.csv"), index=False)
    summ = NA.beta_summary(beta)
    summ.to_csv(os.path.join(C.RESULTS, "step2_beta_summary.csv"), index=False)
    amp_path = os.path.join(C.RESULTS, "step2_state_amplitude.csv")
    if not os.path.exists(amp_path):   # noise-free state amplitude: is sigma the same SNR at every r?
        rows = []
        for kind in ["HM", "random"]:
            for i in range(20):
                inst = i if kind == "HM" else i * a.nulls
                base = C.make_base_network(U, kind, inst); seed = C.net_seed(kind, inst, 0)
                rng = np.random.default_rng(seed); w_in = rng.uniform(-C.W_IN_SCALE, C.W_IN_SCALE, C.N)
                u = rng.standard_normal(C.T_STEPS)
                for r in R_LEVELS:
                    A = C.with_reciprocity(U, base, r, kind, inst, C.ridx(r))
                    X = C._harvest(C.scale_to_radius(A, rho), w_in, u, 0.0, 1)[200:]
                    rows.append(dict(network_type=kind, instance=inst, reciprocity=r, state_rms=float(np.sqrt((X ** 2).mean()))))
        pd.DataFrame(rows).to_csv(amp_path, index=False)
    sens = NA.beta_summary(NA.compute_beta_table(df, SIGMAS, max_delay=6))   # common delay range 1..6
    sens.to_csv(os.path.join(C.RESULTS, "step2_beta_summary_delays_le6.csv"), index=False)
    corr = NA.correlations(df, beta, SIGMAS)
    corr.to_csv(os.path.join(C.RESULTS, "step2_correlations.csv"), index=False)
    mc = df.groupby(["network_type", "reciprocity", "sigma"]).MC.agg(["mean", "std"]).reset_index()
    mc.to_csv(os.path.join(C.RESULTS, "step2_mc_summary.csv"), index=False)
    print("\n=== beta (log10 sigma*_n ~ beta * log10 n), mean [95% bootstrap CI over networks] ===")
    for _, r in summ.iterrows():
        print(f"{r.network_type:7s} r={r.reciprocity:4.2f}  beta={r.beta_mean:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
              f"  defined for {r.n_beta_defined}/{r.n_networks} networks, mean delays used {r.mean_delays_used:.1f}")


if __name__ == "__main__":
    main()
