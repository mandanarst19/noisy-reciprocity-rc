#!/usr/bin/env python
"""POST-HOC diagnostic for Step 1 (not part of the reproduction itself).

Question: how much of the gap between my MC slope and the paper's (-1.26) can be attributed to the
state-noise protocol and the spectral radius?  Re-computes MC for the same 50 HM instances / 11
reciprocity levels with my own reservoir (validated against the authors' function at sigma = 0)
under alternative noise protocols.  Variants are reported side by side; none is selected as 'the' result.
"""
import os, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import authors_code, common as C

U, _ = authors_code.load()
R = [round(0.1 * i, 1) for i in range(11)]
VARIANTS = {  # name: (sigma_train, sigma_test, readout_includes_u)
    "no noise": (0.0, 0.0, False),
    "Gaussian 1e-4, train only (paper text)": (1e-4, 0.0, False),
    "Gaussian 1e-4, train+test": (1e-4, 1e-4, False),
    "Gaussian 1e-4, train only, readout [x;u] (paper text)": (1e-4, 0.0, True),
    "no noise, readout [x;u]": (0.0, 0.0, True),
}
rows = []
for i in range(50):
    base = C.make_base_network(U, "HM", i)
    seed = C.net_seed("HM", i, 0)
    for ri, r in enumerate(R):
        A = C.with_reciprocity(U, base, r, "HM", i, ri)
        for rho in (0.95, 0.99):
            for name, (s_tr, s_te, ro) in VARIANTS.items():
                c = C.mc_curve(A, rho, seed, s_tr, noise_seed=C.net_seed("HM", i, ri) + 17, sigma_test=s_te, readout_input=ro)
                rows.append(dict(instance=i, reciprocity=r, rho=rho, variant=name, MC=C.mc_from_curve(c)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(C.RESULTS, "step1_diagnostic.csv"), index=False)
out = []
for (rho, name), g in df.groupby(["rho", "variant"]):
    m = g.groupby("reciprocity").MC.mean()
    f = C.linfit(m.index.values, m.values)
    out.append(dict(rho=rho, variant=name, MC_r0=m.iloc[0], MC_r1=m.iloc[-1], slope=f["slope"], R2=f["r2"]))
res = pd.DataFrame(out)
print(res.round(3).to_string(index=False))
res.to_csv(os.path.join(C.RESULTS, "step1_diagnostic_summary.csv"), index=False)
print("paper: slope -1.26, R2 0.995")
