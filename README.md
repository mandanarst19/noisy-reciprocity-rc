This repository reproduces part of Hadaeghi et al. (bioRxiv 2025.09.24.678310) using the authors' public code ([metrics repo](https://github.com/m00rcheh/network-reciprocity-RC-memory-capacity-and-kernel-rank), [NRC repo](https://github.com/m00rcheh/NRC_binary_and_weighted_Network_Reciprocity_Control)) and adds a small preliminary extension on state noise. It is not affiliated with the authors.

## What this is

1. **Step 1 (reproduction).** Memory capacity (MC) and kernel rank (KR) of binary hierarchical-modular (HM) echo-state networks (N = 64, k = 6) as a function of link reciprocity, versus matched random k-out nulls, using the authors' functions for network generation, reciprocity control, MC and KR.
2. **Step 2 (PRELIMINARY extension).** The same networks with Gaussian state noise of varying size in training and testing.

**Motivation for Step 2.** The paper adds only σ = 1e-4 state noise during training (Sec. 4.6) and does not study robustness to noise. Low-reciprocity networks are more non-normal (Sec. 2.3), and non-normal transient amplification is associated with better memory. An open question is whether the same amplification also amplifies noise. This repository looks at that in one small setting and makes no general claim.

## Headline observations (details and caveats in REPORT.md)

| | this repo | paper |
|---|---|---|
| MC slope vs r (HM, level means) | -0.98 (R² 0.987) | -1.26 (R² 0.995) |
| KR slope vs r (HM, level means) | -0.50 (R² 0.951) | -0.45 (R² 0.882) |

The qualitative trend (both decline roughly linearly with reciprocity) is reproduced; the MC slope is about 22% shallower than reported and I have not found the reason. In Step 2, MC of low-reciprocity networks was higher at every noise level tested, with the absolute gap shrinking as noise grows; the noise-threshold exponent β did not give a consistent dependence on reciprocity (it depends on the delay window used).

## How to run

```bash
pip install -r requirements.txt
python step1_reproduce.py     # ~15 min on one CPU core; resumable
python step2_noise.py         # ~3 min; needs results/step1_rho_choice.json from step 1
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # figures + tables
```

Needs internet and `git` on first run: `authors_code.py` clones the authors' repositories into `external/` at pinned commits and patches three problems in memory (see REPORT.md, Step 0). Reduced runs: `--instances 20 --nulls 2`. Optional: `python diagnose_step1.py` (post-hoc diagnostic of the MC-slope gap).

## Layout

`step1_reproduce.py`, `step2_noise.py`, `analysis.ipynb`; helpers `authors_code.py`, `common.py`, `noise_analysis.py`; `results/` (CSVs; `step2_noise.csv` is ~24 MB), `figures/`, `REPORT.md`, `LICENSE` (Apache-2.0), `NOTICE`.

## Honest scope

Single topology family (HM), single size (N = 64, k = 6), binary weights, linear memory task only, one input seed per network, CPU-only. Step 2 is exploratory. Nothing here is a claim about biological brains.

## Attribution and references

- Hadaeghi F., Fakhar K., Khajehnejad M., Hilgetag C.C. *A Computational Perspective on the No-Strong-Loops Principle in Brain Networks.* bioRxiv 10.1101/2025.09.24.678310 (v2, 2026).
- Hadaeghi F., Fakhar K., Hilgetag C.C. network reciprocity control (NRC), *Chaos* 36, 023116 (2026) (citation details as given in the task brief; please verify before citing).
- echoes 0.0.8 (F. Damicelli) is used through the authors' code.

The metrics repository had no LICENSE file when checked, so its code is not redistributed here. See `NOTICE`.
