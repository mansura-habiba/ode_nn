# RQ4 — Preliminary spatial-resolution transfer results

**Study:** [RQ4 CCNN Spatial-Resolution Transfer Colab](../RQ4_CCNN_Spatial_Resolution_Transfer_Colab.ipynb)  
**Source of record:** [Research charter — RQ4](../RESEARCH_QUESTIONS_CCNN_ATTENTION_WEIGHT_REPAIR.md)  
**Evidence:** [Three-seed raw metrics, training costs, paired differences, and hypothesis ledger](RQ4_preliminary_3seed_metrics.json)

> **Evidence status: local CPU exploratory run, 15 November 2026; not a Colab benchmark, not peer-reviewed, and not statistically confirmed.** The notebook produces a new live ledger whenever rerun. These results are included to record *both* evidence and failure without retrospectively rewriting the hypothesis.

## Experimental conditions

All trainable models were fitted using **16 spatial channels**, 256 training initial fields and 30 epochs per seed, across seeds **17, 29 and 43**. Evaluation used the same underlying continuous functions sampled at **16, 32, 64 and 128 channels**, with forecast horizons 0.35, 0.75 and 1.4; the training horizon was within 0.12–0.85. Two generators were used: local coupled diffusion–reaction dynamics, and a nonlocal symmetric shift operator that is not well described by diffusion. The continuous reference was checked for consistency across grids. Metrics and code versions are stored in the archived JSON.

The tested architecture is a **linear PDE–ODE semigroup subcase**, not a general nonlinear CCNN, and its local generator shares its mathematical operator family. Baselines include an unscaled stencil, spectrally consistent CCNN, pointwise Neural ODE, small FNO-style layer, and coordinate-aware attention. Parameter counts and training times differ substantially; the comparison should not be interpreted as a fully compute-matched benchmark.

## Hypothesis ledger

| Hypothesis | Measured evidence (N=128, t=0.75 unless noted) | Toy-study status |
|---|---|---|
| H4a — mesh scaling matters | Scaled CCNN MSE 0.00255; unscaled stencil 0.0848 | **Supported** |
| H4b — scaled CCNN beats strong resolution-neutral controls | Spectral CCNN MSE 0.0000001242, better than scaled | **Contradicted** |
| H4c — diffusion CCNN generalizes to nonlocal dynamics | Spectral CCNN MSE 0.03064; FNO-lite 0.00278 | **Contradicted** |
| H4d — relative error remains stable across N | Scaled CCNN relative L2 rises 0.03495 (N16) → 0.08058 (N128) | **Contradicted by predeclared tolerance** |
| H4e — better accuracy per compute | Exact semigroup and RK4 not solver-matched | **Inconclusive** |
| H4f — one weight edit transfers to finer grids | No weight-edit intervention performed | **Not tested** |

Here **supported** and **contradicted** mean *within this synthetic study*, not mathematical proof or universal refutation. Seed-level paired differences and descriptive bootstrap intervals are archived, but three seeds are insufficient for a robust statistical conclusion.

## What the evidence changes

1. **A positive result:** correct `1/Δs²` scaling is essential for a discrete diffusion operator to represent comparable physics as grid size changes. A filter learned at N=16 without scaling can diverge from the intended continuous operator at N=128.
2. **A counterexample to a broad superiority claim:** the spectrally consistent operator performs best on the local data, which exactly follow a linear diffusion generator. Therefore, **grid-aware finite-volume CCNN did not establish a unique advantage**.
3. **An inductive-bias limitation:** a local diffusion parameterization was worse than a Fourier baseline for the nonlocal generator. A method that succeeds only when the generator matches its assumed PDE operator is not a universal attention augmentation.
4. **A separate discretization issue:** although the scaled model remains numerically finite, its relative error increases across grids, consistent with coarse-grid parameter estimation compensating for finite-difference dispersion bias.
5. **A computational limitation:** inference timings cannot establish algorithmic efficiency without solver accuracy, equal hardware and matched optimization budgets.

## Next research actions — not yet implemented

**Next T1a:** isolate *coarse-grid dispersion compensation*. Compare a learned spectral multiplier with a conservative finite-volume operator using physically consistent wavenumbers and the same parameter-count/solver budgets; inspect learned Fourier decay rates versus the continuous reference.

**Next T1b:** extend the mathematical operator **only after diagnosing T1a**. A candidate combines local conservative flux with a learned nonlocal interaction kernel. Compare it to a tuned FNO and an attention-based operator on local, nonlocal and heterogeneous-coefficient generators—not just on the favorable local reference.

**Gate to RQ1/T2:** augment attention with CCNN only if a structurally distinct operator shows improvement over the strongest resolution-neutral baselines under matched budgets, and with acceptable runtime. **Gate to RQ8/T3:** test weight-space repair only after a reproducible, localized defect exists; the same edit must then survive 16→32/64/128 transfer and original-domain non-regression.

The correct outcome of RQ4 so far is **refinement of the hypothesis**, not declaring CCNN superior or prematurely scaling an LLM-based repair system.
