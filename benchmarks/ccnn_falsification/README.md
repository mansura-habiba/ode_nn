# CCNN generalization and gate-free repair benchmark (V3)

**Status:** falsification-oriented implementation; **not executed or independently evaluated in this change**. No general-superiority claim. The V2 publication notebook remains untouched.

**Governing charter:** [original research questions — archived](../../notebooks/archive/RESEARCH_QUESTIONS_CCNN_ATTENTION_WEIGHT_REPAIR.md). Its invariant goal is better predictive performance, generalization and efficiency of **attention augmented by coupled CCNN**, and precise non-regressive **weight-space repair**. Solver superiority and demonstrating continuous-time integration are **not** research goals.

### Research Identity Record

```yaml
research_goal: "Improve attention through coupled CCNN dynamics; assess non-regressive weight repair"
primary_question: "RQ1-M (attention augmentation), RQ6-M (mechanism-aligned repair); RQ7 deferred"
hypothesis: "Compact lag structure helps on a subset of unmatched processes and/or enables more efficient repair than expressive matched-budget controls"
mechanism_under_test: "Lag-kernel attention augmentation and gate-free shared-weight repair"
novelty_vs_prior_art: "Conditional predictive complementarity and repair geometry, not first combination of convolution and attention"
minimum_baselines: ["attention-only", "attention+static FIR", "attention+dynamic FIR", "attention+exponential SSM", "structured edit", "tangent edit", "spectral edit", "low-rank edit", "full FIR edit"]
negative_control: "Out-of-family teachers and no-fault edits"
primary_metric: "Paired held-out MSE (augmentation); shifted recovery subject to source-domain test preservation (repair)"
compute_budget: "Record model parameters, training time, adaptation time; match edit scalars and steps; training compute matching still pending"
train_validation_test_boundary: "Independent splits; domain labels paired on the same inputs within each split; validation selects edits; test never selects"
acceptance_and_stop_rules: "Val source <= baseline*1.10+0.0001 and val shift improves; source preservation must independently pass on test"
evidence_status: "implemented"
```


## Invariant questions

1. **Generalization:** Does the coupled continuous-convolution parameterization help on dynamics it does not encode?
2. **Repair geometry:** Given the *same frozen CCNN source model*, does an interpretable structured edit achieve a better shifted-domain recovery/preservation trade-off than unstructured edits with the same number of adjustable scalars?
3. **Preservation:** Can one shared repair improve the shifted operating domain without materially regressing an independent original-domain test, *without* a domain gate?

The predecessor `notebooks/CCNN_Lag_Kernel_Attention_Repair_Publication_V2_Colab.ipynb` found matched-generator mean test MSE around `6.346e-6` (CCNN) versus `8.227e-4` (best non-CCNN model), with structured Adam repair around `8.991e-6` versus a one-coordinate dynamic-filter control around `8.7e-3`. Those are exploratory two-seed observations, **not** comparable evidence of generic superiority.

## Factorial falsification suite

| Teacher | Kernel relationship to CCNN | Mechanism under shift | Why it matters |
| --- | --- | --- | --- |
| Gaussian chirp | Matched | Shift kernel center | Positive feasibility control only |
| Two separated lobes | Unmatched | Displace centers in opposite directions | Non-single-peak structure |
| Two boxcar-like pulses | Unmatched | Shift one pulse | Sharp support boundaries |
| Damped oscillator | Unmatched | Change angular frequency | Oscillatory, non-Gaussian memory |
| Switching modes | Unmatched | Change query-dependent switching threshold | Abrupt regime changes |
| Feedback ODE | Unmatched kernel **and** dynamics | Shift mixing threshold | Tests the fixed downstream ODE hypothesis |

**Architecture controls:** attention-only; attention + static FIR; attention + unstructured query-conditioned FIR; attention + exponential SSM; attention + CCNN lag kernel. All use independent train/validation/test sequences; report test MSE, train time, model parameters. Architectural sizes are recorded, **not yet exactly matched**.

**Repair controls (same frozen base, no domain gate):** One- and five-parameter *tangent* edits use the numerical local sensitivity of the **same learned CCNN kernel**. These stronger controls probe whether an apparent structured advantage is merely the result of allowing one scalar to modify many effective weights; they share prior knowledge of the CCNN tangent space and must be interpreted accordingly. the CCNN source model is fixed and **every adapter is active for both populations**. All adapters change the kernel but not the attention backbone.

| Learnable budget | Structured / hybrid | Unstructured Fourier/DCT basis | Low rank | Full kernel |
| ---: | --- | --- | --- | --- |
| 1 | lag center only (1) | 1 spectral coefficient; **1 CCNN tangent coordinate** | — | — |
| 5 | five CCNN kernel parameters | 5 spectral coefficients; **5 CCNN tangent coordinates** | — | — |
| 18 | five structured + 13 residual coefficients | 18 spectral coefficients | rank-1 (14×1 + 1×4 = 18) | — |
| 36 | five structured + 31 residual coefficients | 36 spectral coefficients | rank-2 (14×2 + 2×4 = 36) | — |
| 56 | five structured + 51 residual coefficients | 56 spectral coefficients | — | all 14×4 coefficients |

At >5 parameters, **structured is a hybrid**, so interpret separately from pure CCNN edits. Each spectral/FIR correction is conditioned on four predetermined features: `[1,q,q²,sin(πq)]`. Parameter counts are validated in code. Baselines are optimized with the same optimizer, step count, data sizes and validation selection rules; compare measured wall time as well.

**Validation protocol:** source and shifted labels use the **same input sequences within each split**, exposing the inherent compatibility limit without a hidden domain identity. Train, validation, and test splits are otherwise independent. Select an edit only when its shifted-domain validation error decreases *and* the original-domain validation error is <= original baseline ×1.10 + 0.0001. Identity/no-edit is always a candidate. The shifted and original test sets are held out and must **never** select edits. Report both raw errors, relative recovery, source-domain drift, and a binary test-time preservation flag; no test-time gate.

**Negative control:** repeat adaptation with shift=0. A method proposing nontrivial improvement from a non-existent fault is evidence of overfitting or an underlying poorly fitted source model; assess against independent test measurements.

## Running

The fast invariant test file checks parameter counts, identity-at-zero edits, first-order structured/tangent agreement, and absence of a domain-selective forward argument. **Its assertions are committed but have not been executed or independently verified in this session.** Run it before trusting V3 benchmark outcomes.

```bash
python -m unittest discover -s benchmarks/ccnn_falsification -p 'test_*.py'
```

Run the stand-alone `ccnn_generalization_benchmark.py` from this directory or Colab. Dependencies: Python 3.10+, `numpy`, `torch`, `matplotlib`.

```bash
python ccnn_generalization_benchmark.py --profile smoke --out results/smoke
python ccnn_generalization_benchmark.py --profile full --out results/full
# Change shift magnitude without changing the selection protocol:
python ccnn_generalization_benchmark.py --profile smoke --shift 0.06 --out results/shift006
```

Default smoke: 1 seed, 3 kernel families, 10 training epochs, 15 adaptation steps. Default full: 10 seeds, all 6 families, 35 epochs, 85 adaptation steps. Either is computationally feasible but the full profile is much more expensive. Smoke plots are **illustrative only**.

Output contract: `architecture.csv`, `repair.csv`, `no_defect.csv`, `manifest.json`, and source-generated `Figure_00/01/02` PNG, SVG and vector PDF figures. Inspect the manifest before citing numerical results. Publication decisions require 10+ paired seeds, 95% paired bootstrap confidence intervals, multiple shift magnitudes, a compute-matched architecture sensitivity analysis, and separate replication.

## Failure modes / mandatory interpretation

- A matched chirp win is only an inductive-bias check.
- **The main goal is attention augmentation and weight-space repair (RQ1-M/RQ6-M).** Solver-error minimization is not a research endpoint; existing discretization must be documented as a validity constraint.
- The original teacher uses the *same lag integrator* for five of six families. Thus even an unmatched kernel is not a fully independent physical process; feedback ODE is an initial dynamical stressor, not a substitute for physical data.
- The two operating populations can demand incompatible outputs for the same inputs. No ungated shared model can be guaranteed to optimize both; the recovery-versus-preservation Pareto front is itself informative.
- Low-rank **parameter count** is controlled, but optimization geometry, initialization and Jacobian conditioning differ; run multiple learning rates and optimization methods before making superiority statements.
- `source_pass_test` is evaluated independently, not guaranteed by a gate or enforced on the test set.
- A 56-parameter full FIR perturbation on a frozen CCNN kernel is not equivalent to full backbone/network fine-tuning. Include true end-to-end fine-tuning as a further benchmark.
- Further work: irregular-time observations and adaptive quadrature, unseen system regimes and empirical datasets, end-to-end Neural ODE/SSM controls, and identification under reparameterization.

## Related files

- [Architecture](ARCHITECTURE.md) — paper-style three-panel PDE/ODE schematic and precise computational formulation.
- [Discussion](DISCUSSION.md) — conservative manuscript wording and decision rules.
- [Original V2 notebook](../../notebooks/CCNN_Lag_Kernel_Attention_Repair_Publication_V2_Colab.ipynb) — preserved without edits.
