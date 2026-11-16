# CCNN V3 execution ledger — 15 November 2026

**Research charter:** RQ1-M attention augmentation and RQ6-M ungated weight-space repair. Numerical integration superiority is **not** an endpoint.

## Verified fast execution

- Workflow: https://github.com/mansura-habiba/ode_nn/actions/runs/43741881779
- Commit: `cadbd06595b6b1421b2fbc731c857a29ec1176a2`
- Status: **completed — success**; five unit/invariant tests passed.
- Settings: 1 seed (11), 3 teachers, **2 epochs**, **3 adaptation steps** (smoke only).
- Outputs: 15 architectural evaluations, 45 repair evaluations, 15 no-fault evaluations; 45/45 test source-preservation checks passed the predeclared tolerance.
- All output figures and CSVs archived by the GitHub Actions artifact `ccnn-v3-fast-evidence`.

### Held-out test MSE, smoke settings

| Architecture | Matched chirp | Bimodal unmatched | Boxcar unmatched |
|---|---:|---:|---:|
| Attention only | 0.015656 | 0.070863 | 0.062954 |
| Static FIR | 0.037790 | **0.024144** | **0.023841** |
| Dynamic FIR | 0.035977 | 0.027211 | 0.025182 |
| Exponential SSM | 0.020987 | 0.062893 | 0.056472 |
| CCNN lag + attention | **0.011155** | 0.106350 | 0.097224 |

The initial matched-versus-unmatched ranking is falsification-relevant but **does not establish a generalization deficit**: two epochs are insufficient for convergence assessment, the architecture capacities and compute are not fully matched, and there is only one model seed. Validation-gated source preservation does not by itself establish meaningful shifted recovery.

## Extended execution — completed, scientifically INCONCLUSIVE / repair invalidated

- Workflow: https://github.com/mansura-habiba/ode_nn/actions/runs/43742185115
- Status: **completed successfully as code execution** on the 15 November 2026 run.
- Configuration: **three seeds (11–13), six teacher families, 12 epochs, 20 repair steps**.
- Measured evaluation count: 90 architecture, 270 repair, 90 no-defect observations.

### RQ1-M measured held-out test MSE (3-seed mean; 12 epochs)

| Teacher | Attention | Static FIR | Dynamic FIR | Exponential SSM | CCNN + attention |
|---|---:|---:|---:|---:|---:|
| Matched chirp | 0.013898 | 0.008060 | 0.006847 | **0.013822** | **0.000629** |
| Unmatched bimodal | 0.053042 | 0.008357 | **0.007132** | 0.035736 | 0.052631 |
| Unmatched boxcar | 0.045920 | 0.010524 | **0.009460** | 0.032671 | 0.045244 |
| Unmatched damped oscillator | 0.017636 | 0.007324 | **0.006903** | 0.011686 | 0.020656 |
| Unmatched switching | 0.068865 | 0.041237 | **0.012687** | 0.038081 | 0.074707 |
| Unmatched feedback ODE | 0.069277 | 0.042169 | **0.014478** | 0.039448 | 0.074905 |

Only the matched chirp favors CCNN in this configuration. **Five unmatched families favor non-CCNN controls**, but convergence, architecture capacity, and compute match were not demonstrated. This is strong motivation to challenge the hypothesis, **not** a verified generalization conclusion. The original experiment's shared downstream ODE for most teachers is a further confound.

### RQ6-M gate-free repair: fail the mechanism-specificity check

- **87 / 90 no-defect repair trials accepted updates despite the shift being exactly zero** (96.7%). This is evidence that updates may be ordinary source-model fitting/continuation rather than repair of a localized fault. The original source model was not shown to have converged or to be calibrated enough for mechanism identification.
- **255 / 270 shifted repairs passed the independent source-preservation threshold**. Consequently **15 / 270** fail independent non-regression, even after validation-based filtering.
- **254 / 270 shifted repair arms accepted edits on validation**. Acceptance is not the same as successful test-time recovery and source retention.
- The 56-parameter structured arm is actually **CCNN plus an unstructured residual**, not a pure five-parameter CCNN. Comparisons must label it hybrid.

**Scientific decision:** DO NOT claim H-B/RQ6-M supported from V3. Before additional expensive repair runs, establish source-model convergence, include same-budget *continued training on unshifted data* as a negative-control baseline, demonstrate near-zero no-defect false-acceptance on test, and compare each edit to a source-only update and a nonrepair baseline. There is no domain gate in V3; this makes the preservation assessment honest, but does not by itself make the experiment valid.

### Stop/advance criteria

1. **Baseline adequacy:** evidence of convergence by validation curves and predeclared source-performance threshold; if failed, train source model longer before injecting any fault.
2. **Negative-control specificity:** the shifted-domain repair success rate should *exceed* no-defect improvement under the same process after multiple-test correction; otherwise repair specificity is **not identifiable**.
3. **Trade-off:** require independently verified shifted-test improvement with predeclared source-test preservation, not simply passing one validation constraint.
4. **Fairness:** tune model capacity, optimizer and training budget fairly; >=10 paired independent seeds for confirmatory RQ1/RQ6 claims.

No numerical integration-superiority hypothesis is being assessed.

## Next scientific decision

Reject global-superiority language until a matched-budget study with converged models across unmatched dynamics and transparent shifted/source Pareto recovery establishes an advantage. Preserve all negative outcomes. The V2 notebook and its original output cells remain unchanged.
