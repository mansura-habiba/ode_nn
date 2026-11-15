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

## Extended execution

Workflow: https://github.com/mansura-habiba/ode_nn/actions/runs/43742185115

Planned input settings: **three paired seeds**, **six teacher families**, **12 training epochs**, **20 adaptation steps**; 90 architecture, 270 repair, 90 no-defect observations. The workflow must finish successfully and output its audited summary before numerical results are added here. Even then, this is exploratory and cannot replace the 10+-seed/fully tuned confirmatory run.

## Next scientific decision

Reject global-superiority language until a matched-budget study with converged models across unmatched dynamics and transparent shifted/source Pareto recovery establishes an advantage. Preserve all negative outcomes. The V2 notebook and its original output cells remain unchanged.
