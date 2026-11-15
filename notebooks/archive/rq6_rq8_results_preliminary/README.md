# RQ6/RQ8 — Localized, non-regressive repair of a trained neural operator

**Overarching invariant goal:** discover whether *coupled continuous dynamics* contribute something existing architectures cannot achieve as effectively. **Do not interpret this toy as a demonstration of that distinctive advantage.**

- [Executable Google Colab notebook](../RQ6_RQ8_NonRegressive_Operator_Weight_Repair_Colab.ipynb)
- [Standalone reproducible Python source](../rq6_rq8_operator_weight_repair.py)
- [Full exploratory results (3 seeds, all resolutions, validation and controls)](RQ6_RQ8_3seed_metrics.json)
- [Research charter — RQ6, RQ7 and RQ8](../RESEARCH_QUESTIONS_CCNN_ATTENTION_WEIGHT_REPAIR.md)

> **Evidence status:** exploratory synthetic CPU run, 15 November 2026. Three seeds (17, 29, 43), not a confirmatory benchmark, no LLM/Hugging Face and no Transformer attention model tested. No peer-reviewed or generalized performance claim.

## Abstract

We test whether a previously trained, resolution-neutral spectral neural operator with coupled global dynamics admits a *single-parameter repair* when an observable operating condition shifts beyond the training distribution. A parameter-sparse update is proposed using a local Jacobian or matched Adam optimization; it is accepted only if shifted-domain validation improves while original-domain validation remains within a preregistered tolerance. The same accepted edit is tested at 16, 32, 64 and 128 spatial positions without retraining. Crucial controls include random single-weight proposals, full-model fine-tuning, no-defect trials, an unrepresentable advection shift, and a parameter-matched **direct** state-transition model with no continuous-time global ODE. The repair is unusually favorable because the correct nonlinear conditioning feature is included in the learned operator even though inactive during its original training. The results establish toy feasibility for targeted repair, but **do not** isolate a unique contribution from coupled continuous dynamics.

## Conditions and selection boundaries

- **Original-domain training:** 400 trajectories per seed, condition `c∈[-0.20,0.20]`, grid `N=16`.
- **Repair training:** 64 shifted trajectories, condition `c∈[0.62,1.05]`, grid `N=16`.
- **Validation:** independent original (120) and shifted (96) trajectories, used only to accept/reject and choose edits.
- **Held-out test:** 128 fresh continuous functions sampled at each of `N=16,32,64,128`. The Fourier coefficients and conditions are shared *across resolutions*, not between train, validation and test.
- **Prevention of original-domain regression:** old validation MSE cannot exceed `1.1 × frozen_old_val_MSE + 1e-6`.
- **Improvement required:** shifted validation MSE reduction at least `max(1e-5, 1% × frozen_shift_val_MSE)`.
- **Model capacity:** a six-parameter conditional spectral head and four global state parameters. The direct-map control has the same number of trainable parameters.
- **Training budget:** 120 steps for base model, 70 steps for repair alternatives (not equal total FLOPs or wall time). Random one-coordinate search uses 48 attempts.

## Test results, N=128 (three-seed mean)

| Model and repair | Shifted test MSE ↓ | Original test MSE ↓ | Relative shift gain vs coupled frozen |
|---|---:|---:|---:|
| Frozen coupled operator | 0.011047 | 9.05e-9 | — |
| Coupled + Jacobian, one weight | 2.92e-5 | 9.05e-9 | 99.74% |
| **Coupled + sparse Adam, one weight** | **5.69e-7** | 9.05e-9 | **99.99%** |
| Coupled + bounded random search | 7.14e-5 | 9.05e-9 | 99.35% |
| Coupled + full fine-tuning | 0.011047 | 9.05e-9 | 0% (candidate rejected) |
| Direct-state-map + sparse Adam | 8.75e-7 | 1.08e-8 | 99.99% |

**Interpretation:** a single weight edit can transfer unchanged across the four grids *when the correct conditional feature already exists*. But random search also works very well in this six-parameter toy, and a **direct learned transition** is repaired nearly as effectively. This does not establish any unique benefit of the continuous global ODE, nor superiority of Jacobian repair over Adam. Full fine-tuning is under-tuned for the stringent original-domain gate; its rejection is not evidence that full fine-tuning is universally inferior.

## Falsification controls and hypothesis ledger

| Hypothesis | Three-seed toy finding | Status |
|---|---|---|
| H6a — A non-regressive one-weight repair exists | Sparse Adam reduces shifted MSE ~99.99%; old-domain error preserved | **Supported in toy** |
| H8a — Same edit transfers at 16/32/64/128 | Transfer measured with no refitting | **Supported in toy; favorable band-limited setup** |
| H6b — No-defect requests are rejected | No accepted candidate across tested control trials | **Supported in toy** |
| H6c — Missing operator structure cannot be fully repaired by scalar diffusion edit | Advection negative-control MSE remains ~0.136 after repair vs ~0.202 frozen | **Residual persists** |
| H1/H6 — Coupled dynamics confer a unique repair advantage | Direct-state-map + Adam reaches comparable ~8.75e-7 | **Not supported by this toy** |
| H7 — Hugging Face WSP outperforms numerical optimization | No LLM inference performed | **Not tested** |

"Supported in toy" does **not** imply a mathematical proof; only three random seeds were evaluated. This test was built to answer an existence question, not the stronger question of discoverable repair in unknown architectures.

## What we learned and what should happen next

The synthetic system contains a conditional hinge-square feature that is identically zero on the original training regime. An edit to its coefficient is *guaranteed not to perturb the original regime*, provided the rest of the model is frozen. This mechanism partly explains the positive result and restricts the conclusion. Likewise, using a Fourier-consistent operator on band-limited reference functions makes the 16→128 transfer unusually easy.

**Next decisive question:** does a *continuous* coupled global generator outperform parameter-matched discrete mappings, generic Neural ODEs and neural operators **when elapsed time varies, observed states arrive irregularly, and global coupling itself changes dynamically**? The fixed-horizon direct mapping can implement the same linear transition as an ODE matrix exponential, so this experiment does not distinguish those methods.

**Next experiment T3a (proposed, not implemented):** train the models at several elapsed times, hold out long times and irregular observation schedules; add a condition-dependent, evolving coupling operator and shifts that demand structural edits rather than one known feature. Compare error, original-domain non-regression, numerical stability, solver evaluations, runtime, and parameter-matched baselines. Introduce three Hugging Face weight-programming models **only after** numerical controls are competitive and the failure cannot be solved trivially by the provided feature basis.

## Research integrity

No attention model, model of arbitrary hidden weights, full FNO benchmark, real music data, HF-WSP advantage, or general architectural superiority is demonstrated. Generated outputs and acceptance states are saved by the Colab notebook. Always report non-accepted edits as rejected, not improvements.
