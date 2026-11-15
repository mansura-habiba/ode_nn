# CCNN: Mechanism-Isolating Experimental Protocol (RQ1-M / RQ6-M)

**Evidence status:** literature-informed PROPOSAL; not implemented or validated. **Do not rerun the current V3 repair suite as confirmatory evidence.**

**Research Identity Record**

```yaml
research_goal: "Improve attention with coupled CCNN lag memory; test precise, transferable, non-regressive model-weight repair"
primary_question: "RQ1-M: incremental attention value; RQ6-M: mechanism-specific weight repair"
hypothesis: "CCNN's conditional lag-memory structure improves a task-level accuracy/cost frontier on some independently defined dynamics, and/or affords fault-specific correction beyond sham retraining"
mechanism_under_test: "Causal lag-kernel conditioning and its identifiable weight-edit subspace"
novelty_vs_prior_art: "Not merely attention+convolution, not a general ODE integrator, not scalar weight tuning"
minimum_baselines: ["attention-only", "attention+static FIR", "attention+dynamic FIR MLP-conditioned", "attention+SSM", "attention+generic Neural ODE", "source-only continued Adam", "sparse Adam", "Gauss-Newton", "kernel tangent basis", "low-rank FIR", "full FIR", "full fine-tuning"]
negative_control: "Unmatched generators for RQ1; no-fault placebo, non-delay fault, random perturbation and random repair coordinates for RQ6"
primary_metric: "RQ1 paired test MSE/accuracy-vs-cost; RQ6 fault-specific excess recovery under held-out preservation"
compute_budget: "Same inputs, search protocol, data accesses, update budgets, measured FLOPs and time; do not equate equal epochs to equal compute"
train_validation_test_boundary: "Seed-paired train/validation/calibration/locked-test splits; no fault labels, clean checkpoint, oracle shifts or test responses to repair algorithms"
acceptance_and_stop_rules: "Pass protocol validity gates below before interpreting or comparing algorithms"
evidence_status: "proposed"
```

## 1. Literature and what to transfer (not identical tasks)

1. **Hymba (ICLR 2025)**: Hybrid attention/SSM heads; controlled architecture comparisons under identical training conditions and component ablations. Transfer: separate the incremental memory effect from adding capacity. https://proceedings.iclr.cc/paper_files/paper/2025/hash/f32def07618040e540e0a6182e290562-Abstract-Conference.html
2. **Samba (ICLR 2025)**: Mamba+sliding-window attention; train on shorter contexts and evaluate much longer passkey/phonebook retrieval, plus throughput. Transfer: tests should force temporal-memory capability and extrapolate beyond training rather than reward a matched teacher. https://proceedings.iclr.cc/paper_files/paper/2025/hash/84a7fc24ed52e8eff514c33e8ac76ea3-Abstract-Conference.html
3. **TimeXer (NeurIPS 2024)**: Exogenous variables integrated explicitly; evaluated on twelve real-world forecasting benchmarks. Transfer: if the environment changes conditional dynamics, make informative physical covariates observable to all competitors rather than withholding them or using a hidden domain gate. https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html
4. **AlphaEdit (ICLR 2025)**: Preservation via null-space projection; multiple LMs and repeated-edit evaluation. Transfer: compare preservation-constrained numerical edits on the same checkpoint. https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html
5. **CrispEdit (ICML 2026)**: Low-curvature constrained weight updates, including formulation not requiring model convergence; small-scale exact-curvature checks plus LLM evaluations. Transfer: preserve functional behavior, use a curvature/Jacobian baseline, and avoid saying every unconverged base invalidates scientific inference. https://proceedings.mlr.press/v306/ikram26a.html
6. **Are We Evaluating the Edit Locality of LLM Model Editing Properly? (2026)**: Demonstrates low sensitivity of common locality metrics to preservation strength; proposes testing metric sensitivity. Transfer: evaluate preservation across a prespecified range of tolerances and make edit quality sensitive to meaningful change. https://arxiv.org/abs/2601.17343
7. **The Mirage of Model Editing (ACL 2025)** and **SEED (EMNLP Findings 2026)**: Editing claims can collapse under realistic testing, untemplated contexts, related facts and multi-edit interactions. Transfer: use untouched independent probes and evaluate repeat repairs, not a single conveniently chosen edit. https://aclanthology.org/2025.acl-long.745/ ; https://aclanthology.org/2026.findings-emnlp.43/
8. **Sparks of Weight-Space Programming (MIT CSAIL, 2 Nov 2026)** vs **Ahmed critique (5 Nov 2026)**: Early model-weight reasoning demonstrations and counterexamples involving memorized class/label priors. Transfer only to RQ7 *after* numerical controls: hide defect identity, randomize equivalent parameterizations, corrupt diagnostic summaries, and equalize probes/forward passes. https://rikiyat.github.io/blog/weight-space-programming/ ; https://arxiv.org/abs/2611.08070

9. **Selective Kalman-filter updating under drift (Scientific Reports, 2025)**: Evaluates parameter-subset updating on a simple drifting system, a stirred tank reactor, insulin sensitivity and an industrial fluid catalytic cracker with multiple simulated faults, measuring error and update cost against retraining/fine-tuning. This is a particularly close comparator for our RQ6-M correction claim: a strong numerical selective-update method must be included. https://www.nature.com/articles/s41598-025-24558-8
10. **Domain Adaptation with Hybrid Modeling for Learning Dynamical Systems (ACML 2025)**: Hybrid physics/Neural-ODE systems tested on multicopter *concept shifts*, using target-domain data efficiency and adaptation improvement. This motivates keeping true environment/concept-shift adaptation separate from network-parameter fault recovery. https://proceedings.mlr.press/v304/osaka26a.html
11. **Mechanistic Unlearning (ICML 2025)**: Compares editing components identified by different localization strategies on sports-fact and CounterFact tasks, including robustness and side effects. This warns that high input-output sensitivity is not automatically proof of mechanistically correct localization; include fault-family diagnosis and intermediate-response checks. https://proceedings.mlr.press/v267/guo25k.html

## 2. RQ1-M: isolate incremental memory complementarity

**One invariant forecasting/detection problem; multiple independently specified data mechanisms.**

- Use at least one **external** or independently simulated dynamical system whose target is not computed with the CCNN kernel formula *or* the exact same downstream leaky ODE.
- Include time-dependent lag retrieval, two-peak history, abrupt switching, distributed memory, and a nonconvolutional state mechanism; matched chirp is a **positive calibration** control, excluded from the primary unmatched-family aggregate.
- Use task-relevant outcomes (e.g., held-out delayed-event classification/recall and forecast MSE), with sufficient difficulty and broad support; report both accuracy and training/inference budget.
- Hold attention architecture, input information, timestamps/covariates and data splits fixed across arms. Add (i) no memory, (ii) static FIR, (iii) affine query-dependent FIR, (iv) **MLP-conditioned unconstrained FIR** and (v) SSM or generic ODE. All controls access identical observations and covariates.
- Two complementary comparisons:
  - **Frozen backbone test**: pretrain an attention backbone, clone/freeze it across arms, then train only temporal-memory branches, controlling trainable scalar and measured compute budget. This isolates residual utility of the memory mechanism.
  - **Joint training test**: train each complete architecture to its own independently selected validation checkpoint, with a shared hyperparameter-search policy and comparable compute. This shows whether any benefit survives realistic optimization.
- Report training curves and assess whether a plateau or common budget was reached. **Convergence is a diagnostic, not a magical requirement; compare performance also at matched FLOPs.**
- Prespecify primary unmatched-family summary; show every teacher and seed. Do not pool a matched-generator win into the main claim.

**RQ1 advancement condition**: a CCNN-specific gain must survive strong dynamic-FIR and SSM baselines on independently specified unseen tasks, with uncertainty and cost transparently reported. Loss on unmatched mechanisms is a legitimate falsification result.

## 3. RQ6-M: replace environment label-shift with identifiable model fault

### Experiment B1 — controlled *model-weight* fault (primary causal test)

1. Train a CCNN+attention checkpoint `theta_clean` using a fixed underlying input-to-target mapping; choose the checkpoint only from training/validation. Retain a locked clean model for *evaluation/oracle comparisons only*.
2. **Inject a fault in the *model parameters*, not the teacher or target labels:** `theta_fault = theta_clean + corruption`. Fault families: center, width, chirp/frequency, query-dependent center gain, and combined/wrong-family faults. Sample magnitudes by a *separate calibration set*, not by test-set scores. Repair methods must NOT see the fault family, injected vector or `theta_clean`.
3. Keep data labels unchanged across intact, corrupt and repaired models. Partition a *single observable* covariate `q` into low-/high-sensitivity regimes; for example, a perturbation of the learned coefficient of `q` affects high-|q| inputs more strongly. **No domain flag/gate** enters prediction. Source/affected samples differ in observable `q` values but share the same true target function.
4. Establish fault detectability: on a calibration population, `theta_fault` must materially underperform `theta_clean` (threshold defined before viewing locked test); otherwise mark `fault_not_detectable` rather than counting an edit failure.
5. From the exact same frozen `theta_fault`, run: unchanged baseline; ordinary Adam continuation (same examples/updates); sparse Adam; Gauss–Newton; 1/5-dimensional structured edit; 1/5-dimensional **tangent-aligned unstructured edit**; spectral and low-rank corrections; full 14x4 FIR correction; and full fine-tuning as an upper-cost control. Distinguish *edit coordinate count*, *functional effect*, gradient evaluations, optimization calls and wall time.
6. Repeat **paired placebo trials** from `theta_clean` with identical editable surfaces, examples, solver budgets and optimizer, but zero injected fault. Also use wrong-parameter faults and a matched-norm random direction.
7. Use separate adaptation-train, validation and locked test populations. All candidates use identical data access; validation selects the edit under a source-retention criterion. Test data must NEVER select edits.

### Quantities that distinguish repair from ordinary additional optimization

Let `L_clean`, `L_fault` and `L_repaired` be MSE on the **same locked affected test set**, before corruption, after corruption and after repair. Report:

```
fault_headroom      = L_fault - L_clean
recovered_fraction  = (L_fault - L_repaired) / fault_headroom
fault_improvement   = L_fault - L_repaired
sham_improvement    = L_clean - L_sham_edited
specificity_excess  = fault_improvement - sham_improvement
source_drift        = L_source_repaired - L_source_clean
```

- Treat `recovered_fraction` as **undefined** if `fault_headroom` is too small. Record failure to identify a fault separately from rejection of an edit.
- `specificity_excess` is the matched-placebo **difference in improvements**, not sufficient alone for mechanistic proof; it is one necessary contrast against continued training.
- Track fault-family identification separately from correction quality. An edit of a known poisoned center coordinate should be an **oracle ceiling**, not the main blinded repair result.
- Measure original/source preservation on independently held-out source inputs against **both** clean and faulted checkpoints. A source gate must not make preservation automatic.
- Test sensitivity to source preservation thresholds and flag model identifiability failures, especially if multiple parameter perturbations yield nearly indistinguishable outputs.

### Experiment B2 — observable operating-regime adaptation (secondary)

Only after B1 passes mechanism-specificity checks: introduce an actual deployment shift with a measurable covariate or history signal available to all competing models, and compare shared-weight adaptation. If the same complete input produces two incompatible labels from distinct domains and no identifying context is available, do **not** score that conflict as a method-specific repair failure. Preserve it as an information-theoretic negative control. For scalar targets, the minimum sum of squared errors across the two contradictory labels is `(y0-y1)^2/2`, irrespective of model architecture.

## 4. Predeclared reporting / stop gates

| Gate | Requirement | If it fails |
|---|---|---|
| G0 — test isolation | Locked tests unavailable to checkpoint, hyperparameter, edit and fault-amplitude selection | Abort inference, regenerate splits |
| G1 — mechanism | Fault touches model parameters, known evaluator-only target mapping remains unchanged; perturbations detectable on calibration | Do not label result mechanistic repair |
| G2 — placebo | Measure no-fault continuation and compare paired fault-specific excess improvement | A positive absolute edit delta alone is not evidence |
| G3 — comparators | Strong MLP-conditioned FIR for RQ1 and tangent/low-rank/full edits + Adam/GN for RQ6 | Do not claim superiority |
| G4 — preservation | Evaluate original behavior on untouched source test; no gate or test selection | Mark source regressions and plot full Pareto front |
| G5 — replication | >=10 paired seeds for confirmatory result, paired seed-level CIs and full per-family reporting | Mark exploratory |
| G6 — interpretation | Matched-generator and oracle repairs separately labelled calibration ceilings | Reject broad inductive-bias/mechanistic claims |

**No additional V3 run should be interpreted as confirming RQ6 before these changes are implemented.**

## 5. Implementation sequence (minimal and directly auditable)

1. Refactor `ccnn_generalization_benchmark.py` so teacher-generated **label shift**, model-weight **corruption**, and **no-fault continuation** are separate entry points; never silently alias them.
2. Add deterministic `FaultInjector` with evaluator-only fault metadata; save clean/faulted checkpoint hashes.
3. Build paired split records keyed by source/affected observation id; assert same targets across clean/corrupt/repaired model evaluations.
4. Add tests for exact-zero fault, blinded fault identity, headroom, preservation, and a tiny analytic case with known reversible corruption.
5. Produce a single CSV row per seed/family/fault/corruption method/repair method/budget with all six quantities above. Never promote visual plots ahead of numerical audit.
6. Run invariant/smoke execution for correctness **only**, then separately the preregistered, converged/compute-matched scientific benchmark; preserve both positive and null outcomes.

**Decision boundary**: If source-only continued Adam or tangent/full-kernel repair achieves equal or better fault-specific recovery and original preservation at lower cost, the stronger structured-repair claim is rejected. A scientifically useful result may instead be compact repair geometry *for identified fault classes*, explicitly conditional on fault family and budget.
