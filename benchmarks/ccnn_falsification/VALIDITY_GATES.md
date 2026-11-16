# Research-validity gates: RQ1-M / RQ6-M

**Status: draft protocol, NOT yet preregistered or confirmed.** The accompanying `protocol.yaml` is a YAML 1.2-compatible JSON document intentionally parsable with the Python standard library (`json`). Do not change the status to `FROZEN_PRE_RUN` until the protocol is reviewed and all primary decisions are locked before confirmatory test evaluation.

## G0 — Independence and leakage prevention
- [ ] Train/validation/calibration/locked-test sample IDs are disjoint for each seed and family.
- [ ] No test data select a checkpoint, learning rate, fault magnitude, fault family, parameter subset or acceptance decision.
- [ ] Edited models cannot inspect fault oracle information or clean checkpoint weights.

## G1 — RQ1-M incremental attention
- [ ] Matched chirp is calibration-only, excluded from the unmatched primary aggregate.
- [ ] Strong dynamic MLP FIR, SSM and generic ODE controls get identical inputs and timestamp features.
- [ ] Frozen identical attention backbone and separately jointly trained arms are both measured.
- [ ] Independent teachers do not call the CCNN lag kernel or its leaky ODE in target generation.
- [ ] Paired raw results, training curves, error–compute Pareto and negative outcomes retained.

## G2 — RQ6-M intervention identification
- [ ] Primary repair injects corruption into saved **model weights**, leaving the original mapping and labels unchanged.
- [ ] A locked clean reference is used ONLY by evaluator to measure fault headroom and kernel recovery.
- [ ] Corruption is pre-calibrated on a calibration population, never on locked test; insufficient headroom is flagged explicitly.
- [ ] No-fault placebo branch begins from the same clean checkpoint and receives the same data and compute.

## G3 — Meaningful repair and preservation
- [ ] Report `fault_improvement`, `placebo_improvement`, `specificity_excess`, `headroom` and `recovered_fraction` (undefined if headroom inadequate).
- [ ] Include structured, tangent, numerical optimizers, low-rank, full FIR and full-fine-tune controls; measure functional edit size.
- [ ] No operating-domain gate exists in the shared model forward path.
- [ ] Validation can select an edit, but independently locked source and affected tests determine actual success; plot full recovery/retention trade-off.
- [ ] Same-input contradictory-label scenario is only an impossibility negative control, not the primary repair endpoint.

## G4 — Statistical and reporting standard
- [ ] At least 10 independent paired seeds and a prespecified paired 95% CI for confirmatory interpretation.
- [ ] All aborted/rejected/no-load/inconclusive and no-fault outcomes retained in auditable machine-readable files.
- [ ] Report compute and actual updated parameter count alongside error. Do not equate equal epochs with equal compute.
- [ ] Every claim maps to a frozen run manifest and traceable CSV; smoke runs are **engineering checks only**.
- [ ] Source-domain preservation and fault specificity must both hold. Otherwise report **failed/inconclusive**, not supported.

## Claim boundaries
No claim of general CCNN superiority; not a study of superior numerical ODE integration. Distinguish architecture *inductive bias* from source-task match and repair *semantic geometry* from ordinary continued optimization.

## Freeze procedure
1. Agree to the actual tasks, noise levels, time budgets, optimizer search space, fault magnitudes, primary effect size and failure thresholds.
2. Check that each control has a runnable implementation; record its config and checkpoint-generation rules.
3. Review `protocol.yaml` and this document, mark the registration `FROZEN_PRE_RUN` in an auditable commit, and preserve this SHA.
4. Only then conduct confirmatory runs and lock the analysis to that commit.
