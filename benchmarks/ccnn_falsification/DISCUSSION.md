# Manuscript Discussion — conservative scope and future confirmatory study

> The preliminary experiments demonstrate the feasibility of structure-aware convolutional repair, but the observed advantage is conditional on the data-generating mechanism and the choice of repair parameterization. The present evidence is insufficient to establish superiority over expressive unstructured filters or conventional optimization methods.

The original two-seed experiment deliberately employed a Gaussian-chirp teacher whose parameterization overlaps the CCNN lag branch. Its extremely low matched-generator test error is therefore consistent with a strong inductive bias, not evidence of broad predictive superiority. Likewise, the one-coordinate dynamic-filter repair arm cannot fairly stand in for low-rank adapters, an arbitrary full-kernel correction, or end-to-end fine-tuning. Validation-selected delay edits applied only under an explicit shifted-domain gate trivially preserve original-domain predictions; this must not be conflated with learning to avoid forgetting.

The follow-up experiment should test dynamics outside the hypothesized kernel class, include both parameter-matched low-rank and fully expressive repair options, and score source preservation and shift recovery on independent data without a domain gate. Since the two domains may require different outputs for identical observations, preservation and recovery can be structurally incompatible. A failure to jointly improve them is therefore scientifically informative: plot the full Pareto frontier, not only the best shifted score.

**Decision boundaries, preregister before a confirmatory run.**
- Generalization claim supported only if CCNN outperforms strongest non-CCNN control on a *majority* of independently specified unmatched families at matched task/compute budgets with paired 95% intervals excluding no difference.
- Repair claim supported only if matched-budget structured edits consistently improve the shifted test error **without** exceeding the prespecified original-domain test degradation tolerance, across independently seeded teachers and faults.
- If unmatched families eliminate the accuracy advantage but compact structured repair remains valuable on identifiable delay faults, reposition the contribution as a **mechanism-aligned intervention interface**.
- If sufficiently expressive full-kernel updates dominate under equivalent compute, reject a claim of repair superiority and characterize structure as a low-data/low-budget prior.
- Do not claim a continuous-time solver/general neural ODE advantage until irregular-grid integration error and NFE are measured. The current implementation is a discretized coupled PDE/ODE approximation.

**Minimum additional reporting:** 10+ independent paired seeds, shift magnitudes including zero, separate train/validation/test populations, total parameters and *updated* parameters, wall time and optimization evaluations, accepted-edits rate, source regression counts, test effect sizes with paired bootstrap CIs, and failure cases. Add at least one non-synthetic real sequence benchmark.

No new benchmark results in the V3 proposal should be described as independently replicated until the suite has been executed and audited.
