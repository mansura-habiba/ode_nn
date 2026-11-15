# Research charter: coupled Continuous CNNs, attention augmentation, and verified weight-space repair

**Status:** living research charter, version 1.0 — 15 November 2026  
**Repository:** `mansura-habiba/ode_nn`  
**Scope:** research questions and experimental decision rules, **not** a new notebook, algorithmic result, or performance claim.  
**Primary source:** Habiba & Pearlmutter, *Continuous Convolutional Neural Networks: Coupled Neural PDE and ODE* (2021) [R1].

## Abstract

Attention-based neural networks provide effective content-dependent interactions, but the treatment of continuous dynamics, physical/spatial resolution changes, and adaptation to distribution shift remains an open architectural design problem. Coupled Continuous Convolutional Neural Networks (CCNNs), in the sense of the neural PDE–ODE framework of Habiba and Pearlmutter, provide a possible complementary mechanism: a field with shared spatial operators evolves continuously alongside a compact global state. This programme asks whether such dynamics **augment existing attention architectures** with useful information that cannot be obtained just as effectively by discrete convolution, generic Neural ODEs, or additional trainable parameters. A second, conditionally dependent programme tests **Physics-Constrained Weight-Space Repair (PCWSR)**: whether failures of a trained attention–CCNN hybrid can be localized and corrected through small parameter interventions, including edits proposed by open-weight language models. The investigation begins with falsifiable synthetic benchmarks spanning temporal extrapolation, irregular observations, spatial-resolution transfer, and controlled faults. It explicitly includes nonlocal negative controls, resolution-neutral baselines, numerical stability measurements, and WSP mechanism audits. Only after these tests justify the architecture will the programme expand to long-context and music tasks. This document registers questions, hypotheses, baselines, failure criteria, and decision gates; **no improvement is presumed**.

## 1. Invariant research goal

> **Improve the predictive performance, generalization, and efficiency of attention-based neural networks by augmenting them with coupled Continuous CNN/PDE–ODE dynamics, and determine whether targeted weight-space edits can repair the resulting systems without full retraining or regression.**

The desired contribution is **complementary dynamics** that improve an existing attention system. Replacing attention, reproducing ordinary CNNs, demonstrating that ODE integration works, producing attractive diagrams, or showing an LLM can emit JSON are **not research goals** by themselves.

**Terminology.** Here “CCNN” refers to the **coupled neural PDE–ODE formulation** [R1]. It does not mean merely a conventional CNN, a continuous convolution kernel of the CKConv type, or a generic dense Neural ODE. Any experiment using “CCNN” must explicitly identify the field operator, the global ODE, and the coupling between them.

**The two connected workstreams:**

- **A — CCNN × attention:** determine *when and why* continuous coupled spatial–temporal structure adds value to attention.
- **B — PCWSR × CCNN × attention:** determine whether a trained hybrid can be **mechanistically repaired**, especially across time and spatial resolution, using localized changes proposed by numerical methods and optionally LLMs.

Workstream B depends on a well-characterized model and failure mode from A. It must not quietly replace the main objective with “make an LLM predict weights.”

## 2. Current empirical state: negative results matter

Prior exploratory toy tests in this conversation (small synthetic runs, **not independently verified publication results**) indicate:

- Attention augmented with coupled CCNN improved *long-gap temporal forecasting* on a generator containing local diffusion and global coupling, but a **generic Neural ODE was slightly better**.
- The advantage disappeared or reversed for **nonlocal dynamics without diffusion**.
- CCNN imposed an additional **runtime cost**.
- The music spectrogram experiment did **not** establish a meaningful advantage over persistence.

Consequently, **there is no general CCNN superiority claim**. The unresolved question is not “can an ODE improve long-gap predictions?” but “does **coupled, shared, spatial structure** contribute a robust advantage over generic continuous dynamics and competitive sequence models?”

Record these as motivating observations, not verified numbers. Run a fresh, reproducible benchmark before citing numerical values in a manuscript.

## 3. Proposed mathematical investigation (hypothesis, not established method)

Let \(x_t\) be irregularly observed input, \(u(s,t)\) a continuous spatial/feature field, \(z(t)\) a global latent state, and \(h_t\) an attention representation.

A *candidate* coupled operator is

\[
\partial_t u(s,t)
= -\partial_s F_{\theta}(u,z,x_t,s)
+ \partial_s\!\big(D_{\theta}(u,z,x_t,s)\partial_s u\big)
+ R_{\theta}(u,z,x_t,s),
\]

\[
\dot z(t)=g_\phi\!\left(z(t),\operatorname{Pool}(u),x_t,\Delta t\right).
\]

A hybrid attention layer may use either the field as an additional residual feature,

\[
h'_t=h_t+\alpha_\psi(z_t,x_t)\odot P_\omega[u(\cdot,t)]
       +\operatorname{Attention}(h_t),
\]

or a **dynamical gate** on the attention output. Compare these *separately*: they test different mechanisms. Here \(P_\omega\) maps fields to the attention feature dimension. The learned flux \(F_\theta\), diffusion \(D_\theta\ge0\), reaction \(R_\theta\), and global coupling should be separately ablated.

**Critical mathematical issue — resolution.** On a uniform grid of width \(\Delta s\), an ordinary second-derivative discretization contains \(1/\Delta s^2\); applying the same unscaled convolution at 16 and 128 channels **does not** represent the same physical operator. This suggests comparing grid-aware conservative flux / finite-volume schemes against a naïve fixed-kernel CNN. But scaling also changes solver stiffness and work: a spatially accurate scheme can become computationally unattractive as resolution increases. Report solver steps, accuracy, stability and time.

No theorem of attention improvement, resolution invariance, or memory compression is implied by these equations.

## 4. Core research questions and falsifiable hypotheses

### RQ1 — Complementarity: does coupled continuous dynamics add value to attention?

**Question.** For matched trainable capacity and compute, does adding a coupled PDE–ODE component to an attention model improve prediction beyond attention alone, attention + ordinary CNN, and attention + generic Neural ODE?

**Hypothesis H1.** In systems where local interactions and a slowly varying global state jointly affect future observations, the hybrid may improve held-out forecasting, especially out-of-distribution.

**Disconfirming evidence.** Any gain disappears when parameter counts or training budgets are matched, or attention + generic Neural ODE matches or beats it.

**Primary metric.** Held-out error and accuracy–latency/memory Pareto comparison, across random seeds.

### RQ2 — Causal mechanism within the architecture: which component matters?

**Question.** Is a measured gain attributable to the **local spatial operator**, **global ODE state**, **coupling**, **continuity**, or merely additional model capacity?

**Hypothesis H2.** Specific dynamics/coupling ablations will reduce performance on tasks that require those components, while unrelated tasks should not show artificial gains.

**Controls.** PDE-only; ODE-only; no coupling; no diffusion; no transport; no dynamic gate; parameter-matched static CNN; generic Neural ODE; shuffled spatial neighborhoods.

**Disconfirming evidence.** The gain survives all mechanism removals, or an equal-sized non-dynamical adapter does as well.

### RQ3 — Temporal extrapolation and irregular observations

**Question.** Can the attention–CCNN hybrid extrapolate to **unseen forecast gaps**, missing observations and irregular timestamps better than competitive attention / continuous-time baselines?

**Hypothesis H3.** Continuous evolution helps when true elapsed time is important and observations arrive asynchronously.

**Controls.** Attention with explicit elapsed-time encoding; attention + time-aware CNN; attention + generic Neural ODE; Neural CDE/ODE-RNN where relevant. Do **not** handicap attention by withholding \(\Delta t\).

**Evaluation.** In-range and out-of-range forecast error, calibration, long-gap degradation, missingness stress tests, solver cost.

**Disconfirming evidence.** A generic Neural ODE or time-aware attention obtains equal or better performance at lower cost.

### RQ4 — Spatial resolution and operator transfer (first decisive toy study)

**Question.** After training **only on a 16-channel discretization**, can *the same shared CCNN parameters* forecast at **32, 64, and 128 channels** without retraining, while preserving accuracy and numerical stability?

**Hypothesis H4.** A properly discretized local shared operator may transfer better than a model whose computation is tied to a particular grid.

**Design requirements:**

1. Generate all targets from the **same underlying continuous system**; changing \(N\) must change sampling resolution, **not** the true problem.
2. Use fixed physical domain and time units; make initial conditions, boundary conditions, noise and reference solver precision explicit.
3. Train at \(N=16\), test \(N\in\{16,32,64,128\}\); report both **pointwise error** and **spatially integrated error**.
4. Compare a naïve convolution, a grid-scaled/conservative CCNN, a **resolution-neutral pointwise Neural ODE**, a neural operator (e.g., FNO), and attention with relative spatial coordinates. A fixed-size dense ODE is only a **secondary** capacity control because it cannot accept \(N\ne16\) without an adapter.
5. Measure wall-clock time, function evaluations, memory, stability/divergence, and solver sensitivity at each resolution. For explicit diffusion schemes, test the stability cost as \(\Delta s\to0\).
6. Include a **nonlocal negative-control generator** and tasks with changing spatial coefficients. If a method only succeeds on an exactly matching diffusion generator, report that narrow scope.

**Disconfirming evidence.** Resolution transfer is no better than resolution-neutral pointwise ODE, FNO, or attention after matching compute and parameter budget; or it needs prohibitively many solver steps.

### RQ5 — Does CCNN change attention's efficiency–quality frontier?

**Question.** Can a CCNN *help* attention via more informative features, dynamic gating or persistent state **without** incurring greater total computational cost than the performance gained?

**Hypothesis H5.** Chunk-level integration and compact states may improve long-gap accuracy at acceptable overhead; per-token adaptive integration may not.

**Evaluation.** Training FLOPs/proxy, wall-clock, peak memory, inference latency, number of ODE evaluations and task accuracy.

**Disconfirming evidence.** A simpler attention + CNN/Neural ODE control is strictly better on error and cost.

### RQ6 — Can we localize and repair hybrid-model failures with minimal weight edits?

**Question.** When a trained attention–CCNN hybrid fails under a known perturbation (temporal shift, diffusion change, boundary condition, altered coupling), can a **small and identifiable** parameter edit improve the shifted domain while preserving the original one?

**Hypothesis H6.** Certain mechanistically localized defects are easier to correct via restricted edits than indiscriminate full-network retraining.

**Proposed PCWSR selection problem:**

\[
\min_{\delta\theta\in\mathcal S}
  \mathcal L_{\rm shifted}(\theta+\delta\theta)
 +\lambda\,\operatorname{Complexity}(\delta\theta)
\]

subject to

\[
\mathcal L_{\rm original}(\theta+\delta\theta)
-\mathcal L_{\rm original}(\theta)\le\epsilon
\]

and verified numerical stability.

\(\mathcal S\) is a small candidate parameter subset **chosen without test leakage**; \(\epsilon\) is a preregistered original-domain tolerance. The aim is *non-regressive repair*, not isolated improvement on the shifted data.

**Baselines.** No edit, continued Adam, matched sparse Adam, Jacobian/Gauss–Newton, low-rank adapter, full fine-tune, random edits.

**Disconfirming evidence.** Repairs cannot be localized, or standard optimizers reach equal/better performance and preservation at lower cost.

### RQ7 — Does Hugging Face Weight-Space Programming offer anything beyond numerical repair?

**Question.** Can an open-weight instruction model use weight and diagnostic structure to choose **which parameters to change and how**, beyond restating a provided least-squares answer or memorizing familiar network patterns?

**Hypothesis H7.** The LLM could be useful for *structural* repair proposals when the correct edit class is not predetermined; success on scalar coefficient regression alone is not sufficient.

**Controls:** at least **three independent HF instruction checkpoints**; identical edit budget; independent held-out validation; invalid-output rates; random edits; permuted/incorrect diagnostic summaries; renamed hidden units / permutation-equivalent networks; numerical methods; trained adapters; same prompt budget. Never give the LLM the reference weights or the closed-form target update.

**Disconfirming evidence.** The LLM fails the diagnostic-permutation audit, or numerical/Adam edits dominate accuracy–cost, or success depends on leaked semantics/known benchmarks. Cite the independent empirical critique of weight reading [R8].

### RQ8 — Can a repair learned at one resolution transfer to other resolutions?

**Question.** If CCNN's learned operator shares a parameterization across \(N=16,32,64,128\), does a repair at \(N=16\) improve the same physical defect at larger \(N\) **without changing the repair**?

**Hypothesis H8.** Operator-level repair should transfer better than edits tied to a fixed vector of discretized channels when the same underlying mechanism is defective.

**Controls.** Parameter edits at 16 only; separate per-resolution retraining; naïve interpolation of edited filters; resolution-neutral neural operator baseline. Train/choose repairs on **16-channel validation data only** and evaluate larger grids strictly as test.

**Disconfirming evidence.** Repairs break stability or lose effectiveness as the grid changes, or other resolution-neutral repair schemes transfer equally well.

### RQ9 — Only after passing the toy gates: real-world applications

**Question.** Do gains transfer to music, sensor streams or other irregularly sampled data with genuine local/global structure?

**Hypothesis H9.** A structure-matched hybrid might improve cross-recording or cross-domain metrics and remain repairable under controlled shift.

**Conditions.** Recording-disjoint and, where feasible, artist/session-disjoint splits; meaningful long horizons; onset/transient and high-frequency metrics; tuned GRU/TCN/Transformer/Neural ODE and suitable pretrained forecasters. The present music results are **not** proof of this claim.

## 5. Minimal experimental programme — no architecture proliferation

| Phase | Minimal decisive test | Primary comparisons | Advancement gate |
|---|---|---|---|
| **T0 — Reproduce** | Repeat existing irregular-time long-gap toy and nonlocal negative control | Attention, +CNN, +CCNN, +Neural ODE | Comparable metrics and full cost accounting |
| **T1 — Resolve spatial transfer** | Train \(N=16\), test \(N=32,64,128\) | CCNN, grid-aware CCNN, pointwise ODE, FNO, attention | Demonstrable cross-resolution benefit not explained by discretization confounds |
| **T2 — Isolate attention gain** | Place winning operator inside an attention residual block | Attention, +static CNN, +generic ODE, +CCNN | Improvement attributable to coupling, not parameter count |
| **T3 — Repair** | Freeze hybrid; inject a known localized defect; repair at \(N=16\) | Numerical, Adam, random, HF-WSP | Verified shifted-domain gain with original-domain non-regression |
| **T4 — Stress transfer** | Apply the *same* repair at 32/64/128 and altered gaps | Repaired hybrid and matched controls | Stable, reproducible transfer rather than one-grid overfit |
| **T5 — Domain application** | Real recording-disjoint music or sensor data | Strong sequence and pretrained baselines | Robust task-level benefit with a reasonable compute trade-off |

**Do not build T3–T5 on an unexplained failure of T1/T2.** If a generic Neural ODE wins, investigate why CCNN adds no value; change the mathematics only in response to a diagnosed limitation (e.g., grid inconsistency, harmful diffusion, insufficient nonlocal transport, inability to retain sharp events).

## 6. Measurement and decision protocol

Pre-register before running any experiment:

1. **Outcome:** primary error metric; secondary error, stability and compute metrics; what counts as an unacceptable regression.
2. **Dataset:** generative process, distribution shift, shared continuous reference, train/validation/test split, boundary conditions, random seeds.
3. **Fairness:** comparable trainable parameters, hyperparameter search, observation timing, training budget, solver tolerances, and hardware. Equal epochs alone are not equal compute.
4. **Statistics:** prefer at least ten independent seeds for claims, paired per-seed results, bootstrap confidence intervals or equivalent, and the distribution of individual failures. The preliminary three-seed toy is hypothesis-generating.
5. **Negative controls:** deliberately nonlocal dynamics, zero/altered diffusion, randomized spatial neighborhoods, permuted WSP diagnostics, no-defect repair controls.
6. **Proposed-method highlighting:** comparison plots must show **absolute metrics**, **delta vs a matched baseline**, **confidence intervals**, and **runtime**. Highlighting the proposed model must never imply it wins.
7. **Status discipline:** `not_run`, `failed_to_load`, `invalid_proposal`, `rejected`, `accepted`, and `evaluated` are distinct. No missing model gets a fabricated score; do not treat a rejected WSP repair as a successful one.

**Provisional decision rule, to freeze before the first confirmatory study:** advance a CCNN contribution only when it gives a repeatable, useful improvement in the designated out-of-distribution metric with no unacceptable in-domain regression, *and* it is not dominated on error/cost by the strongest matched alternative. Avoid selecting thresholds after looking at test results. An illustrative threshold for planning is \(\ge5\%\) relative error reduction with a paired confidence interval excluding zero; the actual threshold should be specified per task before the experiment.

## 7. Research-agent operating protocol: preventing goal drift

Every new experiment, notebook or GitHub PR MUST begin with the following **Research Identity Record**:

```yaml
research_goal: "Improve attention through coupled continuous dynamics; assess non-regressive weight repair"
primary_question: "RQ# from this charter"
hypothesis: "Falsifiable one-sentence prediction"
mechanism_under_test: "CCNN operator / attention augmentation / WSP repair"
novelty_vs_prior_art: "What is not already shown by Conformer, Hyena, Mamba, FNO, or WSP"
minimum_baselines: ["named, matched implementations"]
negative_control: "task where the method should not have an unfair advantage"
primary_metric: "one predeclared measure"
compute_budget: "training/inference/runtime/solver requirements"
train_validation_test_boundary: "how leakage is prevented"
acceptance_and_stop_rules: "defined before test run"
evidence_status: "proposed | implemented | smoke_tested | evaluated | replicated"
```

**Before changing direction, the agent must:**

- State the **original goal** and current **RQ identifier**.
- Summarize the latest **confirmed** observation and uncertainty; do not report draft or smoke-test outcomes as peer-reviewed findings.
- Explain which mathematical assumption failed and **why** the proposed change addresses that failure.
- Identify the new baseline required to prove that the change matters.
- Update this charter **only with a reasoned amendment**, preserving the superseded question and experiment history. Do not silently replace the objective.
- Prefer a minimal toy test before adding architectures, datasets, notebooks, diagrams, or complexity.

## 8. Distinguishing novelty from known approaches

- **Conformer** [R2] already combines convolution and attention. Our novelty cannot be “convolutions help attention.”
- **Hyena** [R3] already uses long convolution and input-dependent gates; the mere presence of gates or long filters is not novel.
- **Mamba** [R4] already uses input-dependent state evolution; a generic selective recurrence is not new.
- **Neural operators/FNO** [R5] already target discretization-independent mappings; sharing model weights across grids is not novel by itself.
- **Weight-Space Programming** [R6] already proposes neural-weight interpretation and editing. Its published demonstrations do not establish that LLM-proposed edits outperform numerical repair for this CCNN problem.
- **Mechanistic WSP critique** [R8] cautions that observed performance may reflect local arithmetic and familiar-dataset priors rather than general weight-flow reasoning. Controls are essential.

**Candidate scientific contribution:** characterize the *conditions* in which a conservative, resolution-aware **coupled field–global dynamical operator** adds value to attention *beyond* Neural ODEs/neural operators, and whether **mechanism-localized repairs** transfer across both spatial resolution and temporal shift while preserving previous capabilities. This is a **hypothesis to test**, not an established novelty claim.

## 9. Current decisions and immediate next action

- **Decision A:** attention remains in the architecture; CCNN is an augmentation.
- **Decision B:** use the established negative toy results to motivate *spatial-resolution transfer*, not to claim an attention breakthrough.
- **Decision C:** first test the old discretization against a physically scaled conservative operator; do not arbitrarily complicate CCNN.
- **Decision D:** WSP is a **competing proposal method** at the repair stage, compared with standard optimization and random edits.
- **Decision E:** do not create another broad music notebook yet; complete the decisive toy experiments and log their outcomes first.

**Next executable test (T1):** one Colab-ready spatial-transfer toy with \(N_{\rm train}=16\), \(N_{\rm test}=\{16,32,64,128\}\), a local generator *and* a nonlocal negative control, shared reference trajectories, comparable resolution-neutral baselines and explicit numerical-solver cost. Only if T1 isolates a structural advantage should we proceed to attention augmentation (T2).

## References

- **[R1]** M. Habiba and B. A. Pearlmutter, “Continuous Convolutional Neural Networks: Coupled Neural PDE and ODE,” *arXiv:2111.00343* (2021). https://arxiv.org/abs/2111.00343
- **[R2]** A. Gulati et al., “Conformer: Convolution-augmented Transformer for Speech Recognition,” *arXiv:2005.08100* (2020). https://arxiv.org/abs/2005.08100
- **[R3]** M. Poli et al., “Hyena Hierarchy: Towards Larger Convolutional Language Models,” *ICML* (2023). https://proceedings.mlr.press/v202/poli23a.html
- **[R4]** A. Gu and T. Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” *arXiv:2312.00752* (2023). https://arxiv.org/abs/2312.00752
- **[R5]** N. Kovachki et al., “Neural Operator: Learning Maps Between Function Spaces,” *JMLR* (2023). https://arxiv.org/abs/2108.08481
- **[R6]** R. Takehi, J. Teoh, A. Zhang, and O. Khattab, “Sparks of Weight-Space Programming,” research demonstration (November 2026). https://rikiyat.github.io/blog/weight-space-programming/
- **[R7]** R. T. Q. Chen et al., “Neural Ordinary Differential Equations,” *NeurIPS* (2018). https://arxiv.org/abs/1806.07366
- **[R8]** N. S. Ahmed, “Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring 'Weight-Space Programming',” *arXiv:2611.08070* (2026). https://arxiv.org/abs/2611.08070

---

**Research continuity rule:** Cite this charter's RQ ID and version in every future experiment. A new notebook is a test instrument; **the goal is the invariant research question, not notebook production**.

## RQ4 evidence log — 15 November 2026 (preliminary)

**Experiment:** [Spatial-resolution transfer toy](RQ4_CCNN_Spatial_Resolution_Transfer_Colab.ipynb) and [three-seed local CPU evidence](rq4_results_preliminary/README.md).

**Status:** implemented and locally executed with three seeds; **not** independently replicated, compute-matched, or executed in Colab. Train grid N=16; test N=16/32/64/128; local-diffusion generator and nonlocal negative control.

- **H4a supported in toy:** scaled conservative diffusion beats an unscaled stencil at finer grids.
- **H4b contradicted in toy:** finite-volume CCNN does not beat a spectrally consistent coupled operator; the latter nearly matches the local reference because they share its operator family.
- **H4c contradicted in toy:** diffusion-only structure does not generalize to the nonlocal interaction as well as the FNO-style baseline.
- **H4d contradicted by predeclared tolerance:** scaled CCNN remains finite but its relative error grows as resolution increases.
- **H4e inconclusive:** runtime measurement compares different numerical strategies and cannot establish efficiency.
- **H4f not tested:** no weight edit is attempted yet.

**Decision and next action:** remain within RQ4/T1. Diagnose coarse-grid dispersion/parameter compensation, then test a genuinely learned nonlocal or adaptive operator against a tuned Fourier neural operator and attention baseline with fair compute budgets. **Do not claim CCNN superiority, add Transformer-scale integrations, or pursue WSP repair merely because the naive-stencil comparison is positive.** Preserve all earlier questions and negative-control results.
