# CCNN Research Wiki — Session Findings, Decisions, and Next Action

**Research checkpoint:** 18 November 2026  
**Repository:** [mansura-habiba/ode_nn](https://github.com/mansura-habiba/ode_nn)  
**Scientific status:** **No validated CCNN attention improvement or fault-specific weight-repair advantage.** Original PDE–ODE-aligned repair experiment **not yet implemented or executed**.  
**Current decision:** **Option B first**, one bounded experiment of the *original architecture*; **Option A second**, a separate SIM-inspired continual-research programme.  
**Canonical tracker:** [Issue #4 — CCNN attention + repair](https://github.com/mansura-habiba/ode_nn/issues/4)  
**Document role:** A reader-oriented, durable summary of the 15–18 November research discussion and repository evidence. Existing issue #4 remains the living checklist; this wiki is the session checkpoint, not an alternative epic or an additional hypothesis catalogue.

> **Scientific verdict in one sentence:** Exploratory CCNN lag-kernel results were strong only when the teacher matched its structure; later MQAR models learned the task but did not use the CCNN memory branch; proposed weight-repair and open-weight LLM advantages remain **unmeasured**, so only **one final faithful-architecture test** is justified before deciding to stop.

## Contents

1. [The central research question](#1-the-central-research-question)
2. [What the original architecture actually is](#2-what-the-original-architecture-actually-is)
3. [Chronology and decisions](#3-chronology-and-decisions)
4. [Measured results and what they do not establish](#4-measured-results-and-what-they-do-not-establish)
5. [Repair methods and LLM experiment](#5-repair-methods-and-llm-experiment)
6. [Research interpretation and limitations](#6-research-interpretation-and-limitations)
7. [Next action — Option B](#7-next-action--option-b)
8. [Subsequent action — Option A](#8-subsequent-action--option-a)
9. [Source and artifact index](#9-source-and-artifact-index)

## 1. The central research question

**Primary question retained across the session:** Under which dynamical conditions does the **original, time-varying CCNN causal lag-memory / coupled PDE–ODE architecture** contribute information to attention, enable selective weight-space repair, or **both**?

These are **distinct endpoints**:

- **Attention contribution:** improve recall/prediction, or provide complementary information beyond adequately trained attention, FIR, SSM and other competitive controls.
- **Weight-space repair:** after **verified functional damage** to a trained model, restore affected behavior while preserving unaffected behavior, using a useful and identifiable weight-editing space.
- **LLM × CCNN extension:** examine whether an **open-weight LLM** makes better *actual* edit proposals from semantically correct CCNN diagnostics than from otherwise information-matched anonymous/mislabelled weights; measure this **against numeric editing and placebo**, not as a substitute for repair evidence.

A positive result in any one endpoint **does not imply** a positive result in the others. An outcome may be attention-only, repair-only, both, or neither. Negative/boundary findings must remain visible.

## 2. What the original architecture actually is

The canonical source is **Habiba & Pearlmutter (2021), _Continuous Convolutional Neural Networks: Coupled Neural PDE and ODE_**, [arXiv:2111.00343](https://arxiv.org/pdf/2111.00343). The governing time-delay convolution is

\[
m_\theta(t)=\int_0^T K_\theta(\tau,t)y(t-\tau)\,d\tau.
\]

The lag-domain accumulated state is

\[
\partial_\tau u(t,\tau)=K_\theta(\tau,t)y(t-\tau),
\qquad u(t,0)=0,
\]

coupled to the time-domain dynamics

\[
\dot y(t)=f(u(t,T)).
\]

**Non-negotiable distinctions:**

- The PDE coordinate **\(\tau\)** is **historical lag**, **not spatial diffusion**.
- The **PDE lag integral and coupled ODE** define the architecture; a filter without the coupling is not enough to claim architectural reproduction.
- The paper allows a **learnable, time-varying kernel**. Our later five-parameter Gaussian/chirp kernel is a **research-specific approximation**, **not a limitation imposed by the 2021 paper**.
- The notebook's \`hist\` + five-parameter \`CCNNLag\` + fixed \`ode_trace\` leaky recurrence + attention-logit bias constitute a **simplified experimental adapter**, **not a faithful original PDE–ODE solver**.
- An extended transport-history PDE for \`q(t,tau)=y(t-tau)\` is a **possible mathematical implementation extension**, not an equation to attribute to the original authors.
- The 2021 paper does **not** claim Transformer attention superiority, LLM-driven repairs, or competitive neural weight editing.

**Before implementing an original-aligned arm**, freeze the exact state/history/initial-boundary contract, representation of \`K(tau,W,t)\`, lag discretization/interpolation, external driving inputs, coupled ODE solver and training/gradient convention. Mark source-specified facts versus implementation choices.

## 3. Chronology and decisions

| Period / decision | What changed | Outcome or status |
|---|---|---|
| **15 Nov — V3 evidence** | Three-seed synthetic comparison across matched chirp and five unmatched teachers; repair arms and zero-defect checks | CCNN strong on matched chirp, **lost to dynamic FIR on all five unmatched families**; repair specificity failed |
| **16 Nov — research focus clarified** | Retain **both** attention augmentation and weight repair as separable endpoints | No broad superiority claim; published Zoology MQAR selected for controlled attention evaluation |
| **16–17 Nov — first repair Colab** | Same faulted attention–CCNN checkpoint; structured/tangent/FIR/Adam; sham placebo; calibration and locked test | Engineering smoke completed; full PILOT initially gave **zero eligible repair comparisons** |
| **17 Nov — dataset/coupling revision** | Shift the diagnostic PILOT from 8,192-token/6k-example MQAR to **official Zoology generator with 256-token/10k-example diagnostic profile**; add query/key-conditioned memory bias | Clean recall became excellent; **CCNN branch still nearly unused** |
| **17 Nov — actual PILOT evidence** | Two seeds; measure memory-off ablation and fault calibration | **0/2 ready**; **0/8 fault-sensitive**; **48 skipped method records** — *not a negative repair-method benchmark* |
| **17 Nov — LLM proposal extension** | Add raw vs correctly labelled CCNN vs permuted-label prompt arms; blinded vector proposal; no-fault placebo | Proposed/test code exists; **no qualifying real LLM repair run**, no LLM improvement claim |
| **17 Nov — open-weight model choice** | Local Hugging Face **Qwen/Qwen3-4B-Instruct-2507**, 4-bit where available, rather than proprietary API model | Notebook integration exists; local-GPU weight-edit efficacy **not verified** |
| **17 Nov — original architecture correction** | Reassert **full history-lag PDE and coupled ODE**, not simplified five-coordinate adapter or spatial diffusion | Architecture fidelity contract required before further research inference |
| **17 Nov — SIM paper reviewed** | [arXiv:2611.18856](https://arxiv.org/abs/2611.18856), Self-Improvement Memory / continual recursive self-improvement | Potential **separate research-process direction**, not proof of CCNN repair |
| **17–18 Nov — final sequence** | **Option B, then Option A**; no parallel speculative implementation | B is **protocol design / source-fidelity freeze**; A **not started** |

**Repository state caution:** [PR #3](https://github.com/mansura-habiba/ode_nn/pull/3) was **merged**. Some later experiments and the focused notebook were subsequently edited **directly on** [\`research/ccnn-v3-falsification-executable\`](https://github.com/mansura-habiba/ode_nn/tree/research/ccnn-v3-falsification-executable), and **must not be assumed present on \`master\`**. Preserve links with explicit branch names. This wiki lives on \`master\`.

## 4. Measured results and what they do not establish

### 4.1 V3 synthetic lag-kernel comparison: evidence of a narrow match

The [V3 evidence ledger](../papers/CCNN_V3_Execution_Evidence_Ledger.md) reports **three independent seeds, 12 training epochs, 20 repair steps**. Below are **held-out MSE means** from that exploratory run; **lower is better**.

| Temporal teacher | Attention only | Dynamic FIR | CCNN + attention | Measured leading arm among these |
|---|---:|---:|---:|---|
| Matched chirp | 0.013898 | 0.006847 | **0.000629** | CCNN |
| Unmatched bimodal | 0.053042 | **0.007132** | 0.052631 | FIR |
| Unmatched boxcar | 0.045920 | **0.009460** | 0.045244 | FIR |
| Unmatched damped oscillator | 0.017636 | **0.006903** | 0.020656 | FIR |
| Unmatched switching | 0.068865 | **0.012687** | 0.074707 | FIR |
| Unmatched feedback ODE | 0.069277 | **0.014478** | 0.074905 | FIR |

**Finding:** a potentially useful *matched-structure inductive bias*. **Not established:** general CCNN superiority, convergence under equal compute/capacity, or an effect from the **faithful 2021** coupled PDE–ODE implementation. Several controls and generator properties confound direct generalization; **do not** promote matched chirp as a publication-level success.

### 4.2 V3 repair: mechanism-specificity invalidated

From the same ledger:

- **87/90 zero-defect trials accepted edits (96.7%)**, although there was no fault. This is consistent with continued fitting rather than fault-specific repair.
- **255/270 shifted repairs** passed the independent source-preservation tolerance, so **15/270 regressed** on source behavior.
- **254/270 shifted repair arms** accepted updates on validation. **Acceptance is not independent test recovery.**
- The older 56-parameter "structured" arm was **CCNN plus an unstructured residual**; never call it a pure five-parameter structured repairer.

**Verdict:** **repair benefit not demonstrated**. The no-fault acceptance rate is a substantive experiment-design failure.

### 4.3 Zoology MQAR: clean model learns but does not use the CCNN branch

The latest user-reported **PILOT** used the **pinned upstream Zoology MQAR generator**, with a smaller diagnostic profile (**vocabulary 256; 64-token sequences; 4 key–value pairs; 10,000 training examples**). It was **not** a reproduction of the original ICLR 2024 Figure-2 8,192-token published setup.

| Checkpoint seed | Clean validation recall | CCNN memory-off minus memory-on validation NLL | Best epoch | Readiness |
|---|---:|---:|---:|---|
| 11 | **0.999792** | **0.0000001299** | 34 | Failed: branch not contributing |
| 23 | **1.000000** | **0.000485826** | 34 | Failed: branch not contributing |

**Calibration damage in affected-query NLL** for the four fault classes:

| Seed | Lag center | Bandwidth | Query dependence | Off-surface signal |
|---|---:|---:|---:|---:|
| 11 | 0.000004 | 0.000004 | 0.000002 | 0.000001 |
| 23 | 0.000227 | 0.000061 | 0.000227 | 0.002976 |

Fault detection threshold had been **+0.05 NLL**, so **0/8 faults qualified**, and **0/2 models passed memory readiness**. The notebook generated **48 \`unready_model\` rows**, **not 48 evaluated repair results**. All numerical repair editors and LLM repair **were skipped**; a blank scatter plot reflected absence of eligible experiments, **not identical zero improvement by every method**.

**Finding:** ordinary attention learned easy MQAR almost perfectly while this proposed simplified/query-conditioned CCNN add-on had negligible *incremental* contribution. Thus perturbing its mostly unused parameters produced negligible damage. **This cannot test whether the original 2021 PDE–ODE CCNN is repairable.**

### 4.4 Engineering work versus scientific evidence

The focused [Colab repair notebook on the research branch](https://github.com/mansura-habiba/ode_nn/blob/research/ccnn-v3-falsification-executable/notebooks/CCNN_Weight_Repair_Boundary_Colab.ipynb) includes frozen-data hashing, calibration/validation/test separation, numerical editing surfaces, a strict readiness gate, and optional local Hugging Face proposals. **Smoke runs passing means code executed**, not that any scientific claim was validated. Earlier snapshot filename collisions between SMOKE and PILOT were fixed by separating dataset profiles.

The [published Zoology baseline runner on the research branch](https://github.com/mansura-habiba/ode_nn/blob/research/ccnn-v3-falsification-executable/benchmarks/ccnn_falsification/run_official_zoology_fig2_baselines.py) contains code for validation-selected checkpoints and a locked test, but **a real complete native training run of that selection path was not established in the session**. Do not report published baseline dominance or model equivalence.

## 5. Repair methods and LLM experiment

The proposed comparison uses **one fixed, trained, functionally damaged checkpoint** per fault and identical train/validation/calibration/test splits. Candidate methods:

| Arm | Intended conclusion / control |
|---|---|
| Nonlinear structured CCNN edit | Does a constrained original-architecture parameterization make the damage selectively repairable? |
| Local tangent / Jacobian edit | Is the benefit merely local geometry? |
| Matched-size FIR / spectral edit | Is comparable flexible convolution better? |
| Sparse Adam on **same coordinates** | Does ordinary optimization explain apparent CCNN improvement? |
| Full Adam / more flexible adaptation | Strong high-capacity recovery control; report extra budget |
| No-edit, **zero-defect placebo** | Eliminate false repairs and continued-training confounding |

**Necessary tests:** detectable functional fault, held-out affected recovery, unaffected-source preservation, no-fault placebo adjustment, edited parameter count and latency/compute, plus out-of-manifold faults that should challenge the structural editor.

**LLM hypothesis (not demonstrated):** after a *genuinely recoverable* fault, ask the **same locally run open-weight model** to propose bounded edits from (i) anonymous weights, (ii) mathematically accurate CCNN semantics, and (iii) permuted/misleading labels, with **information-matched numerical probes**. Validate any proposal separately and log abstentions, invalid JSON, prompt tokens, compute and placebo. Selected model in the notebook: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). **No real LLM efficacy results exist yet** because there were no eligible damaged checkpoints.

**Do not conflate:** outputting JSON edits, successfully executing a local model, or interpreting a kernel plot is **not** evidence that an LLM understands or repairs neural weights.

## 6. Research interpretation and limitations

**Supported from reported observations:**

1. A reduced CCNN kernel approximation has an exploratory **matched-chirp** advantage and loses to dynamic FIR in **five unmatched** generator families.
2. The V3 repair procedure accepts far too many **zero-fault** edits; fault-specific benefit is not identifiable.
3. Small-MQAR training can reach ~100% recall **without meaningful dependence** on the optional CCNN branch, even after query-conditioned modulation.
4. Under those conditions, corrupting CCNN branch weights does not create substantial repair headroom; the proposed numerical and LLM methods were therefore **not compared**.

**Not established:**

- Original 2021 coupled PDE–ODE architecture under the new repair setup.
- General attention improvement or a fair published-baseline win.
- Selective repair superior to FIR, tangent, sparse/full Adam, or no-edit.
- Open-weight LLM × CCNN semantics outperforming raw or permuted weight descriptions.
- Whether a CCNN **compensatory adapter** can repair *active attention faults*. This was a proposed alternative, not an executed experiment.
- Whether the SIM paper improves the neural model or salvages its scientific claims. It addresses **agent research/optimization memory**, a different object.

**Central methodological lesson:** before diagnosing or repairing model weights, confirm that the affected mechanism is **functionally active**. A model can solve the benchmark beautifully while the proposed new module contributes effectively nothing.

## 7. Next action — Option B

**Decision:** one **bounded final** test of the original 2021 architecture; **do not** begin yet another general sweep or keep adapting the task until CCNN succeeds. A result may be a positive conditional advantage, a sharp failure boundary, or a negative verdict.

### B1. Next immediate task: freeze the faithful architectural contract — **NOT DONE**

Read the primary paper and explicitly specify:

- [ ] Time-varying \`K(tau,W_K,t)\` and its learned parameters; what is prescribed vs chosen.
- [ ] Initial/history state including negative-time behavior, \`u(t,0)=0\`, lag-grid resolution, boundary/interpolation and variable timestamps.
- [ ] Exact coupling from integrated \`u(t,T)\` to a **time-evolving** ODE state, rather than a leaky stand-in; external input interface and gradient/adjoint strategy.
- [ ] What can be edited (kernel, ODE, coupling), identifiability and a fair equal-parameter-control mapping.
- [ ] Minimal numerical validity checks; solver checks **cannot** substitute for functional evidence.

**Deliverable:** a **one-page source-to-mathematical-contract**, annotated with "2021 source", "necessary numerical approximation", and "new experimental extension", appended **in the existing canonical research document / issue #4**, not a new notebook or pile of test files. Explicitly freeze what will be coded and compared.

### B2. Pre-register the single decisive experiment

- [ ] Choose **one independent published or externally defined delayed-history task** that can expose the original CCNN operator but is **not generated from the same chirped-Gaussian family**. Keep a competitive FIR/SSM/attention control.
- [ ] Train one original-architecture-aligned checkpoint; demonstrate reasonable convergence, meaningful memory ablation **and** calibration-set fault damage. **Stop if the mechanism is still unused**; do not lower gates.
- [ ] Evaluate same-checkpoint structured repair, tangent, capacity-matched FIR, same-coordinate sparse Adam, full Adam, no-edit and no-fault placebo with locked test evaluated after validation selection.
- [ ] Report **effect → source preservation → cost → fault class → boundary**, including methods that beat CCNN. Replicate once independently if a credible signal exists; do not manufacture a winner by iterative teacher selection.
- [ ] If no active memory/detectable fault or no conditional improvement under fair controls, **close B as unsupported**, preserving complete results and provenance. This is a valid scientific stop.

**Next responsible actor / action:** researcher or coding agent to do **B1 source-to-contract in issue #4**. **Do not launch GPU runs or LLM calls yet.**

## 8. Subsequent action — Option A

**Only after the B go/no-go decision**, investigate a **separate** paper-inspired research question:

> Can a continuously improving research agent preserve experimental evidence with its validity conditions, identify when previous conclusions cease to transfer after benchmark/architecture changes, and avoid repeating invalidated work?

**Motivating paper:** [arXiv:2611.18856](https://arxiv.org/abs/2611.18856), discussed in this session as a **Self-Improvement Memory (SIM)** / continual recursive self-improvement approach. Its relevance here is **research-optimization memory**, **not biological/CCNN temporal memory**. Reported paper claims are not replication results of this repository.

**A protocol (not started):**

- [ ] Use appropriate baselines: **Cold Start**, **Harness Transfer**, **Problem-Only Memory**, and **Full SIM**. Do not compare only with an agent having no memory.
- [ ] Store attempted edit, rationale, predicted and actual effect, benchmark version and validity conditions.
- [ ] Test **validity-aware transfer** when experimental conditions change: stale conclusion retention, repeated failed experiments, useful hypothesis transfer, evidence provenance and compute per meaningful improvement.
- [ ] Include the CCNN project's negative/revised results as **one case study**, not as a proof of the method. Require additional independent research tasks.
- [ ] Begin Option A as an independent scientific protocol only after formally closing or advancing B; no concurrent scope expansion.

## 9. Source and artifact index

**Primary papers and benchmark**

- [Habiba & Pearlmutter, original CCNN/PDE–ODE paper (2021)](https://arxiv.org/pdf/2111.00343) — architecture and source-fidelity constraints.
- [Zoology: Measuring and Improving Recall in Efficient Language Models (ICLR 2024)](https://arxiv.org/abs/2312.04927); [pinned upstream MQAR generator](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/data/multiquery_ar.py) — published external task used in diagnostic experiments.
- [Self-Improvement Memory / CRSI paper (2026)](https://arxiv.org/abs/2611.18856) — Option A inspiration, **not** a CCNN repair result.
- [WeightBench (2026)](https://arxiv.org/abs/2611.18553) — motivation for rigorous LLM weight-space editing controls.

**Repository evidence and implementation**

- [Single research issue #4](https://github.com/mansura-habiba/ode_nn/issues/4) — authoritative current execution decisions, gates, and checklists.
- [V3 measured evidence ledger on \`master\`](../papers/CCNN_V3_Execution_Evidence_Ledger.md) — exact synthetic metrics and failed repair specificity.
- [Original research-question charter and fidelity correction](../notebooks/archive/RESEARCH_QUESTIONS_CCNN_ATTENTION_WEIGHT_REPAIR.md) — historical scope; later correction to lag PDE takes precedence over spatial-diffusion sections.
- [Focused weight-repair Colab on research branch](https://github.com/mansura-habiba/ode_nn/blob/research/ccnn-v3-falsification-executable/notebooks/CCNN_Weight_Repair_Boundary_Colab.ipynb) — proposed methods, readiness/LLM code, **not** original-architecture proof.
- [CCNN attention + repair paper-style Colab on research branch](https://github.com/mansura-habiba/ode_nn/blob/research/ccnn-v3-falsification-executable/notebooks/CCNN_Attention_Weight_Space_Repair_Paper_Colab.ipynb) — published-benchmark experimental design.
- [Early paper draft on \`master\`](../papers/CCNN_Attention_Structured_Weight_Repair_Paper_Draft.md) — **historical preliminary matched-teacher findings**, not the final current verdict; its small synthetic successes must not override the later V3 and MQAR negative controls.
- [Merged PR #3](https://github.com/mansura-habiba/ode_nn/pull/3) — prior experiment implementation history; see explicit branch paths above for later edits.
- [Repository research-agent instructions](../AGENTS.md) — resume source-fidelity and existing issue before new code; avoid redundant artifacts.

---

### Resume checkpoint

**Current scientific verdict:** conditional matched-kernel indication only; no validated original-CCNN attention or repair advantage.  
**What changed in the session:** a supposedly strong MQAR model proved to **ignore** its CCNN branch; zero-fault repair contamination was documented; original 2021 lag-PDE/ODE fidelity re-established; LLM and SIM approaches retained as **separate hypotheses**, not results.  
**Current stage:** **B1 original-source computational contract**, before implementing the last bounded experiment.  
**Next concrete action:** in **issue #4**, freeze the faithful **history-lag PDE → endpoint → time-ODE** implementation contract and matched control table.  
**After the B stop/go decision:** **Option A** validity-aware SIM-style continual research optimization.
