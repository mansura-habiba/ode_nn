# Structure-Aware Continuous Convolution for Attention Augmentation and Verified Neural Weight Repair

**Research manuscript — preliminary technical report, 15 November 2026**  
**Research identities:** RQ1-M (attention augmentation), RQ6-M (mechanism-specific repair), RQ7 (LLM-proposed weight edits, not yet evaluated)  
**Evidence level:** controlled synthetic pilot, two independent model seeds; no real-world claims

## Abstract

Attention mechanisms learn content-dependent interactions between observations, but they do not by themselves enforce a compact, continuously parameterized causal response over historical lags. Coupled Continuous Convolutional Neural Networks (CCNNs), in the formulation of Habiba and Pearlmutter (2021), represent an accumulated lag convolution through a partial differential equation (PDE), whose boundary value supplies an ordinary differential equation (ODE). We ask whether this mathematical structure contributes something useful to attention and whether its interpretable kernel parameters enable more precise, verifiable repairs than less structured neural-filter parameters. To investigate these questions independently, we construct a synthetic sequence-forecasting task driven by a time-varying Gaussian-windowed chirp filter. We compare attention with static convolution, query-conditioned discrete convolution, exponential-state memory, and a compact lag-kernel/PDE–ODE branch. We then inject a delay shift and evaluate one-parameter Jacobian, Adam, and random repairs against a one-coordinate unstructured filter repair. In a preliminary two-seed experiment, attention augmented with the structured lag kernel obtains a mean held-out mean-squared error (MSE) of **6.35 × 10⁻⁶**, compared with **8.23 × 10⁻⁴** for the strongest query-conditioned discrete convolution control. Sparse Adam repair reduces shifted-domain error from approximately **1.00 × 10⁻²** to **8.99 × 10⁻⁶**. The experiment deliberately gives the structured model a matched generative family; consequently the observations establish controlled feasibility, not general architectural superiority. Original-domain preservation is guaranteed by a domain-gated intervention rather than discovered by the optimizer. We outline nonmatching-generator, strong-adapter, computational-budget and open-weight language-model controls required before advancing the claims.

## 1. Introduction

The invariant research question is:

> **Can the mathematical structure of coupled continuous convolution make attention more effective and neural-network weight repairs more precise than less structured alternatives?**

The question contains two **independent** claims. First, an explicit response over time lag might complement content-addressed attention. Second, the semantic structure of a learned convolutional kernel (delay, bandwidth, chirp rate) might permit controlled local repairs after shifts in the data-generating mechanism. Neither claim entails the other. A conventional time-aware convolution or state-space model may match or exceed the proposed augmented attention mechanism, and a numerical optimizer may outperform a language-model weight proposer.

Our previous exploratory diffusion toys investigated a spatial operator, which is **not** the defining historical-lag convolution from the original CCNN paper [1]. The present investigation deliberately restores that distinction and does not rely on spatial diffusion or resolution-transfer outcomes to justify attention augmentation.

## 2. Source mechanism and adaptation

The original paper [1, Eqs. (2)–(6)] proposes a causal, time-varying convolution of the historical trajectory:

\[
\dot y(t)=f\!\left(\int_0^T K_\theta(\tau,t)y(t-\tau)\,d\tau\right).
\]

The accumulated lag-PDE state satisfies

\[
\frac{\partial u(t,\tau)}{\partial \tau}=K_\theta(\tau,t)y(t-\tau),\qquad u(t,0)=0,
\]

and supplies the ODE through \(\dot y(t)=f(u(t,T))\). The delay term in the PDE's right-hand side is a genuine implementation issue discussed in the paper; writing the system in terms of \(u\) does not by itself yield an autonomous Markov state without representing the history. The present toy supplies the observed history explicitly.

### 2.1 Proposed attention augmentation

Let \(x_i=x(t_i)\) be a scalar observed signal and \(q(t_i)\) an external, time-varying query. The proposal constructs a historical kernel response:

\[
m_\theta(t)=\int_0^T K_\theta(\tau,q(t))x(t-\tau)\,d\tau,
\]

which drives a leaky latent ODE \(\dot z=-\lambda z+\lambda\tanh(2m_\theta(t))\). A common self-attention backbone emits \(a_\psi(x,q)\); the augmented forecast is \(\hat y=a_\psi(x,q)+g z(t_{\mathrm{final}})\). We use cumulative lag quadrature and exact exponential stepping for piecewise-constant forcing in the pilot. These are discrete numerical approximations, not a complete continuous PDE solver implementation.

The kernel is a normalized Gaussian-windowed chirp with conditional center:

\[
K_\theta(\tau,q)\propto\exp\left(-\frac{(\tau-\mu_0-\mu_q q)^2}{2\sigma^2}\right)
\cos(2\pi[f_0\tau+c\tau^2]).
\]

The five principal kernel parameters are center, query-dependent displacement, bandwidth, carrier frequency, and chirp rate. In contrast, the unstructured dynamic-filter control learns 14 free kernel coefficients plus 14 query-response coefficients. Its extra freedom is a serious baseline, and it should not be dismissed as an ordinary static CNN.

### 2.2 Method diagram

```mermaid
flowchart LR
    A[Observed stream and query q] --> B[Common self-attention]
    A --> C[Explicit causal lag history]
    C --> D[Time-varying structured kernel K]
    D --> E[Lag-PDE cumulative integral]
    E --> F[Latent ODE state z]
    F --> G[Augmented prediction]
    B --> G
    A --> H[Query-conditioned discrete convolution control]
    H --> I[Baseline prediction]
```

The **novelty under examination** is not the existence of a continuous integral, an ODE, or a convolution alone. It is the *incremental utility* of the original lag-structured mechanism when used alongside attention, and the identifiability of its parameter repair surface.

## 2.3 Architecture and V3 falsification alignment

The detailed, versioned architecture specification is maintained in **[CCNN Architecture: Coupled Lag Dynamics and Gate-Free Structure-Aware Repair](CCNN_ARCHITECTURE.md)**. It relates the original Habiba–Pearlmutter lag PDE–ODE formulation to the implemented V3 discrete model and the planned mechanism-specific evaluations.

In the original source theory, the lag field accumulates delayed history and its boundary drives dynamical evolution. **V3 approximates this** with an explicitly constructed, zero-padded causal history `H(t,j)=x(t-j)`, a normalized query-conditioned Gaussian-chirp kernel, discrete cumulative lag sums, and exponential stepping of a leaky latent state. The output adds a learned-scale lag response to a common self-attention prediction; this is an attention augmentation experiment, not a claim to implement an adaptive continuous-time PDE/ODE solver.

For repair, V3 **freezes the source attention–CCNN model** and tests shared, ungated kernel edits on *both* source and shifted populations. Its budgets `1, 5, 18, 36, 56` compare structured/hybrid, spectral, rank-limited and full lag-by-query correction surfaces, with validation-only edit selection and independent test-domain retention. Above five adjustable scalars, “structured” denotes **structured plus spectral residual** rather than a purely five-parameter kernel intervention. A source-validation tolerance is not a guarantee of source-test preservation.

The supplied four-panel illustration maps to **(A)** original theory, **(B)** implemented V3 surrogate, **(C)** proposed shared-weight repair comparisons, and **(D)** decisive planned falsification tests. Unmatched kernel families and independent-domain evaluation are in V3; loss-landscape trajectories are a **planned measurement**, not an observed empirical figure. See the companion architecture for the equations, Mermaid diagrams, explicit repair-budget table and evidence boundary.

## 3. Research hypotheses and prior-defined evaluation

- **H-A: attention augmentation.** A structured lag kernel may outperform attention alone and comparably sized static/dynamic convolution and exponential-memory controls on a changing-delay matched-filter task.
- **H-B: structured precision of weight repair.** After a known delay shift, a one-scalar edit to the lag-kernel center may outperform a one-scalar edit to an arbitrary discrete-filter coordinate.
- **H-C: mechanism specificity.** A delay edit should not fully repair an unrelated chirp-frequency change; no edit should be needed if the generating system is unchanged.
- **H-D: reasoning-driven WSP.** One of three open-weight instruction models might propose a useful bounded edit, but there is no presumption that it improves on Jacobian/Adam optimization.

**Primary endpoints:** held-out test MSE for H-A and H-B, independent validation acceptance for H-B, and negative-control false-acceptance rate for H-C. Secondary endpoints include model parameter counts, training time, and response to changed query dynamics. Seed-level evidence should be retained; two seeds are too few for confirmatory statistical tests.

## 4. Experimental design

### 4.1 Representative toy data

The toy uses 48 observed time points, 14 historical lag positions, additive noise, randomized chirp-like motifs, and a time-varying query affecting the center of the matching kernel. The ground-truth target is an accumulated kernel response propagated through a leaky ODE. Each seed trains on 480 examples for 30 epochs and evaluates separate validation/test sequences. A second test split changes the range of temporal-query variation while preserving the kernel family.

The generating process was intentionally selected from the structured model family to isolate a possible inductive-bias effect; it is **not** an independent test of universal operator learning.

### 4.2 Attention and temporal baselines

The five systems use a shared small attention architecture. The control modules are: none (attention only), a static 14-lag filter, a query-conditioned 14-lag filter, a small exponential-memory SSM, and the proposed lag-kernel/PDE–ODE branch. They are not exactly parameter or compute matched; the training and inference budgets and parameter counts must be taken into account in later confirmatory studies.

### 4.3 Controlled repair

For a shifted operating condition, the target kernel center is displaced by 0.12 normalized lag units. **The base network is frozen.** A candidate scalar \(\delta\in[-0.3,0.3]\) is applied to the structured delay for shifted-domain examples only. For the less structured filter, the candidate edits one selected discrete filter coordinate. Both interventions use one scalar, though their induced function changes are not equally expressive. Jacobian repair uses local sensitivity, Adam optimizes one scalar, and random search proposes a bounded set of candidates. A candidate is accepted if it improves validation loss by at least 0.5% versus no edit.

Source-domain predictions never receive the new-domain adapter, so old-domain behavior is *unchanged by construction*. This is a useful safety design but **not** an empirical demonstration that unconstrained shared-weight changes can avoid regression. The same-domain validation decision and final test are kept separate.

## 5. Preliminary experimental outcomes (two seeds)

### 5.1 Predictive performance

| Model | Held-out MSE ↓ |
|---|---:|
| Attention | 0.0137637 |
| Attention + static filter | 0.00305383 |
| Attention + exponential SSM | 0.0135316 |
| Attention + query-conditioned discrete filter | 0.000822747 |
| **Attention + structured lag CCNN** | **0.0000063463** |

These means summarize two independent seeds (11 and 23). The structured lag kernel accurately matches the teacher family, while the dynamic unstructured filter is itself much stronger than attention alone. The relative advantage may shrink under nonmatching generators, longer sequences, adaptive numerical solvers, stronger state-space controls, or tuned capacity.

### 5.2 Mechanism-specific repair

| Repair arm | Shifted-domain test MSE ↓ |
|---|---:|
| Structured lag kernel — Jacobian edit | 0.000511476 |
| **Structured lag kernel — Adam edit** | **0.0000089911** |
| Structured lag kernel — random edit | 0.00293512 |
| Query-conditioned filter — Adam, one coordinate | 0.00870626 |
| Query-conditioned filter — Jacobian, one coordinate | 0.00867422 |

Before repair, the shifted structured model's test MSE is approximately 0.0100. Adam identifies a delay edit close to the data-generating displacement. The stronger result must be interpreted as **a favorable, low-dimensional fault/repair alignment**: one center shift changes many lag coefficients coherently, whereas the control's single-coordinate edit cannot shift its full response. A fairer next comparison will allow the dynamic-filter baseline to edit a full tangent direction or a trainable low-rank basis over lag coefficients.

### 5.3 Negative controls

Both seeds reject edits when no fault exists in the tested validation set. When the fault changes the chirp frequency rather than delay, optimizing delay alone reduces but does not eliminate the error (approximately 0.0227→0.0154 and 0.0268→0.0165 across the two seeds). This is expected from the restricted mechanism class; it is not a universal fault-diagnosis guarantee.

### 5.4 WSP outcome status

**Not evaluated.** The notebook offers an optional three-model Hugging Face inference arm using two Qwen sizes and SmolLM2. Without executed proposals, their generation time, validity rate, accuracy, and benefit over Adam/Jacobian methods are unknown. The LLM sees aggregate diagnostic statistics, not reference shift values; even a successful scalar proposal would not establish general weight-space reasoning.

## 5.5 Publication visualizations and figure provenance

The accompanying [executable publication notebook](../notebooks/CCNN_Lag_Kernel_Attention_Repair_Publication_V2_Colab.ipynb) generates five figure sets. All are exported as SVG/PDF and high-resolution PNG with a JSON figure manifest; the paper should reuse the vector exports rather than raster screenshots.

- **Figure 1 — Proposed mechanism and verification.** Panel A separates attention from the structured lag integral and coupled ODE. Panel B shows the restricted edit and independent acceptance gate. The heavy outlines identify the *proposed mechanism*; they do not imply a positive empirical result.
- **Figure 2 — Causal memory field.** A representative teacher signal and query produce a signed time-varying lag kernel, a discrete cumulative lag-integral state, and an evolving ODE state. This is a mathematical illustration, not learned model evidence.
- **Figure 3 — Architecture comparison with cost.** Held-out and shifted-query MSE are displayed with every available per-seed value, and cost is shown against prediction error. With two seeds, confidence intervals would not support confirmatory inference.
- **Figure 4 — Mechanism-specific weight edits.** A representative learned/shifted/repaired kernel, the validation loss as a function of edit magnitude, and the held-out error from Jacobian, Adam and random edits. Kernel shapes are not identifiable from scalar-output loss alone; the example is diagnostic rather than evidence of exact physical parameter recovery.
- **Figure 5 — Negative controls and model capacity.** No-defect edit acceptance, frequency-fault failure of delay-only repair, and unmatched parameter counts prevent a misleading positive-only narrative.

**Scientific visualization policy.** Never mix an illustrative teacher sample with held-out experimental estimates without labelling the distinction. The proposal is highlighted by a consistent outline or hatch, but bars and axes are not rescaled to manufacture a win. Exported data and the scripts generating them remain associated with the manuscript. This pilot is not a journal-grade benchmark: confirmation requires many seeds, strong alternative models, out-of-family sources, matched budgets, and transparent uncertainty estimates.

## 6. Discussion and limitations

The experiment supplies initial evidence that the original CCNN lag-kernel architecture is a useful *inductive bias* for matching a time-varying causal filter and that this structure creates an intelligible repair coordinate. It does **not** establish superiority on independent processes, solve irregular sampling in continuous time, or demonstrate a novel theorem of precision. The direct dynamic-filter baseline already captures much of the task, making its comparison central rather than optional. The use of an external query and a matched teacher limits what can be inferred about attention itself.

The domain-gated parameterization makes non-regression automatic. It cannot by itself justify a claim of successful non-regressive shared-weight fine-tuning. An unrestricted learned kernel may be repairable with few **function-space** degrees of freedom even when several raw weights must change; sparsity in parameter coordinates is representation-dependent. Any paper on repair precision should distinguish number of parameters, amount of function disruption, edit time, and preserved source-domain accuracy.

## 7. Conclusion and future research

The exploratory matched-generator result supports pursuing a **narrower, mechanism-based investigation** of structured historical convolution as an attention complement and repair interface. It does not authorize a broad claim that CCNN exceeds Transformers or SSMs. Future work has a clear order: (1) compare on nonmatching and mixed-mechanism generators with longer histories and missing/irregular timestamps; (2) allow flexible but compute-matched dynamic-kernel/SSM/Hyena-like competitors; (3) evaluate automatic fault localization and multi-coordinate repair under a genuine original-domain non-regression constraint; (4) run a three-model Hugging Face WSP audit with diagnostic corruption and weight-coordinate permutation controls; (5) repeat with ≥10 independent seeds and rigorous timing. A paper-level conclusion will depend on those tests, including outcomes that contradict the proposed advantage.

## References

1. M. Habiba and B. A. Pearlmutter. *Continuous Convolutional Neural Networks: Coupled Neural PDE and ODE*. arXiv:2111.00343 (2021). https://arxiv.org/abs/2111.00343
2. A. Vaswani et al. *Attention Is All You Need*. NeurIPS (2017). https://arxiv.org/abs/1706.03762
3. M. Poli et al. *Hyena Hierarchy: Towards Larger Convolutional Language Models*. ICML (2023). https://proceedings.mlr.press/v202/poli23a.html
4. A. Gu and T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752 (2023). https://arxiv.org/abs/2312.00752
5. R. Takehi, J. Teoh, A. Zhang, O. Khattab. *Sparks of Weight-Space Programming*. MIT CSAIL research demonstration (2026). https://rikiyat.github.io/blog/weight-space-programming/
6. N. S. Ahmed. *Can an LLM Actually Reason and Program about Neural Network's Weights?* arXiv:2611.08070 (2026). https://arxiv.org/abs/2611.08070