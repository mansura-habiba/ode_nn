# Executable research notebooks

> **Research direction (read first):** [CCNN × attention × verified weight-space repair — unified research charter](RESEARCH_QUESTIONS_CCNN_ATTENTION_WEIGHT_REPAIR.md). This charter defines the invariant goal, RQ1–RQ9, falsifiable hypotheses, toy-study decisions, baseline obligations, and the Research Identity Record required before new experiments. No more research-question drift or notebook proliferation without an explicit question and evidence gate.

> **15 November 2026 — scientific correction:** The original CCNN paper is primarily about a **causal time-lag convolution encoded through a coupled PDE–ODE**, not the spatial diffusion operator used in our exploratory RQ4 toy. The [research charter](RESEARCH_QUESTIONS_CCNN_ATTENTION_WEIGHT_REPAIR.md) now distinguishes the original lag-kernel mechanism (RQ1-M, RQ6-M) from the later spatial-operator extension. Negative diffusion-stencil results do not refute the original CCNN. The next method-level questions are whether the lag kernel **complements attention** and enables **structured, verified kernel-weight repair**.

This directory is the canonical location for research notebooks in `ode_nn`. Updates to the music PDE–ODE and PCWSR studies should be committed here, preserving a reviewable change history.

## Current RQ4 study — spatial-resolution transfer (15 November 2026)

- **[RQ4 Spatial-Resolution Transfer Colab](RQ4_CCNN_Spatial_Resolution_Transfer_Colab.ipynb)** — trains at 16 spatial channels, evaluates 16/32/64/128, compares grid-aware and unscaled CCNNs with a spectral CCNN, pointwise Neural ODE, FNO-lite and coordinate-aware attention on both local and nonlocal generators. Its live **hypothesis evidence ledger** reports supported, contradicted, inconclusive and untested outcomes, then generates a conclusion and next experiments.
- **[RQ4 preliminary three-seed evidence](rq4_results_preliminary/README.md)** and [raw evaluation metrics](rq4_results_preliminary/RQ4_preliminary_3seed_metrics.json) — a local CPU diagnostic, **not a general CCNN win**. Mesh scaling is supported relative to an unscaled stencil; the strong spectral baseline and nonlocal negative control challenge broader claims. Cross-resolution WSP repair remains untested.

**Research decision:** complete the operator-consistency and nonlocal tests before proceeding to attention augmentation (RQ1) or weight-space repair (RQ8). Preserve negative results.


## Current RQ6/RQ8 repair-feasibility pilot (15 November 2026)

- **[RQ6/RQ8 — Non-Regressive Operator Weight Repair Colab](RQ6_RQ8_NonRegressive_Operator_Weight_Repair_Colab.ipynb)** — tests a trained, resolution-neutral Fourier/coupled global operator under a conditional shift. A one-weight edit is selected by independent two-domain validation, then transferred without refitting from 16 to 128 channels. A parameter-matched *direct state-transition* control challenges the need for continuous coupling.
- **[Reproducible Python experiment](rq6_rq8_operator_weight_repair.py)** and **[three-seed evidence and hypothesis ledger](rq6_rq8_results_preliminary/README.md)**; [raw JSON](rq6_rq8_results_preliminary/RQ6_RQ8_3seed_metrics.json).

**Result:** one-weight repair is feasible for this favorable representable toy; sparse Adam performs extremely well but a direct-state-map comparator does almost as well. **No unique advantage of coupled continuous dynamics is demonstrated.** This is an exploratory precursor to RQ6/RQ8, not completion of the main CCNN–attention or WSP questions. Next: varying elapsed times, irregular observations and dynamic coupling against matched controls.


## Current studies

| Notebook | Proposed method | Primary evaluation |
|---|---|---|
| [Coupled Neural PDE–ODE](Music_Coupled_Neural_PDE_ODE_v2_Colab.ipynb) | Coupled spectral-field / global-state dynamics | One-step forecasting against ablations and persistence |
| [Complex-task forecasting](Music_Coupled_PDE_ODE_Complex_Task_v2_Colab.ipynb) | Autoregressive continuous-time spectral forecasting | Multi-step drift, transient error and future event dynamics |
| [Recording-level benchmark](Music_Persistent_PDE_ODE_Recording_Benchmark_v2_Colab.ipynb) | Persistent coupled spectral PDE–ODE state | Recording-disjoint evaluation, neural baselines, and 3 Hugging Face forecasting models |
| [PCWSR + Hugging Face WSP](Music_PCWSR_HuggingFace_WSP_v2_Colab.ipynb) | LLM-proposed neural weight edits with independent validation | Frozen vs Jacobian vs Adam vs 3 open-weight proposers |

## Required academic structure for all new revisions

1. Title and **abstract** in academic prose: problem, gap, contribution, experimental design, and claim boundary.
2. Introduction, literature and falsifiable research questions.
3. **Proposed method**, equations, scope and expected outcomes (explicitly *hypotheses*, not measured results).
4. Method diagrams implemented inside the notebook, clearly highlighting the proposed mechanism rather than decoration.
5. Step-by-step methodological explanations immediately before executable code.
6. Evaluation: independent splits, strong baselines, outcome-specific metrics, failures, compute budgets and plots emphasizing the proposed method.
7. Evidence-based discussion, limitations, **conclusion and future research**.
8. References to original papers, software, datasets and model cards.

Never present hypothetical numbers, failed model loads, rejected weight edits or synthetic-only performance as measured superiority. Code cells should not expose Hugging Face credentials.

## Colab / Hugging Face

These notebooks use Python/Google Colab. A private GitHub repository may require signing in to view or import notebooks. The default public Hugging Face checkpoints should not need `HF_TOKEN`; for a gated model, use the Colab **Secrets** panel and enable notebook access rather than storing tokens in source.

The recording-level study includes `ibm-research/ttm-research-r2`, `amazon/chronos-bolt-tiny` and `amazon/chronos-t5-tiny`. TTM R2 expects 512 true context steps and is pretrained for minute-to-hour time-series resolutions, not 16 ms audio spectra. Its out-of-vocabulary frequency token (0) is an explicitly **out-of-domain** experimental choice. A model that fails to load is logged as a failure rather than assigned an evaluation score. Chronos uses its own pipeline and is evaluated across selected spectral bands.

The weight-space study instead uses `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct` and `HuggingFaceTB/SmolLM2-1.7B-Instruct` as instruction-model edit proposers. These are different research roles from time-series forecasting checkpoints.

## Verification status

Notebook JSON and Python code syntax were checked in the authorship environment. **Full Google Colab training, checkpoint downloads, numerical comparisons, and multi-seed reproducibility are not yet verified.** Publish generated results only after running the corresponding notebook and recording software versions, device, random seeds and complete error logs.

## Future development

Subsequent notebooks should be added to this directory, using manuscript-style text plus executable methods. Proposed future experiments include event-triggered PDE–ODE dynamics, phase-aware spectral operators, JAX/Equinox/Diffrax baselines, and non-regressive neural weight repair under instrument/tempo shift.

## New V3 research directions (14 November 2026)

- **[Event-conditioned persistent PDE–ODE V3](Music_Event_Conditioned_Persistent_PDE_ODE_V3_Colab.ipynb)** — replaces the failed one-step/reset-at-every-step music methodology with one joint future solve of spectral field and latent state, history-predicted event forcing, multi-horizon loss, irregular observation masks, recording-disjoint evaluation, GRU/TCN/Transformer/neural ODE/persistence controls, and a **shared-subset 3-model Hugging Face transfer comparison**. Its main HF models are `ibm-research/ttm-research-r2`, `amazon/chronos-bolt-tiny`, and `amazon/chronos-t5-tiny`. Real recordings and full Colab execution are still required for any performance claims.
- **[Music PCWSR/WSP mechanistic audit V3](Music_PCWSR_HF_WSP_Mechanistic_Audit_V3_Colab.ipynb)** — evaluates three open-weight instruction models proposing edits to frozen music convolution weights. New controls deliberately permute the training residual projection or generate random bounded edits, testing diagnostic sensitivity without using those controls for candidate selection. It cites the critical empirical audit [Ahmed, arXiv:2611.08070v1](https://arxiv.org/abs/2611.08070), alongside [Hu et al., arXiv:2611.09382](https://arxiv.org/abs/2611.09382) and Takehi et al.

**Read the V3 notebooks first for new research.** The four V2 notebooks remain for reproducibility and historical comparisons. The original eight-frame plot does not show a coupled-model advantage; V3 is a new hypothesis, **not** a prevalidated improvement. All pretrained-model errors, invalid WSP proposals and negative outcomes must be retained in exported metrics.

## Original temporal toy: GitHub archive

- [Attention + CCNN temporal-extrapolation Colab](CCNN_Attention_Toy_Theory_Test_Colab.ipynb)
- [Standalone reproducible PyTorch runner](ccnn_attention_toy.py)

These are archival **spatial-diffusion toy experiments**, not a faithful evaluation of the original paper's time-lag convolution. Interpret their positive and negative observations only within those synthetic generators.
