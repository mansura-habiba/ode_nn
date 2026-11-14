# Executable research notebooks

This directory is the canonical location for research notebooks in `ode_nn`. Updates to the music PDE–ODE and PCWSR studies should be committed here, preserving a reviewable change history.

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
