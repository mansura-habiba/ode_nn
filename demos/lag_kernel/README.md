# Continuous Convolution Research Lab — interactive mechanism demo

This Gradio demo illustrates the proposed **causal lag kernel** independently from the actual paper benchmark.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

For a temporary Colab-hosted public URL, set `GRADIO_SHARE=true` before launching. Do not use real private data when making the demo public.

## Four research views

- **Attention augmentation:** matched delayed chirps versus nonlocal negative-control generator.
- **Verified repair:** delay-shift fault, restricted edits, validation acceptance and held-out results.
- **Across-seed evidence:** reproducible toy repeats, per-seed and paired uncertainty summaries.
- **Protocol:** explicit limitations and prerequisites for a serious benchmark.

This is an **interactive scientific demonstration**, not a complete replication of the 2021 coupled PDE–ODE learner, a trained Transformer benchmark, or evidence that Hugging Face LLMs outperform numerical repair. The downloadable JSON includes generated settings and measured outputs. The [paper-style Colab notebook](../../notebooks/CCNN_Lag_Kernel_Attention_Repair_Publication_V2_Colab.ipynb) contains high-resolution mechanism and evaluation figures.

## Verification

Six core unit tests passed locally and the app imports in Gradio 6. Full production deployment and GPU/hardware stress tests have not been performed.
