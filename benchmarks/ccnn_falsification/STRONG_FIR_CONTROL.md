# Strong conditional FIR baseline — Issue #7

`QueryMLPFIR` predicts an unrestricted length-14 causal FIR from five observed
features (q, q², sin(πq), timestamp, timestamp²). Unlike an affine dynamic FIR,
a nonlinear MLP can represent abrupt conditional changes and multi-peaked
filters. The response passes through the same leaky-state transformation as
CCNN in V3 to isolate the **kernel parameterization** being compared.

`AttentionPlusMLPFIR` accepts a shared attention module and supports
`freeze_attention` to train the residual branch while holding attention
identical. Width/layer count are configurable for trainable-parameter and
compute matching.

**Status:** baseline **module and tests only**. Do not call this a matched
benchmark without adding it to the same data/training pipeline, tuning its
capacity fairly, and recording training curves, FLOPs, wall time and seed-level
paired results. No real-world or superiority claim yet.

Next: integrate in the single V3 benchmark and Colab, preserving exact code
synchronization as required by Issue #13. Do not make a fourth notebook.
