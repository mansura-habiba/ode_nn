# Recording-disjoint music forecasting benchmark

**Archived notebook:** [Music_Persistent_PDE_ODE_Recording_Benchmark_v2_Colab.ipynb](../Music_Persistent_PDE_ODE_Recording_Benchmark_v2_Colab.ipynb)  
**Original Git blob:** `fca3832d2e0dbb47a0797ef495ba53d6b988bfeb`  
**Archive status:** Historical/exploratory record; pre-existing experiment cells preserved.

## What this notebook works on
Compares persistent coupled spectral/global dynamics against neural ODE, temporal CNN, GRU, Transformer, simple predictors and three out-of-domain pretrained forecasting checkpoints.

## Research question
Music RQ3: Does persistent coupled evolution improve eight-step accuracy on unseen recordings after accounting for high-frequency/onset metrics and compute cost?

## Output and evidence status
On execution: `music_recording_level_raw_metrics.csv`, `music_recording_level_horizon_metrics.csv`, `music_solver_sensitivity.csv`, `music_recording_level_summary.csv`, comparison figures, `music_benchmark_protocol.json` and HF metrics; no saved execution results.

The output inventory is not evidence that the notebook was executed. No notebook training or inference was run during archiving. The archived notebook also includes this explanation in its first Markdown cell.
