# RQ4 spatial resolution transfer

**Archived notebook:** [RQ4_CCNN_Spatial_Resolution_Transfer_Colab.ipynb](../RQ4_CCNN_Spatial_Resolution_Transfer_Colab.ipynb)  
**Original Git blob:** `7f0c5be3b6160d08c7cd5c58124282de1aa7ab07`  
**Archive status:** Historical/exploratory record; pre-existing experiment cells preserved.

## What this notebook works on
Tests a grid-aware operator trained at N=16 and evaluated unchanged on N=32/64/128 against unscaled, spectral, FNO-lite, pointwise ODE and attention controls on local/nonlocal generators.

## Research question
RQ4/H4: Does a spatially consistent coupled operator transfer robustly across resolutions/horizons and retain an advantage over strong resolution-neutral methods?

## Output and evidence status
Notebook outputs: `rq4_raw_metrics.csv`, `rq4_training_metrics.csv`, `rq4_summary.csv`, figures, `rq4_hypothesis_ledger.csv/.json`, `rq4_paired_uncertainty.csv` and `RQ4_CONCLUSION_AND_NEXT_RESEARCH.md`. Archived `rq4_results_preliminary/` contains three-seed CPU evidence: grid scaling helped; spectral/nonlocal controls challenge broad CCNN superiority.

The output inventory is not evidence that the notebook was executed. No notebook training or inference was run during archiving. The archived notebook also includes this explanation in its first Markdown cell.
