# Local CCNN/attention temporal extrapolation

**Archived notebook:** [CCNN_Attention_Toy_Theory_Test_Colab.ipynb](../CCNN_Attention_Toy_Theory_Test_Colab.ipynb)  
**Original Git blob:** `54922e34381d59c7ae91c5bb26df7a9d9d0f3eaf`  
**Archive status:** Historical/exploratory record; pre-existing experiment cells preserved.

## What this notebook works on
Compares horizon-aware attention with four-step coupled local continuous-time dynamics, one-step Euler, dense Neural ODE, structural ablations and a nonlocal negative control on irregular synthetic trajectories.

## Research question
RQ1: Does CCNN improve unseen long-gap extrapolation? RQ2: Which structural mechanism contributes? RQ3: How does it behave when the local/diffusion assumption is violated?

## Output and evidence status
On execution: `local_coupled.json`, `nonlocal.json`, `raw_per_seed.csv`, `summary.csv`, prediction/compute figures and an optional output ZIP; no saved execution results.

The output inventory is not evidence that the notebook was executed. No notebook training or inference was run during archiving. The archived notebook also includes this explanation in its first Markdown cell.
