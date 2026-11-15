# RQ6/RQ8 single-weight repair and cross-grid transfer

**Archived notebook:** [RQ6_RQ8_NonRegressive_Operator_Weight_Repair_Colab.ipynb](../RQ6_RQ8_NonRegressive_Operator_Weight_Repair_Colab.ipynb)  
**Original Git blob:** `248988850bd2a804f61908d906ba589fce9927ca`  
**Archive status:** Historical/exploratory record; pre-existing experiment cells preserved.

## What this notebook works on
Tests validation-gated one-weight repairs in a synthetic Fourier/coupled global operator under conditional shift, against random, Adam, full fine-tuning, no-defect, advection and direct-map controls.

## Research question
RQ6: Can a small edit correct shifted-domain error without original-domain regression? RQ8: Will the same edit transfer unchanged from 16 to 32/64/128? RQ7 (LLM proposals) is untested.

## Output and evidence status
Notebook outputs: `rq6_rq8_weight_repair_results.json`, `rq6_rq8_repair_comparison.png`, `rq6_rq8_resolution_transfer.png`, `rq6_rq8_hypothesis_ledger.csv`. Archived `rq6_rq8_results_preliminary/` has three-seed CPU evidence of favorable single-weight repair, but no unique coupled-dynamics advantage.

The output inventory is not evidence that the notebook was executed. No notebook training or inference was run during archiving. The archived notebook also includes this explanation in its first Markdown cell.
