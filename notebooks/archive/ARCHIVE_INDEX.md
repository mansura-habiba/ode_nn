# Research notebook archive

**Archived:** 15 November 2026. The nine prior research notebooks and their supporting material are grouped here. Each archived notebook begins with a new Markdown cell documenting its purpose, research question and output; its original experiment cells remain unchanged. A matching Markdown note is supplied separately for discovery.

**Active file protected from changes:** [CCNN Lag Kernel Attention Repair Publication V2](../CCNN_Lag_Kernel_Attention_Repair_Publication_V2_Colab.ipynb).

| Archived notebook | RQ label | Research note |
|---|---|---|
| [Local CCNN/attention temporal extrapolation](<CCNN_Attention_Toy_Theory_Test_Colab.ipynb>) | RQ1 | [Purpose, question, outputs](<notes/CCNN_Attention_Toy_Theory_Test_Colab.md>) |
| [Music one-step coupled spectral prediction](<Music_Coupled_Neural_PDE_ODE_v2_Colab.ipynb>) | Music RQ1 | [Purpose, question, outputs](<notes/Music_Coupled_Neural_PDE_ODE_v2_Colab.md>) |
| [Music eight-step autoregressive rollout](<Music_Coupled_PDE_ODE_Complex_Task_v2_Colab.ipynb>) | Music RQ2 | [Purpose, question, outputs](<notes/Music_Coupled_PDE_ODE_Complex_Task_v2_Colab.md>) |
| [Event-conditioned persistent music dynamics](<Music_Event_Conditioned_Persistent_PDE_ODE_V3_Colab.ipynb>) | ECPD RQ1–RQ5 | [Purpose, question, outputs](<notes/Music_Event_Conditioned_Persistent_PDE_ODE_V3_Colab.md>) |
| [Weight-space programming mechanistic audit](<Music_PCWSR_HF_WSP_Mechanistic_Audit_V3_Colab.ipynb>) | Music WSP RQ4 | [Purpose, question, outputs](<notes/Music_PCWSR_HF_WSP_Mechanistic_Audit_V3_Colab.md>) |
| [Weight-space programming with three instruction LMs](<Music_PCWSR_HuggingFace_WSP_v2_Colab.ipynb>) | Music WSP RQ4 | [Purpose, question, outputs](<notes/Music_PCWSR_HuggingFace_WSP_v2_Colab.md>) |
| [Recording-disjoint music forecasting benchmark](<Music_Persistent_PDE_ODE_Recording_Benchmark_v2_Colab.ipynb>) | Music RQ3 | [Purpose, question, outputs](<notes/Music_Persistent_PDE_ODE_Recording_Benchmark_v2_Colab.md>) |
| [RQ4 spatial resolution transfer](<RQ4_CCNN_Spatial_Resolution_Transfer_Colab.ipynb>) | RQ4/H4 | [Purpose, question, outputs](<notes/RQ4_CCNN_Spatial_Resolution_Transfer_Colab.md>) |
| [RQ6/RQ8 single-weight repair and cross-grid transfer](<RQ6_RQ8_NonRegressive_Operator_Weight_Repair_Colab.ipynb>) | RQ6 | [Purpose, question, outputs](<notes/RQ6_RQ8_NonRegressive_Operator_Weight_Repair_Colab.md>) |

## Other archived files

- [Prior notebooks README](README.md) — historical index and research protocols.
- [Original research charter](RESEARCH_QUESTIONS_CCNN_ATTENTION_WEIGHT_REPAIR.md), including the clarification that the original CCNN uses causal time-lag convolution rather than spatial diffusion.
- [Standalone attention toy Python script](ccnn_attention_toy.py) and [standalone operator repair script](rq6_rq8_operator_weight_repair.py).
- [RQ4 preliminary three-seed evidence](rq4_results_preliminary/README.md), [raw results JSON](rq4_results_preliminary/RQ4_preliminary_3seed_metrics.json).
- [RQ6/RQ8 preliminary three-seed evidence](rq6_rq8_results_preliminary/README.md), [raw results JSON](rq6_rq8_results_preliminary/RQ6_RQ8_3seed_metrics.json).

**Evidence integrity:** These experiments are exploratory. Generated output filenames should not be confused with committed result files. Only the preliminary CPU studies above have separately stored metrics, and those do not establish unique CCNN superiority over established baselines.
