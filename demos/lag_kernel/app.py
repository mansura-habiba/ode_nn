"""Interactive CCNN × attention × weight-repair *mechanism* demo.

Run with `python app.py` (and in a Colab runtime with Gradio share=True).
No network models are loaded; this is a transparent toy, not a Nature benchmark.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import attention_experiment, repair_experiment, as_json

SCENARIOS=["Matched delayed chirps", "Nonlocal negative control"]
SCOPES=["Condition-gated (observable shift)", "Global edit (shared across domains)"]


def make_file(content: str, stem: str) -> str:
    d=Path(tempfile.mkdtemp(prefix='ccnn_demo_'))
    p=d/f'{stem}.json'
    p.write_text(content,encoding='utf-8')
    return str(p)


def chart_evaluation(rows: list[dict], *, title: str, metric: str,
                     highlight: str) -> plt.Figure:
    fig, ax=plt.subplots(figsize=(10,4.2),layout='constrained')
    names=[r['model'] if 'model' in r else r['method'] for r in rows]
    values=[r[metric] for r in rows]
    bars=ax.barh(np.arange(len(names)),values)
    for label,bar in zip(names,bars):
        if highlight.lower() in label.lower():
            bar.set_hatch('///')
            bar.set_linewidth(2.0)
    ax.set_yticks(np.arange(len(names)),names,fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f'{metric.replace("_"," ")}  · lower is better')
    ax.set_title(title)
    ax.grid(axis='x',alpha=.22)
    return fig


def chart_lag_mechanism(report: dict) -> plt.Figure:
    sample=report['sample']
    tau=sample['lags']
    fig,axes=plt.subplots(2,2,figsize=(10,6.0),layout='constrained')
    axes[0,0].plot(tau,sample['x'])
    axes[0,0].set_title('Observed signal history')
    axes[0,0].set(xlabel='History lag τ',ylabel='Signal x(t−τ)')
    axes[0,1].plot(tau,sample['reference_kernel'],label='Hidden reference kernel')
    axes[0,1].plot(tau,sample['learned_kernel'],'--',label='Fitted structured kernel')
    axes[0,1].axvline(sample['query'],linestyle=':',label='Query delay')
    axes[0,1].set_title('Query-conditioned signed lag kernel')
    axes[0,1].legend(fontsize=8)
    axes[0,1].set(xlabel='History lag τ',ylabel='Weight K(q,τ)')
    cumulative=np.cumsum(sample['x']*sample['learned_kernel'])
    axes[1,0].plot(tau,cumulative)
    axes[1,0].set(title='CCNN lag accumulation u(τ)',xlabel='History lag τ',
                   ylabel='Cumulative Σ K(τ) x(t−τ)')
    axes[1,1].plot(report['preview']['reference'],label='Reference',linewidth=2)
    axes[1,1].plot(report['preview']['attention'],label='Lag-attention probe')
    axes[1,1].plot(report['preview']['hybrid'],label='Attention + CCNN')
    axes[1,1].set(title='Independent test examples (first 50)',xlabel='Test example',
                   ylabel='Prediction / target')
    axes[1,1].legend(fontsize=8)
    return fig


def run_attention(scenario: str, noise: float, seed: int, ntrain: int):
    report=attention_experiment(scenario=scenario,noise=noise,seed=int(seed),ntrain=int(ntrain))
    metric=pd.DataFrame(report['metrics']).sort_values('test_mse')
    fig=chart_evaluation(report['metrics'],title=f'Attention augmentation · {scenario}',
                         metric='test_mse',highlight='Attention + structured')
    mechanism=chart_lag_mechanism(report)
    best=metric.iloc[0]
    note=(f'**Lowest test MSE in this run:** {best["model"]} ({best["test_mse"]:.5f}).  '
          f'**Research hypotheses:** {report["hypotheses"]}.  '
          f'**Fit runtime:** {report["fit_seconds"]:.2f}s.  '
          'All methods use the same synthetic train/validation/test partitions. '
          'The attention probe is a learned query-window readout, **not a trained Transformer**. '
          'The structured kernel family intentionally resembles the matched generator. '
          'There is no claimed general superiority.')
    return mechanism,fig,metric,note,make_file(as_json(report),'attention_run')


def run_repair(scenario: str,shift: float,scope: str,seed: int):
    report=repair_experiment(scenario=scenario,shift=float(shift),scope=scope,seed=int(seed))
    df=pd.DataFrame(report['metrics'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    sweep=report['repair_sweep']
    axes[0].plot(sweep['offsets'],sweep['train_mse'],label='Shifted-domain repair TRAIN')
    axes[0].axvline(sweep['proposed_delta'],linestyle='--',label='Proposed (training only)')
    axes[0].axvline(sweep['applied_delta'],linestyle=':',label='Actually applied')
    axes[0].set(xlabel='Single kernel delay edit Δ',ylabel='Repair TRAIN MSE',title='One-scalar kernel repair')
    axes[0].legend(fontsize=8)
    bars=axes[1].barh(np.arange(len(df)),df['new_test_mse'])
    axes[1].set_yticks(np.arange(len(df)),df['method'],fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Shifted-domain held-out test MSE ↓',title='Repair comparison on unseen examples')
    bars[1].set_hatch('///');bars[1].set_linewidth(2)
    axes[1].grid(axis='x',alpha=.2)
    val=report['validation_gate']
    note=(f'**Structured edit:** {"accepted" if df.iloc[1]["accepted"] else "rejected"} '
          f'(Δ={sweep["applied_delta"]:+.3f}). '
          f'Validation shifted MSE: {val["shifted_frozen"]:.5f} → '
          f'{val["shifted_candidate"]:.5f}; original-domain allowed increase '
          f'≤ {val["allowed_original_increase"]:.5f}.  '
          f'**Interpretation:** {report["hypotheses"]}.  '
          + ('**Crucial:** original-domain non-regression is guaranteed by the observable-condition gate; '
             'do not interpret it as discovered robustness. ' if scope.startswith('Condition') else
             'Global edits are rejected if the original validation loss regresses too far. ')
          + 'A one-coordinate update is not functionally equivalent to a delay shift; '
            'the 64-weight adapter is provided as a stronger unstructured control. '
            'This is **deterministic numerical weight repair, not LLM/WSP**.')
    return fig,df,note,make_file(as_json(report),'repair_run')


def paired_bootstrap(x: np.ndarray, seed=119, repeats=1200) -> tuple[float,float]:
    rng=np.random.default_rng(seed)
    vals=x[rng.integers(0,len(x),size=(repeats,len(x)))].mean(-1)
    return tuple(np.quantile(vals,[.025,.975]).tolist())


def run_multiseed(seed_count: int, ntrain: int):
    all_rows=[]
    for scenario in SCENARIOS:
        for i in range(int(seed_count)):
            report=attention_experiment(scenario=scenario,seed=31+i*19,ntrain=int(ntrain))
            for r in report['metrics']:
                all_rows.append(dict(scenario=scenario,seed=31+i*19,
                                     model=r['model'],mse=r['test_mse']))
    raw=pd.DataFrame(all_rows)
    group=raw.groupby(['scenario','model'],sort=False).mse.agg(['mean','std']).reset_index()
    paired=[]
    for scenario in SCENARIOS:
        sub=raw[raw.scenario==scenario]
        att=sub[sub.model=='Lag attention probe (64 weights)'].sort_values('seed').mse.to_numpy()
        hybrid=sub[sub.model=='Attention + structured lag kernel (hybrid)'].sort_values('seed').mse.to_numpy()
        query=sub[sub.model=='Query-conditioned conv (320 weights)'].sort_values('seed').mse.to_numpy()
        for name,reference in [('Lag-attention probe',att),('Query-conditioned conv',query)]:
            diff=hybrid-reference
            lo,hi=paired_bootstrap(diff)
            paired.append(dict(scenario=scenario,comparison=f'Hybrid minus {name}',
                               mean_paired_mse_delta=diff.mean(),ci_low=lo,ci_high=hi,
                               interpretation='negative favors hybrid'))
    paired_df=pd.DataFrame(paired)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,scenario in zip(axes,SCENARIOS):
        sub=group[group.scenario==scenario]
        labels=[x.split(' (')[0] for x in sub.model]
        bars=ax.barh(np.arange(len(sub)),sub['mean'])
        for j,m in enumerate(sub.model):
            if 'hybrid' in m:bars[j].set_hatch('///');bars[j].set_linewidth(2)
        ax.set_yticks(np.arange(len(sub)),labels,fontsize=8)
        ax.invert_yaxis()
        ax.set(xlabel='Mean test MSE ↓',title=scenario)
        ax.grid(axis='x',alpha=.2)
    note=(f'**Exploratory evaluation:** {seed_count} paired synthetic seeds, '
          f'{ntrain} train examples each. The confidence intervals are descriptive bootstrap '
          'intervals and not evidence of real-world generalization. '
          'The negative-control panel is essential: a prior-matched model should not '
          'be expected to win when its mathematical assumptions fail. '
          'This suite omits a trained Transformer, SSM, Hyena, and full CCNN PDE–ODE integration.')
    report=dict(raw=raw.to_dict('records'),aggregate=group.to_dict('records'),
                paired=paired_df.to_dict('records'),status='exploratory_mechanism_demo')
    return fig,group,paired_df,note,make_file(json.dumps(report,indent=2),'multiseed_report')

CSS="""
.gradio-container {max-width: 1220px !important; margin:auto}
#research-title {text-align:center; padding:24px 0 4px}
#research-title h1 {font-weight:760; letter-spacing:-.025em;}
"""

with gr.Blocks(title='CCNN · Attention · Verifiable Repair — Research Demo',css=CSS) as demo:
    gr.Markdown("""
# Continuous Convolution Research Lab
### Can lag-structured CCNN improve attention and make neural weight repairs more precise?

**Interactive mechanistic demonstration** · synthetic causal history · transparent numerical calculations · downloadable evidence

This is the *original causal-lag idea* of CCNN in simplified discrete form:
\(u(t,\tau)=\int_0^\tau K(s,t)x(t-s)\,ds\). The plots display its accumulated lag integral.
It is **not** a complete implementation of the 2021 PDE–ODE learner and **not** a peer-reviewed benchmark.
    """,elem_id='research-title')
    gr.Markdown("""**How to use this demo:** run the matched delayed-chirp task, then switch to the **nonlocal negative control**. Next, inject a delay fault and observe how a one-scalar kernel edit compares with unstructured edits. Finally run several seeds and inspect the uncertainty intervals. No scores are prefilled.""")
    with gr.Tabs():
        with gr.TabItem('01 · Attention augmentation'):
            gr.Markdown('#### Mechanism probe: does a structured causal-lag kernel help a query-window attention readout?')
            with gr.Row():
                with gr.Column(scale=1):
                    a_scenario=gr.Radio(SCENARIOS,value=SCENARIOS[0],label='True generator / negative control')
                    a_noise=gr.Slider(0.0,.15,value=.025,step=.005,label='Observation / label noise')
                    a_seed=gr.Number(value=17,precision=0,label='Random seed')
                    a_train=gr.Slider(160,800,value=450,step=40,label='Training examples')
                    a_button=gr.Button('Run paired model comparison',variant='primary')
                    gr.Markdown('**Controls:** Static convolution, learned lag-window attention proxy, 320-weight query-conditioned convolution, structured kernel, and their hybrid.')
                with gr.Column(scale=2):
                    a_text=gr.Markdown('Choose settings, then run the comparison.')
                    a_table=gr.Dataframe(label='Measured held-out metrics')
                    a_file=gr.File(label='Evidence JSON (includes splits and hypothesis outcomes)')
            a_fig=gr.Plot(label='Model comparison (proposed hybrid hatched)')
            a_mechanism=gr.Plot(label='History, causal kernel and cumulative lag PDE state')
            a_button.click(run_attention,[a_scenario,a_noise,a_seed,a_train],
                           [a_mechanism,a_fig,a_table,a_text,a_file])
        with gr.TabItem('02 · Verified kernel repair'):
            gr.Markdown('#### Fault injection: can one kernel-delay edit repair a shifted history response?')
            with gr.Row():
                with gr.Column(scale=1):
                    r_scenario=gr.Radio(SCENARIOS,value=SCENARIOS[0],label='True generator / negative control')
                    r_shift=gr.Slider(0,.2,value=.11,step=.01,label='Unknown physical delay shift')
                    r_scope=gr.Radio(SCOPES,value=SCOPES[0],label='Where is edit applied?')
                    r_seed=gr.Number(value=17,precision=0,label='Random seed')
                    r_button=gr.Button('Diagnose → edit → verify',variant='primary')
                    gr.Markdown('**Validation gate:** require at least 1% shifted-domain validation improvement and ≤ max(0.002, 10% baseline MSE) original-domain regression.')
                with gr.Column(scale=2):
                    r_text=gr.Markdown('Run the repair laboratory to see candidate and verification outcomes.')
                    r_table=gr.Dataframe(label='Held-out original and shifted-domain errors')
                    r_file=gr.File(label='Repair proposals and gate decisions')
            r_fig=gr.Plot(label='Edit loss curve and comparative held-out repair')
            r_button.click(run_repair,[r_scenario,r_shift,r_scope,r_seed],
                           [r_fig,r_table,r_text,r_file])
        with gr.TabItem('03 · Evidence across seeds'):
            gr.Markdown('#### Mechanism stress test: matched versus mismatched generators')
            with gr.Row():
                many_seeds=gr.Slider(3,15,value=5,step=1,label='Paired random seeds')
                many_train=gr.Slider(160,800,value=400,step=40,label='Training examples per seed')
                many_button=gr.Button('Run reproducible toy suite',variant='primary')
            many_text=gr.Markdown('Reports pooled mean MSE and paired bootstrap intervals. Does not establish external validity.')
            many_plot=gr.Plot(label='Model ranking by generator')
            many_table=gr.Dataframe(label='Across-seed mean and standard deviation')
            many_pairs=gr.Dataframe(label='Paired differences (hybrid minus comparator)')
            many_file=gr.File(label='Full per-seed data and intervals JSON')
            many_button.click(run_multiseed,[many_seeds,many_train],
                              [many_plot,many_table,many_pairs,many_text,many_file])
        with gr.TabItem('04 · Scientific protocol'):
            gr.Markdown(r'''
### What the interactive demo can establish

It demonstrates the exact **causal lag accumulation** \(u(\tau)=\int_0^\tau K(s,t)x(t-s)ds\) on discretized history and allows independent users to manipulate the generator, inspect the signed kernel, alter the delay and rerun validation. A restricted, physically interpretable edit can be compared with one-coefficient and 64-parameter unstructured edits. All plots are recomputed from deterministic synthetic examples.

### What the demo **cannot** establish

1. **Not a complete original CCNN replication.** The app computes the lag integral and fits a compact analytic kernel family by grid search; it does not solve the full 2021 coupled PDE–ODE training algorithm.
2. **Not a Transformer benchmark.** The attention baseline is a learned, query-centred lag-window readout, not a trained multi-head Transformer. The strong query-conditioned unstructured convolution is separately included.
3. **Not an LLM weight programmer.** The edit comes from transparent numerical search. Three independent open-weight HF instruction models, diagnostic permutation controls, and numerical comparisons belong in the confirmatory WSP study.
4. **Not a fair equal-capacity benchmark.** Structured kernel templates encode the same family as the positive data generator. The negative control is intentionally outside this family. Report both.
5. **The condition-gated repair** preserves original-domain predictions *by design*. Switch to **Global edit** to enforce a meaningful non-regression gate without that shortcut.

### Confirmatory benchmark before a journal submission

- Train a **real Transformer**, Transformer+standard CNN, Transformer+full lag-PDE/ODE CCNN, Hyena/SSM, and a query-conditioned unstructured model with matched training/tuning budgets.
- Use multi-family generators, delayed noisy chirps, nonlocal controls, irregular observations, changing kernel families, and public datasets with protocol-fixed splits.
- Verify calibration, delay extrapolation, accuracy–latency/parameter Pareto curves, statistical uncertainty, and failure modes across ≥10 seeds.
- Freeze the backbone and inject independent parameter faults. Report repair success, original-domain non-regression, candidate counts, wall-clock cost, instability, sparse Adam, Gauss–Newton, low-rank adapters, and 3 HF weight programmers.
- Publish raw per-seed predictions, code commits, prompts, and unsuccessful trials. **The app is a communication and debugging layer; benchmark data must be produced separately.**

**Sources:** [Habiba & Pearlmutter (2021)](https://arxiv.org/abs/2111.00343) · [Weight-Space Programming (2026)](https://rikiyat.github.io/blog/weight-space-programming/) · [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-overview) · [Hugging Face leaderboards](https://huggingface.co/docs/leaderboards/index) · [Neuronpedia](https://docs.neuronpedia.org/steering)
''')

if __name__=='__main__':
    # Colab users can set GRADIO_SHARE=true for an ephemeral public tunnel.
    demo.launch(share=os.getenv('GRADIO_SHARE','false').lower()=='true')