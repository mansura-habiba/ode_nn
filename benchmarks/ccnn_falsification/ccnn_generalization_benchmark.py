"""RQ1-M and RQ6-M reproducible toy: time-varying causal lag kernel and repair.
No external datasets, GPU, authentication, or downloads required.
"""
from __future__ import annotations
import os, json, time, copy, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N=48; L=14; W=32; SEEDS=[11,23]; torch.set_num_threads(2)

def make_data(count:int, seed:int, shift=0.0, regime='standard'):
    r=np.random.default_rng(seed)
    t=np.linspace(0,1,N,dtype=np.float32)[None,:]
    q0=r.uniform(-.7,.7,(count,1)).astype(np.float32)
    slope=r.uniform(-.25,.25,(count,1)).astype(np.float32) if regime=='standard' else r.uniform(-.85,.85,(count,1)).astype(np.float32)
    q=np.clip(q0+slope*(t-.5)+.23*np.sin(2*np.pi*(t+r.uniform(0,1,(count,1)))), -1.5,1.5).astype(np.float32)
    x=r.normal(0,.4,(count,N)).astype(np.float32)
    # randomized chirp-like motifs with unrelated distractors
    for b in range(count):
        for _ in range(3):
            loc=int(r.integers(0,N))
            idx=np.arange(N)-loc
            envelope=np.exp(-.5*(idx/float(r.uniform(3,7)))**2)
            phase=2*np.pi*(r.uniform(.12,.28)*idx + r.uniform(.001,.009)*idx**2)
            x[b]+=float(r.choice([-1,1])*r.uniform(.3,1.2))*envelope*np.cos(phase)
    return torch.tensor(x),torch.tensor(q)

def lag_history(x):
    # H[b,t,j]=x[b,t-j] and zero for t<j (causal prehistory)
    return torch.stack([F.pad(x[:,:N-j],(j,0)) for j in range(L)],dim=-1)

def kernel(q,mu0,muq,logsigma,freq,chirp,domain_shift=None):
    tau=torch.arange(L,device=q.device,dtype=q.dtype).view(1,1,L)/(L-1)
    mu=mu0+muq*q.unsqueeze(-1)
    if domain_shift is not None:mu=mu+domain_shift
    width=F.softplus(torch.as_tensor(logsigma,device=q.device,dtype=q.dtype))+.04
    envelope=torch.exp(-.5*((tau-mu)/width)**2)
    phase=2*torch.pi*(freq*tau+chirp*tau.square())
    raw=envelope*torch.cos(phase)
    return raw/(raw.square().sum(-1,keepdim=True).sqrt()+1e-7)

def integrate_drive(m):
    # z' = -8z + 8 tanh(2m); exponential time stepping over dt=1/N.
    alpha=1.-math.exp(-8./N)
    z=torch.zeros_like(m[:,0])
    for j in range(m.shape[1]):z=(1-alpha)*z+alpha*torch.tanh(2.*m[:,j])
    return z

TRUE=dict(mu0=.43,muq=.145,logsigma=-2.35,freq=2.15,chirp=.62)

def target(x,q,shift=0.0):
    with torch.no_grad():
        k=kernel(q,**TRUE,domain_shift=shift)
        u=torch.cumsum(k*lag_history(x),dim=-1) # lag-PDE quadrature state
        return integrate_drive(u[:,:,-1])

class Attention(nn.Module):
    def __init__(self,width=W):
        super().__init__()
        self.embed=nn.Linear(4,width)
        self.att=nn.MultiheadAttention(width,4,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,24),nn.Tanh(),nn.Linear(24,1))
    def forward(self,x,q):
        p=torch.arange(N,device=x.device,dtype=x.dtype)[None,:]/N
        a=self.embed(torch.stack([x,q,p.expand_as(x),p.expand_as(x).square()],-1))
        attention=self.att(a,a,a,need_weights=False)[0]
        return self.head((a+attention).mean(1)).squeeze(-1)

class LagField(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu0=nn.Parameter(torch.tensor(.32))
        self.muq=nn.Parameter(torch.tensor(.1))
        self.logsigma=nn.Parameter(torch.tensor(-2.2))
        self.freq=nn.Parameter(torch.tensor(2.0))
        self.chirp=nn.Parameter(torch.tensor(.5))
        self.gain=nn.Parameter(torch.tensor(.9))
    def response(self,x,q,delta=None):
        k=kernel(q,self.mu0,self.muq,self.logsigma,self.freq,self.chirp,domain_shift=delta)
        lag_pde=torch.cumsum(k*lag_history(x),-1)
        return self.gain*integrate_drive(lag_pde[:,:,-1])

class StaticFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(torch.randn(L)*.1)
        self.gain=nn.Parameter(torch.tensor(1.))
    def response(self,x,q):
        weights=self.weights/(self.weights.square().sum().sqrt()+1e-7)
        m=(lag_history(x)*weights).sum(-1)
        return self.gain*integrate_drive(m)

class DynamicFilter(nn.Module):
    """Query-conditioned arbitrary discrete lag kernel: 14 weights change with q."""
    def __init__(self):
        super().__init__()
        self.base=nn.Parameter(torch.randn(L)*.1)
        self.by_query=nn.Parameter(torch.randn(L)*.05)
        self.gain=nn.Parameter(torch.tensor(.9))
    def response(self,x,q,delta=None):
        weights=self.base[None,None,:]+q[:,:,None]*self.by_query[None,None,:]
        if delta is not None:
            # A chosen unstructured coordinate; independent scalar edit budget.
            weights=weights+delta*F.one_hot(torch.tensor(4,device=x.device),L).float()
        weights=weights/(weights.square().sum(-1,keepdim=True).sqrt()+1e-7)
        m=(lag_history(x)*weights).sum(-1)
        return self.gain*integrate_drive(m)

class ExpSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_decays=nn.Parameter(torch.tensor([-3.,-1.,1.,3.]))
        self.mix=nn.Parameter(torch.randn(4)*.05)
        self.qmix=nn.Parameter(torch.randn(4)*.05)
        self.gain=nn.Parameter(torch.tensor(.9))
    def response(self,x,q):
        decays=F.softplus(self.raw_decays)+.02
        state=torch.zeros((x.shape[0],4),device=x.device)
        out=[]
        for j in range(N):
            state=state*torch.exp(-decays)[None,:]+x[:,j:j+1]
            out.append((state*(self.mix+q[:,j:j+1]*self.qmix)).sum(-1))
        return self.gain*integrate_drive(torch.stack(out,1))

class Model(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.att=Attention()
        if kind=='lag':self.branch=LagField()
        if kind=='static':self.branch=StaticFilter()
        if kind=='ssm':self.branch=ExpSSM()
        if kind=='dynamic':self.branch=DynamicFilter()
        if kind!='attention':self.gate=nn.Parameter(torch.tensor(.95))
    def forward(self,x,q,delta=None):
        a=self.att(x,q)
        if self.kind=='attention':return a
        r=self.branch.response(x,q,delta) if self.kind in ('lag','dynamic') else self.branch.response(x,q)
        return a+self.gate*r

def fit(kind,seed,epochs=38,train_n=550):
    torch.manual_seed(seed)
    model=Model(kind)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    x,q=make_data(train_n,seed+100);y=target(x,q)
    xv,qv=make_data(150,seed+200);yv=target(xv,qv)
    best=float('inf');state=None; start=time.perf_counter()
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(x)).split(96):
            opt.zero_grad();out=model(x[idx],q[idx]);loss=(out-y[idx]).square().mean();loss.backward();opt.step()
        with torch.no_grad():val=((model(xv,qv)-yv)**2).mean().item()
        if val<best:best=val;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    xt,qt=make_data(240,seed+300);yt=target(xt,qt)
    with torch.no_grad():test=((model(xt,qt)-yt)**2).mean().item()
    # query-shift tests: change shape of time-varying q but not physics
    xs,qs=make_data(240,seed+400,regime='shifted');ys=target(xs,qs)
    with torch.no_grad():ood=((model(xs,qs)-ys)**2).mean().item()
    return model,dict(kind=kind,seed=seed,validation_mse=best,test_mse=test,ood_mse=ood,train_seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()))

def repair_experiment(model,seed,shift=.12):
    # Base trained on normal kernel. New physics has kernel center delayed by shift.
    # New-domain condition is known to the repair operator; all original parameters frozen.
    assert model.kind in ('lag','dynamic')
    for p in model.parameters():p.requires_grad_(False)
    xn,qn=make_data(180,seed+550);yn=target(xn,qn,shift=shift)
    xv,qv=make_data(160,seed+650);yv=target(xv,qv,shift=shift)
    xo,qo=make_data(160,seed+750);yo=target(xo,qo)
    xt,qt=make_data(200,seed+850);yt=target(xt,qt,shift=shift)
    def loss(x,q,y,d):
        return ((model(x,q,delta=d)-y)**2).mean()
    zero=torch.tensor(0.)
    with torch.no_grad():base_shift=float(loss(xt,qt,yt,zero));old_before=float(loss(xo,qo,yo,zero))
    # Numerical finite differences and damped Gauss–Newton single step; no test data.
    # For a correct scalar Jacobian use torch.func.jvp or autograd.functional.jacobian (output vector wrt scalar).
    eps=torch.tensor(.001)
    with torch.no_grad():
        J=((model(xn,qn,delta=eps)-model(xn,qn,delta=-eps))/(2*eps)).detach()
    resid=(yn-model(xn,qn,delta=zero)).detach()
    direct=float((J*resid).sum()/(J.square().sum()+.04))
    variants={}
    for name,candidates in [('jacobian',[direct*t for t in (.25,.5,1.,1.5)]),('random',list(np.random.default_rng(seed+1900).uniform(-.3,.3,12)))]:
        scored=[]
        for val in candidates:
            val=float(np.clip(val,-.3,.3))
            with torch.no_grad():v=float(loss(xv,qv,yv,torch.tensor(val)))
            scored.append((v,val))
        score,chosen=min(scored)
        variants[name]=(score,chosen)
    edit=torch.nn.Parameter(torch.tensor(0.))
    opt=torch.optim.Adam([edit],lr=.035)
    best_val=float('inf');best_edit=0.
    for step in range(70):
        opt.zero_grad()
        err=loss(xn,qn,yn,edit)
        err.backward();opt.step()
        with torch.no_grad():edit.clamp_(-.3,.3)
        if (step+1)%5==0:
            with torch.no_grad():v=float(loss(xv,qv,yv,edit))
            if v<best_val:best_val=v;best_edit=float(edit.detach())
    variants['adam']=(best_val,best_edit)
    with torch.no_grad():base_val=float(loss(xv,qv,yv,zero))
    results=[]
    for name,(val,chosen) in variants.items():
        accepted=val<.995*base_val
        delta=torch.tensor(chosen if accepted else 0.)
        with torch.no_grad():shift_mse=float(loss(xt,qt,yt,delta));orig_mse=float(loss(xo,qo,yo,zero))
        results.append(dict(kind=name,seed=seed,accepted=accepted,delta=float(delta),val_mse=val,test_mse=shift_mse,original_mse=orig_mse))
    return dict(model_kind=model.kind,seed=seed,true_shift=shift,frozen_shift_test_mse=base_shift,original_test_mse=old_before,repair_results=results,train_diagnostics={'JtJ':float(J.square().sum()),'Jtr':float((J*resid).sum()),'current_mu0':float(model.branch.mu0.detach()) if model.kind=='lag' else None,'current_muq':float(model.branch.muq.detach()) if model.kind=='lag' else None})

def main():
    os.makedirs('lag_results',exist_ok=True)
    epochs=int(os.getenv('EPOCHS','30'))
    train_n=int(os.getenv('TRAIN_N','480'))
    seeds=SEEDS if os.getenv('QUICK','0')!='1' else [11]
    summary=[]; repairs=[]
    for seed in seeds:
        for kind in ['attention','static','dynamic','ssm','lag']:
            model,metric=fit(kind,seed,epochs=epochs,train_n=train_n)
            summary.append(metric)
            print('Architecture',metric,flush=True)
            if kind in ('lag','dynamic'):
                repair=repair_experiment(model,seed)
                repairs.append(repair)
                print('Repair',repair,flush=True)
                torch.save(model.state_dict(),f'lag_results/{kind}_model_seed_{seed}.pt')
    result=dict(architecture=summary,repair=repairs,conditions={
        'dataset':'synthetic chirp-like 48-point stream with time-varying causal kernel target',
        'teacher':TRUE,'memory_lags':L,'samples':N,'training_epochs':epochs,
        'train_examples':train_n,'seeds':seeds,'limitations':'Generator deliberately shares the lag-kernel structure; this is a matched-bias feasibility test. HF proposals not run.'})
    with open('lag_results/results.json','w') as f:json.dump(result,f,indent=2)
    return result
# Execution is invoked by a later notebook cell

def diagnostic_negative_controls(seed):
    """Posthoc controls. Uses fitted lag model, never adjusts source-domain weights."""
    model=Model('lag')
    try:state=torch.load(f'lag_results/lag_model_seed_{seed}.pt',map_location='cpu',weights_only=True)
    except TypeError:state=torch.load(f'lag_results/lag_model_seed_{seed}.pt',map_location='cpu')
    model.load_state_dict(state);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    x,q=make_data(160,seed+2700)
    no_fault=target(x,q,shift=0.)
    def loss(y,edit):
        with torch.no_grad():return float(((model(x,q,delta=torch.tensor(float(edit)))-y)**2).mean())
    candidates=np.linspace(-.25,.25,31)
    base=loss(no_fault,0.)
    best=min((loss(no_fault,d),float(d)) for d in candidates)
    # A frequency shift cannot generally be repaired by translating the kernel center.
    with torch.no_grad():
        freq_shift=.6
        kwargs={**TRUE,'freq':TRUE['freq']+freq_shift}
        altered=integrate_drive(torch.cumsum(kernel(q,**kwargs)*lag_history(x),-1)[:,:,-1])
    baseline_freq=loss(altered,0.)
    best_freq=min((loss(altered,d),float(d)) for d in candidates)
    return {'seed':seed,'no_fault_baseline_mse':base,
            'no_fault_best_mse':best[0],
            'no_fault_best_edit':best[1],
            'no_fault_false_accept':best[0]<.995*base,
            'frequency_fault_baseline_mse':baseline_freq,
            'frequency_fault_best_delay_only_mse':best_freq[0],
            'frequency_fault_best_delay_edit':best_freq[1],
            'frequency_fault_true_extra_frequency':freq_shift}


# --- V3: unmatched mechanism and ungated matched-budget repair ---
# This benchmark imports the original V2 model definitions for auditability.
import argparse
import csv
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FAMILIES=('matched_chirp','bimodal','boxcar','damped_oscillator','switching','feedback_ode')
CURRENT_FAMILY='matched_chirp'
SHIFT=.12

def teacher_k(q, family, shift=0.):
    tau=torch.arange(L,device=q.device,dtype=q.dtype).view(1,1,L)/(L-1)
    qq=q.unsqueeze(-1)
    if family=='matched_chirp':
        return kernel(q,**{**TRUE,'mu0':TRUE['mu0']+shift})
    if family=='bimodal':
        a=torch.exp(-.5*((tau-(.22+.08*qq+shift))/.105)**2)
        b=torch.exp(-.5*((tau-(.74-.07*qq-shift))/.09)**2)
        raw=a-.85*b
    elif family=='boxcar':
        s=lambda x:torch.sigmoid(85*x)
        raw=s(tau-.13-.06*qq-shift)-s(tau-.38-.06*qq-shift)-.7*(s(tau-.67+.05*qq)-s(tau-.91+.05*qq))
    elif family=='damped_oscillator':
        raw=torch.exp(-3.2*tau)*torch.sin(2*torch.pi*(2.4+.5*qq+2*shift)*tau)
    elif family in ('switching','feedback_ode'):
        left=torch.exp(-.5*((tau-.17)/.08)**2)
        right=torch.exp(-.5*((tau-.78)/.11)**2)
        mix=torch.sigmoid(12*(qq-shift))
        raw=(1-mix)*left-mix*right
    else:raise ValueError(family)
    return raw/(raw.square().sum(-1,keepdim=True).sqrt()+1e-7)

def target(x,q,shift=0.):
    """Overrides original synthetic target; fit() now uses CURRENT_FAMILY."""
    with torch.no_grad():
        m=(teacher_k(q,CURRENT_FAMILY,shift)*lag_history(x)).sum(-1)
        if CURRENT_FAMILY!='feedback_ode':return integrate_drive(m)
        alpha=1.-math.exp(-8./N)
        z=torch.zeros_like(m[:,0])
        for t in range(N):
            z=(1-alpha)*z+alpha*torch.tanh(2*m[:,t]+1.8*z*x[:,t])
        return z

def qbasis(q):
    return torch.stack((torch.ones_like(q),q,q.square(),torch.sin(torch.pi*q)),-1)

def dct_masks():
    jj=torch.arange(L,dtype=torch.float32)
    basis=F.normalize(torch.stack([torch.cos(torch.pi*(jj+.5)*i/L) for i in range(L)],-1),dim=0)
    out=[]
    for i in range(L):
        for j in range(4):
            a=torch.zeros(L,4)
            a[:,j]=basis[:,i]
            out.append(a)
    return torch.stack(out)

class UngatedAdapter(nn.Module):
    """Trainable edits on ONE frozen CCNN: equal explicit scalar budgets."""
    def __init__(self,frozen,method,budget):
        super().__init__()
        self.frozen=frozen;self.method=method;self.budget=budget
        allowed={'structured':(1,5,18,36,56),
                 'tangent':(1,5),
                 'spectral':(1,5,18,36,56),
                 'rank1':(18,), 'rank2':(36,), 'full':(56,)}
        assert budget in allowed[method]
        self.edit=nn.Parameter(torch.zeros(budget))
        self.register_buffer('dct',dct_masks())
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==budget

    def corrected_kernel(self,q):
        b=self.frozen.branch;e=self.edit
        if self.method=='structured':
            k=kernel(q,b.mu0+e[0],b.muq+(e[1] if len(e)>1 else 0),
                     b.logsigma+(e[2] if len(e)>2 else 0),
                     b.freq+(e[3] if len(e)>3 else 0),
                     b.chirp+(e[4] if len(e)>4 else 0))
            if len(e)<=5:return k
            # >5 is a hybrid with unstructured residual coefficients.
            dw=torch.einsum('n,nlf->lf',e[5:],self.dct[:len(e)-5])
        else:
            k=kernel(q,b.mu0,b.muq,b.logsigma,b.freq,b.chirp)
            if self.method=='tangent':
                # Same local CCNN function-space tangent directions, not arbitrary FIR taps.
                pars=(b.mu0,b.muq,b.logsigma,b.freq,b.chirp)
                eps=1e-3
                d_k=torch.zeros_like(k)
                for i in range(len(e)):
                    plus=list(pars);minus=list(pars)
                    plus[i]=pars[i]+eps;minus[i]=pars[i]-eps
                    direction=(kernel(q,*plus)-kernel(q,*minus))/(2*eps)
                    d_k=d_k+e[i]*direction
                raw=k+d_k
                return raw/(raw.square().sum(-1,keepdim=True).sqrt()+1e-7)
            if self.method=='spectral':
                dw=torch.einsum('n,nlf->lf',e,self.dct[:len(e)])
            elif self.method=='full':
                dw=e.reshape(L,4)
            elif self.method.startswith('rank'):
                rank=1 if self.method=='rank1' else 2
                U=e[:L*rank].reshape(L,rank)
                V=e[L*rank:].reshape(rank,4)+torch.eye(rank,4,device=q.device,dtype=q.dtype)
                dw=U@V
            else:raise ValueError(self.method)
        raw=k+torch.einsum('btf,lf->btl',qbasis(q),dw)
        return raw/(raw.square().sum(-1,keepdim=True).sqrt()+1e-7)

def ungated_predict(base,x,q,adapter=None):
    """NO domain flag: all input populations observe identical repaired weights."""
    a=base.att(x,q)
    k=(kernel(q,base.branch.mu0,base.branch.muq,base.branch.logsigma,
              base.branch.freq,base.branch.chirp) if adapter is None
       else adapter.corrected_kernel(q))
    drive=(k*lag_history(x)).sum(-1)
    return a+base.gate*base.branch.gain*integrate_drive(drive)

def gatefree_repair(base,family,seed,method,budget,steps=85,shift=.12):
    global CURRENT_FAMILY
    CURRENT_FAMILY=family
    # Pair domain inputs, allowing source and shifted targets to conflict.
    # Train/validation/test splits are still statistically independent.
    x1,q1=make_data(128,seed+311); y1=target(x1,q1)
    x2,q2=x1,q1; y2=target(x2,q2,shift=shift)
    vx1,vq1=make_data(160,seed+511); vy1=target(vx1,vq1)
    vx2,vq2=vx1,vq1; vy2=target(vx2,vq2,shift=shift)
    tx1,tq1=make_data(240,seed+711); ty1=target(tx1,tq1)
    tx2,tq2=tx1,tq1; ty2=target(tx2,tq2,shift=shift)
    error=lambda x,q,y,adapter:((ungated_predict(base,x,q,adapter)-y)**2).mean()
    with torch.no_grad():
        source_before=float(error(tx1,tq1,ty1,None))
        shifted_before=float(error(tx2,tq2,ty2,None))
        val_source_before=float(error(vx1,vq1,vy1,None))
        val_shift_before=float(error(vx2,vq2,vy2,None))
    # Absolute + relative preservation tolerance predeclared before training.
    ceiling=val_source_before*1.10+.0001
    adapter=UngatedAdapter(base,method,budget)
    opt=torch.optim.Adam(adapter.parameters(),lr=.018)
    best_shift=val_shift_before;best_edit=None;start=time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        shifted=error(x2,q2,y2,adapter)
        original=error(x1,q1,y1,adapter)
        objective=shifted+.8*original+12*F.relu(original-ceiling)
        objective.backward();opt.step()
        with torch.no_grad():adapter.edit.clamp_(-.8,.8)
        if (step+1)%5==0 or step==steps-1:
            with torch.no_grad():
                vs=float(error(vx1,vq1,vy1,adapter))
                vt=float(error(vx2,vq2,vy2,adapter))
            if vs<=ceiling and vt<best_shift*(1-1e-7):
                best_shift=vt;best_edit=adapter.edit.detach().clone()
    with torch.no_grad():
        if best_edit is None:adapter.edit.zero_()
        else:adapter.edit.copy_(best_edit)
        source_after=float(error(tx1,tq1,ty1,adapter))
        shifted_after=float(error(tx2,tq2,ty2,adapter))
        val_source=float(error(vx1,vq1,vy1,adapter))
        val_shift=float(error(vx2,vq2,vy2,adapter))
    return dict(family=family,seed=seed,shift=shift,method=method,budget=budget,
                accepted=best_edit is not None,source_before=source_before,
                source_after=source_after,shifted_before=shifted_before,
                shifted_after=shifted_after,
                val_source=val_source,val_shift=val_shift,preservation_ceiling=ceiling,
                recovery=1-shifted_after/(shifted_before+1e-12),
                source_drift=source_after-source_before,
                independent_source_pass=source_after<=source_before*1.10+.0001,
                repair_seconds=time.perf_counter()-start)

def paper_diagram(out):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    def box(ax,x,y,s,blue=False):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x,y),2.35,.9,boxstyle='round,pad=.08',
                 edgecolor='.2',facecolor=('#dbeafe' if blue else '.97')))
        ax.text(x+1.175,y+.45,s,ha='center',va='center',fontsize=8)
    def arrow(ax,x1,y1,x2,y2):
        from matplotlib.patches import FancyArrowPatch
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10))
    for a in axs:a.set(xlim=(0,10),ylim=(0,6));a.axis('off')
    a=axs[0]
    box(a,.2,2.5,'History x(t−τ)\nand condition q(t)')
    box(a,3.6,2.5,'Lag PDE state\nu(t,τ), ∂τu=K·x',True)
    box(a,7,2.5,'ODE latent state\nz′=f(z,u)',True)
    box(a,3.6,4.45,'Attention branch\nshared readout')
    arrow(a,2.7,3,3.5,3);arrow(a,6.08,3,6.9,3)
    arrow(a,2.7,3.5,3.5,4.9)
    a.set_title('(A) Coupled continuous formulation')
    a=axs[1]
    for i,(y,s) in enumerate([(4.45,'Gaussian chirp\nmatched'),(2.65,'Bimodal / boxcar\nunmatched'),(.85,'Switching / feedback\nunmatched dynamics')]):
        box(a,.6,y,s,blue=(i==0))
        box(a,6,y,'Independent teacher\nand held-out target')
        arrow(a,3.1,y+.45,5.9,y+.45)
    a.set_title('(B) Kernel and dynamics stress tests')
    a=axs[2]
    box(a,.2,2.6,'ONE frozen CCNN\nsource model')
    box(a,3.5,2.6,'Structured / low-rank\nspectral / full edits',True)
    box(a,7,4.3,'Shifted validation\nrecovery')
    box(a,7,1.1,'Original validation\npreservation')
    arrow(a,2.65,3.05,3.4,3.05);arrow(a,5.95,3.1,6.9,4.7)
    arrow(a,5.95,2.95,6.9,1.5)
    a.set_title('(C) Gate-free shared-model repair')
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'Figure_01_Coupled_PDE_ODE_Architecture.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def paper_results(records,out):
    import matplotlib.pyplot as plt
    if not records:return
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for fam in FAMILIES:
        rs=[r for r in records if r['family']==fam]
        if rs:axs[0].scatter([fam]*len(rs),[r['recovery'] for r in rs],alpha=.45,s=14)
    axs[0].set(title='(A) Shift recovery by teacher',ylabel='Fraction of shifted error removed')
    axs[0].tick_params(axis='x',rotation=45)
    keys=sorted(set((r['method'],r['budget']) for r in records),key=lambda z:(z[1],z[0]))
    for k in keys:
        rs=[r for r in records if (r['method'],r['budget'])==k]
        axs[1].scatter([r['source_drift'] for r in rs],[r['recovery'] for r in rs],
                       s=14,alpha=.6,label=f'{k[0]}:{k[1]}')
    axs[1].set(title='(B) Gate-free recovery / retention',xlabel='Original-domain MSE change',
               ylabel='Shifted recovery fraction')
    axs[1].axvline(0,ls='--',lw=.7,c='.4')
    axs[1].legend(fontsize=5,ncol=2)
    means=[np.mean([r['recovery'] for r in records if (r['method'],r['budget'])==k]) for k in keys]
    axs[2].barh(range(len(keys)),means)
    axs[2].set(yticks=range(len(keys)),yticklabels=[f'{m} (B={b})' for m,b in keys],
               title='(C) Matched repair budgets',xlabel='Mean error reduction')
    axs[2].invert_yaxis()
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'Figure_02_Repair_Tradeoffs.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def paper_kernel_gallery(out):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    lag=np.linspace(0,1,L)
    for ax,family in zip(axs.flat,FAMILIES):
        for q0 in (-.8,0.,.8):
            q=torch.full((1,N),q0)
            ax.plot(lag,teacher_k(q,family)[0,25].detach().numpy(),label=f'q={q0:+.1f}')
        ax.set(xlabel='Normalized history lag',ylabel='Teacher kernel',title=family)
        ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=7)
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'Figure_00_Unmatched_Teacher_Kernels.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def csv_write(path,rows):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def benchmark_cli(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=('smoke','full'),default='smoke')
    parser.add_argument('--out',default='lag_results/ccnn_v3')
    parser.add_argument('--seeds',type=int,default=None)
    parser.add_argument('--epochs',type=int,default=None)
    parser.add_argument('--steps',type=int,default=None)
    parser.add_argument('--shift',type=float,default=.12)
    args=parser.parse_args(argv)
    if args.shift<0:parser.error('--shift must be nonnegative')
    smoke=args.profile=='smoke'
    seeds=list(range(11,11+(args.seeds or (1 if smoke else 10))))
    families=FAMILIES[:3] if smoke else FAMILIES
    epochs=args.epochs or (10 if smoke else 35)
    steps=args.steps or (15 if smoke else 85)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    arms=[('structured',1),('tangent',1),('spectral',1),('structured',5),('tangent',5),('spectral',5),
          ('structured',18),('spectral',18),('rank1',18),
          ('structured',36),('spectral',36),('rank2',36),
          ('structured',56),('spectral',56),('full',56)]
    paper_diagram(out);paper_kernel_gallery(out)
    architecture=[];repairs=[];no_defect=[]
    global CURRENT_FAMILY
    for seed in seeds:
        for fam in families:
            CURRENT_FAMILY=fam
            models={}
            for kind in ('attention','static','dynamic','ssm','lag'):
                model,metrics=fit(kind,seed,epochs=epochs,train_n=(192 if smoke else 480))
                models[kind]=model
                architecture.append(dict(family=fam,**metrics))
                print(f'Architecture {fam} {seed} {kind}: {metrics["test_mse"]:.5g}',flush=True)
            base=models['lag']
            for p in base.parameters():p.requires_grad_(False)
            for kind,budget in arms:
                row=gatefree_repair(base,fam,seed,kind,budget,steps=steps,shift=args.shift)
                repairs.append(row)
                print(f'Repair {fam}/{kind}/{budget}: shifted={row["shifted_after"]:.5g} source={row["source_after"]:.5g}',flush=True)
            for kind,budget in (('structured',1),('tangent',1),('spectral',5),('rank1',18),('full',56)):
                no_defect.append(gatefree_repair(base,fam,seed,kind,budget,steps=steps,shift=0.))
    csv_write(out/'architecture.csv',architecture)
    csv_write(out/'repair.csv',repairs)
    csv_write(out/'no_defect.csv',no_defect)
    paper_results(repairs,out)
    manifest=dict(profile=args.profile,seeds=seeds,families=list(families),
        epochs=epochs,steps=steps,seed_count=len(seeds),no_domain_gate=True,
        preservation_rule='validation source <= baseline*1.10+0.0001',
        selection='shifted val with original val constraint; test excluded',
        matched_budget=[1,5,18,36,56],shift=args.shift,negative_control='shift=0',
        cross_domain_input_pairing='same inputs per split; separate source and shifted labels',
        local_tangent_control='finite-difference CCNN parameter directions with 1 or 5 scalars',
        primary_limitation='non-matched kernel families still share discrete ODE integrator in 5 of 6 families',
        structured_large_budget='B>5 hybrid CCNN + spectral residual, not pure structured kernel',
        hypothesis_status='exploratory; no general-superiority claim')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return {'architecture':architecture,'repair':repairs,'no_defect':no_defect,'manifest':manifest}

if __name__=='__main__':
    benchmark_cli()
