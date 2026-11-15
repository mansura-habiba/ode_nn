## 3. Expected outcomes and empirical boundaries

If the source generator's filter is truly structured, its low-dimensional parameterization should offer improved sample efficiency and a small repair surface. A sufficiently expressive dynamic filter may also learn the generator. **A favorable matched-generator result is not evidence of broad architectural superiority.** The original-domain behavior is protected by an *explicit domain-gated repair parameter*; this preservation is by construction and should not be advertised as a learned general non-regression guarantee. A generic one-scalar dynamic-filter edit is a deliberately restricted control; broader low-rank or full-kernel editing may be much stronger.

### Implementation protocol

The next cell includes all model definitions. There is no external dataset, checkpoint, token, or GitHub file required. The learned and reference kernels share a function family by design. `QUICK=1` provides a smoke test; the default uses two seeds, 30 epochs, and 480 training sequences per seed. CPU suffices.




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
    x1,q1=make_data(128,seed+311); y1=target(x1,q1)
    x2,q2=make_data(128,seed+411); y2=target(x2,q2,shift=shift)
    vx1,vq1=make_data(160,seed+511); vy1=target(vx1,vq1)
    vx2,vq2=make_data(160,seed+611); vy2=target(vx2,vq2,shift=shift)
    tx1,tq1=make_data(240,seed+711); ty1=target(tx1,tq1)
    tx2,tq2=make_data(240,seed+811); ty2=target(tx2,tq2,shift=shift)
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
    args=parser.parse_args(argv)
    smoke=args.profile=='smoke'
    seeds=list(range(11,11+(args.seeds or (1 if smoke else 10))))
    families=FAMILIES[:3] if smoke else FAMILIES
    epochs=args.epochs or (10 if smoke else 35)
    steps=args.steps or (15 if smoke else 85)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    arms=[('structured',1),('spectral',1),('structured',5),('spectral',5),
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
            for kind in ('attention','dynamic','lag'):
                model,metrics=fit(kind,seed,epochs=epochs,train_n=(192 if smoke else 480))
                models[kind]=model
                architecture.append(dict(family=fam,**metrics))
                print(f'Architecture {fam} {seed} {kind}: {metrics["test_mse"]:.5g}',flush=True)
            base=models['lag']
            for p in base.parameters():p.requires_grad_(False)
            for kind,budget in arms:
                row=gatefree_repair(base,fam,seed,kind,budget,steps=steps)
                repairs.append(row)
                print(f'Repair {fam}/{kind}/{budget}: shifted={row["shifted_after"]:.5g} source={row["source_after"]:.5g}',flush=True)
            for kind,budget in (('structured',1),('spectral',5),('rank1',18),('full',56)):
                no_defect.append(gatefree_repair(base,fam,seed,kind,budget,steps=steps,shift=0.))
    csv_write(out/'architecture.csv',architecture)
    csv_write(out/'repair.csv',repairs)
    csv_write(out/'no_defect.csv',no_defect)
    paper_results(repairs,out)
    manifest=dict(profile=args.profile,seeds=seeds,families=list(families),
        epochs=epochs,steps=steps,seed_count=len(seeds),no_domain_gate=True,
        preservation_rule='validation source <= baseline*1.10+0.0001',
        selection='shifted val with original val constraint; test excluded',
        matched_budget=[1,5,18,36,56],negative_control='shift=0',
        primary_limitation='non-matched kernel families still share discrete ODE integrator in 5 of 6 families',
        structured_large_budget='B>5 hybrid CCNN + spectral residual, not pure structured kernel',
        hypothesis_status='exploratory; no general-superiority claim')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return {'architecture':architecture,'repair':repairs,'no_defect':no_defect,'manifest':manifest}

if __name__=='__main__':
    benchmark_cli()
