"""RQ6/RQ8: small, validation-gated neural-operator edits under domain shift.

Reproducible synthetic Fourier neural-operator and coupled mean-state dynamics.
No HF model is involved; this isolates *whether* a repairable weight mechanism exists.
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PI = math.pi
DT = 0.75
A_TRUE = torch.tensor([[-0.08, 0.20], [0.13, -0.11]], dtype=torch.float64)
M_TRUE = torch.matrix_exp(A_TRUE * DT)


def make_data(seed, count, n, shifted=False, negative=False):
    """Same seed gives same continuous inputs at all resolutions n>=16."""
    rng = np.random.default_rng(seed)
    coeff_sin = rng.normal(size=(count, 5)) * (0.75 / np.arange(1, 6) ** 1.1)
    coeff_cos = rng.normal(size=(count, 5)) * (0.75 / np.arange(1, 6) ** 1.1)
    mu = rng.normal(0, 0.25, size=(count, 1))
    z = rng.normal(0, .6, size=(count, 1))
    c = rng.uniform(.62, 1.05, size=(count, 1)) if shifted else rng.uniform(-.20,.20,size=(count,1))
    x = 2 * PI * np.arange(n) / n
    u = mu + sum(coeff_sin[:, j:j+1] * np.sin((j+1)*x)
                + coeff_cos[:, j:j+1] * np.cos((j+1)*x) for j in range(5))
    u = torch.as_tensor(u, dtype=torch.float64)
    z = torch.as_tensor(z, dtype=torch.float64)
    c = torch.as_tensor(c, dtype=torch.float64)
    truth = evolve_true(u, z, c, negative=negative)
    return u,z,c,truth


def evolve_true(u,z,c,negative=False):
    n = u.shape[-1]
    k = torch.arange(n//2+1, dtype=u.dtype, device=u.device)
    hinge = torch.relu(c[:,0] - 0.3)
    D = .045 + .20*hinge.square()
    spectrum = torch.fft.rfft(u,dim=-1)
    amp = torch.exp(-(D[:,None]*k[None,:].square()+.025)*DT)
    spectrum = spectrum * amp
    if negative:  # unrepresentable for real/even diffusion multiplier: advective phase
        v = 1.1 * hinge
        phase = torch.exp(-1j * (k[None,:]*v[:,None]*DT))
        spectrum = spectrum * phase
    mz = torch.stack([u.mean(-1),z[:,0]],dim=-1)
    new_m = mz @ M_TRUE.T
    spectrum[:,0] = (new_m[:,0]*n).to(spectrum.dtype)
    result = torch.fft.irfft(spectrum,n=n,dim=-1)
    return result


class CoupledSpectralOperator(nn.Module):
    """Resolution-neutral Fourier semigroup with learned conditioned diffusivity and ODE."""
    def __init__(self, coupled=True, discrete=False):
        super().__init__()
        self.coupled = coupled
        self.discrete = discrete
        # one mechanism-specific six-weight head; hinge feature is inactive in original domain
        self.w = nn.Parameter(torch.tensor([-3.08,0.,0.,0.,0.,0.],dtype=torch.float64))
        a_init=A_TRUE.clone() + torch.tensor([[.015,-.035],[.025,.012]],dtype=torch.float64)
        if discrete:
            self.M=nn.Parameter(torch.matrix_exp(a_init*DT))
            self.register_parameter('A',None)
        else:
            self.A=nn.Parameter(a_init)

    def features(self,c):
        h = torch.relu(c[:,0]-.3)
        return torch.stack([torch.ones_like(h),c[:,0],h*h,h,c[:,0]**2,
                            torch.sin(2*PI*c[:,0])],dim=-1)

    def forward(self,u,z,c):
        n=u.shape[-1]
        p=self.features(c)
        # globally positive scalar field diffusion; no free phase for the negative control
        D = (torch.nn.functional.softplus(self.w[0]) + p[:,1:] @ self.w[1:]).clamp(min=.0001)
        k=torch.arange(n//2+1,dtype=u.dtype,device=u.device)
        spec=torch.fft.rfft(u,dim=-1)
        amp=torch.exp(-(D[:,None]*k[None,:]**2 + .025)*DT)
        spec=spec*amp
        if self.discrete:
            mat=self.M
        else:
            A=self.A if self.coupled else torch.diag(torch.diagonal(self.A))
            mat=torch.matrix_exp(A*DT)
        mz=torch.stack([u.mean(-1),z[:,0]],dim=-1)
        m=mz@mat.T
        spec[:,0]=(m[:,0]*n).to(spec.dtype)
        return torch.fft.irfft(spec,n=n,dim=-1)


def mse(model,data,with_grad=False):
    u,z,c,y=data
    if with_grad:
        return (model(u,z,c)-y).square().mean()
    with torch.no_grad():
        return float((model(u,z,c)-y).square().mean())


def evaluate(model,orig_val,shift_val):
    return mse(model,orig_val),mse(model,shift_val)


def accepted(model,orig_val,shift_val,base_orig,base_shift,orig_tol=0.10,
             abs_floor=1e-6,shift_gain=.01):
    orig,shift=evaluate(model,orig_val,shift_val)
    ok=((base_shift-shift) >= max(1e-5,shift_gain*base_shift) and
        orig <= base_orig*(1+orig_tol)+abs_floor and
        math.isfinite(orig) and math.isfinite(shift))
    return bool(ok),orig,shift


def train_base(seed,train,steps=120,coupled=True,discrete=False):
    torch.manual_seed(seed)
    model=CoupledSpectralOperator(coupled=coupled,discrete=discrete)
    optim=torch.optim.Adam(model.parameters(), lr=.035)
    u,z,c,y=train
    for s in range(steps):
        ids=torch.randint(len(u),(min(72,len(u)),))
        optim.zero_grad()
        loss=(model(u[ids],z[ids],c[ids])-y[ids]).square().mean()
        loss.backward()
        optim.step()
    return model


def copy_with_edit(base,coordinate,amount):
    new=copy.deepcopy(base)
    with torch.no_grad():new.w[coordinate].add_(float(amount))
    return new


def find_repairs(base, orig_train, shift_train, orig_val, shift_val,
                 seed, edit_limit=.9, adam_steps=70, full_steps=70):
    orig0,shift0=evaluate(base,orig_val,shift_val)
    t0=time.perf_counter()
    # Each coordinate's local linearized effect is estimated from shifted TRAIN only.
    u,z,c,y=shift_train
    selected=(u[:min(48,len(u))],z[:min(48,len(u))],
              c[:min(48,len(u))],y[:min(48,len(u))])
    a,b,h,target=selected
    with torch.no_grad():resid=(target-base(a,b,h)).detach()
    options=[]
    for i in range(len(base.w)):
        def pred(scalar):
            from torch.func import functional_call
            p=dict(base.named_parameters())
            modified=base.w.clone()
            modified=modified.scatter(0,torch.tensor([i]),(base.w[i]+scalar).reshape(1))
            p['w']=modified
            return functional_call(base,p,(a,b,h))
        scalar=torch.zeros((),dtype=torch.float64,requires_grad=True)
        # One Jacobian vector per candidate edit; index i is selected by validation, not oracle.
        J=torch.autograd.functional.jacobian(pred,scalar,vectorize=False)
        step=float((J*resid).sum()/((J*J).sum()+1e-5))
        for fac in (.25,.5,1.,1.5):
            d=float(np.clip(step*fac,-edit_limit,edit_limit))
            trial=copy_with_edit(base,i,d)
            ok,o,s=accepted(trial,orig_val,shift_val,orig0,shift0)
            options.append({'index':i,'delta':d,'orig_val':o,'shift_val':s,'accepted':ok})
    valid=[x for x in options if x['accepted']]
    best=min(valid,key=lambda x:x['shift_val']) if valid else None
    jac_model=copy_with_edit(base,best['index'],best['delta']) if best else copy.deepcopy(base)
    direct_s=time.perf_counter()-t0
    # Sparse Adam is allowed exactly the same selected parameter (if no candidate, select best shift-val probe).
    best_any=min(options,key=lambda x:x['shift_val'])
    idx=best['index'] if best else best_any['index']
    t0=time.perf_counter()
    edit=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
    optim=torch.optim.Adam([edit],lr=.04)
    best_adam=None
    for t in range(adam_steps):
        from torch.func import functional_call
        optim.zero_grad()
        modified=base.w.clone().scatter(0,torch.tensor([idx]),
                                        (base.w[idx]+edit).reshape(1))
        pred=functional_call(base,dict(base.named_parameters(),w=modified),(u,z,c))
        loss=(pred-y).square().mean()
        loss.backward()
        optim.step()
        with torch.no_grad(): edit.clamp_(-edit_limit,edit_limit)
        if (t+1)%5==0:
            candidate=copy_with_edit(base,idx,float(edit.detach()))
            ok,o,s=accepted(candidate,orig_val,shift_val,orig0,shift0)
            if ok and (best_adam is None or s < best_adam['shift_val']):
                best_adam={'index':idx,'delta':float(edit.detach()),'orig_val':o,'shift_val':s}
    adam_model=copy_with_edit(base,idx,best_adam['delta']) if best_adam else copy.deepcopy(base)
    adam_s=time.perf_counter()-t0
    # Full fine-tuning: all parameters, original-domain replay penalty (no test leakage).
    t0=time.perf_counter()
    full=copy.deepcopy(base)
    optim=torch.optim.Adam(full.parameters(),lr=.015)
    ou,oz,oc,oy=orig_train
    best_full=None
    for t in range(full_steps):
        optim.zero_grad()
        s_loss=mse(full,shift_train,True)
        oids=torch.randint(len(ou),(min(48,len(ou)),))
        r_loss=(full(ou[oids],oz[oids],oc[oids])-oy[oids]).square().mean()
        (s_loss+3*r_loss).backward()
        optim.step()
        if (t+1)%5==0:
            ok,o,s=accepted(full,orig_val,shift_val,orig0,shift0)
            if ok and (best_full is None or s < best_full['shift_val']):
                best_full={'state':copy.deepcopy(full.state_dict()),'orig_val':o,'shift_val':s}
    if best_full:full.load_state_dict(best_full['state'])
    else:full=copy.deepcopy(base)
    full_s=time.perf_counter()-t0
    # Random control uses same 1-coordinate edit and identical two-domain validation gate.
    t0=time.perf_counter()
    rng=np.random.default_rng(seed+991)
    best_rand=None
    for _ in range(48):
        ri=int(rng.integers(0,len(base.w)))
        amount=float(rng.uniform(-edit_limit,edit_limit))
        trial=copy_with_edit(base,ri,amount)
        ok,o,s=accepted(trial,orig_val,shift_val,orig0,shift0)
        if ok and (best_rand is None or s<best_rand['shift_val']):
            best_rand={'index':ri,'delta':amount,'orig_val':o,'shift_val':s}
    rand_model=copy_with_edit(base,best_rand['index'],best_rand['delta']) if best_rand else copy.deepcopy(base)
    random_s=time.perf_counter()-t0
    models={'frozen':base,'jacobian_sparse':jac_model,'adam_sparse':adam_model,
            'full_finetune':full,'random_sparse':rand_model}
    meta={
        'frozen':{'accepted':False,'edited_params':0,'time_s':0.},
        'jacobian_sparse':{'accepted':best is not None,'edited_params':int(best is not None),'time_s':direct_s,'edit':best},
        'adam_sparse':{'accepted':best_adam is not None,'edited_params':int(best_adam is not None),'time_s':adam_s,'edit':best_adam},
        'full_finetune':{'accepted':best_full is not None,'edited_params':sum(p.numel() for p in full.parameters()) if best_full else 0,'time_s':full_s},
        'random_sparse':{'accepted':best_rand is not None,'edited_params':int(best_rand is not None),'time_s':random_s,'edit':best_rand},
    }
    return models,meta,{'orig_base_val':orig0,'shift_base_val':shift0,
                        'selected_coordinate':idx,'candidates':options}


def run(seed=17,train_steps=120,repair_steps=70,test_count=128):
    st=time.perf_counter()
    original_train=make_data(seed+100,400,16)
    original_val=make_data(seed+200,120,16)
    shifted_train=make_data(seed+300,64,16,shifted=True)
    shifted_val=make_data(seed+400,96,16,shifted=True)
    base=train_base(seed,original_train,steps=train_steps)
    frozen_test=mse(base,make_data(seed+500,test_count,16))
    models,meta,diagnostics=find_repairs(base,original_train,shifted_train,
                                         original_val,shifted_val,seed,
                                         adam_steps=repair_steps,full_steps=repair_steps)
    # RQ6 falsification: attempt exactly the same procedure on a physics shift
    # that cannot be represented by the diffusion-only operator (advection phase).
    neg_train=make_data(seed+330,64,16,shifted=True,negative=True)
    neg_val=make_data(seed+430,96,16,shifted=True,negative=True)
    neg_models,neg_meta,_=find_repairs(base,original_train,neg_train,
                                        original_val,neg_val,seed+1001,
                                        adam_steps=repair_steps,full_steps=repair_steps)
    # No-defect control: a false repair request on the original regime.
    no_defect_models,no_defect_meta,_=find_repairs(base,original_train,
        make_data(seed+350,64,16),original_val,
        make_data(seed+450,96,16),seed+2001,
        adam_steps=20,full_steps=20)
    # Strong fixed-horizon control: a direct learned state transition M, with the
    # same 6 spectral weights and 4 global-map weights, but no continuous ODE.
    discrete=train_base(seed+3,original_train,steps=train_steps,discrete=True)
    disc_models,disc_meta,_=find_repairs(discrete,original_train,shifted_train,
                                         original_val,shifted_val,seed+3001,
                                         adam_steps=repair_steps,full_steps=repair_steps)
    rows=[]
    for n in (16,32,64,128):
        base_test=make_data(seed+500,test_count,n)
        shifted_test=make_data(seed+600,test_count,n,shifted=True)
        negative_test=make_data(seed+700,test_count,n,shifted=True,negative=True)
        for method,model in models.items():
            o=mse(model,base_test);s=mse(model,shifted_test);neg=mse(model,negative_test)
            rows.append({'seed':seed,'n':n,'method':method,'original_test_mse':o,
                         'shift_test_mse':s,'unrepresentable_shift_test_mse':neg,
                         'accepted':meta[method]['accepted'],
                         'edited_params':meta[method]['edited_params'],
                         'repair_time_s':meta[method]['time_s']})
        for method in ('frozen','jacobian_sparse','adam_sparse'):
            model=disc_models[method]
            rows.append({'seed':seed,'n':n,'method':'direct_map_'+method,
                         'original_test_mse':mse(model,base_test),
                         'shift_test_mse':mse(model,shifted_test),
                         'unrepresentable_shift_test_mse':mse(model,negative_test),
                         'accepted':disc_meta[method]['accepted'],
                         'edited_params':disc_meta[method]['edited_params'],
                         'repair_time_s':disc_meta[method]['time_s']})
    control_metrics={
        'unrepresentable_shift':{method:{
            'accepted':neg_meta[method]['accepted'],
            'test_mse_n128':mse(model,make_data(seed+700,test_count,128,shifted=True,negative=True))
        } for method,model in neg_models.items()},
        'no_defect':{method:{'accepted':no_defect_meta[method]['accepted']}
                      for method in no_defect_models}
    }
    return {'seed':seed,'seconds':time.perf_counter()-st,'controls':control_metrics,'direct_map_base_params':sum(p.numel() for p in discrete.parameters()),'base_params':sum(p.numel() for p in base.parameters()),
            'base_in_domain_training_test_mse':frozen_test,
            'diagnostics':diagnostics,'metadata':meta,'rows':rows}


def summarize(runs):
    rows=[row for x in runs for row in x['rows']]
    outcomes=[]
    for n in (16,32,64,128):
        frozen=[r for r in rows if r['n']==n and r['method']=='frozen']
        mbase=np.mean([r['shift_test_mse'] for r in frozen]);obase=np.mean([r['original_test_mse'] for r in frozen])
        for method in ('frozen','jacobian_sparse','adam_sparse','full_finetune','random_sparse','direct_map_frozen','direct_map_jacobian_sparse','direct_map_adam_sparse'):
            vals=[r for r in rows if r['n']==n and r['method']==method]
            change=np.mean([r['shift_test_mse'] for r in vals])
            original=np.mean([r['original_test_mse'] for r in vals])
            negative=np.mean([r['unrepresentable_shift_test_mse'] for r in vals])
            outcomes.append({'n':n,'method':method,'shift_test_mse':float(change),
                             'relative_shift_gain_pct':float(100*(mbase-change)/mbase),
                             'original_test_mse':float(original),
                             'original_delta_mse':float(original-obase),
                             'negative_test_mse':float(negative),
                             'accepted_fraction':float(np.mean([r['accepted'] for r in vals]))})
    return outcomes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--train-steps',type=int,default=120)
    p.add_argument('--repair-steps',type=int,default=70)
    p.add_argument('--test-count',type=int,default=128)
    p.add_argument('--out',type=str,default='rq6_rq8_operator_repair_results.json')
    args=p.parse_args()
    torch.set_num_threads(min(2,max(1,torch.get_num_threads())))
    runs=[]
    for seed in args.seeds:
        result=run(seed,args.train_steps,args.repair_steps,args.test_count)
        runs.append(result)
        print('seed',seed,'sec',round(result['seconds'],1),
              'edits', {k:v.get('edit',{} ) for k,v in result['metadata'].items() if k in ('jacobian_sparse','adam_sparse')},flush=True)
    agg=summarize(runs)
    obj={'status':'exploratory_synthetic_cpu','task':'RQ6/RQ8 small verified edit to a trained Fourier-coupled neural operator',
         'seed_count':len(args.seeds),'train_steps':args.train_steps,
         'repair_steps':args.repair_steps,'test_count':args.test_count,
         'scope_notes':['same continuous input functions at each grid resolution',
                        'operating condition c visible to model, allowing identifiable conditional shift',
                        'unrepresentable advective negative control has no phase term in the model',
                        'no Hugging Face, no attention model: tests numerical edit feasibility first',
                        'validation gates are not mathematical guarantees or test-set acceptance'],
         'runs':runs,'summary':agg}
    Path(args.out).write_text(json.dumps(obj,indent=2))
    print('\nN=128 outcomes (mean of seeds; lower MSE better)')
    for r in agg:
        if r['n']==128:
            print(f"{r['method']:18} shift={r['shift_test_mse']:.7g} gain={r['relative_shift_gain_pct']:6.2f}% orig={r['original_test_mse']:.7g} neg={r['negative_test_mse']:.7g} accepted={r['accepted_fraction']:.2f}")
    print('saved',args.out)

if __name__=='__main__':main()