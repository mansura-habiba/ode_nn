import torch, numpy as np, scipy.linalg as la, time, json, sys, os
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(2)
N=10; HIST=7; WIDTH=32
# True linear coupled PDE-ODE dynamics for a ring of 10 spatial channels.
L=np.zeros((N,N))
for i in range(N):
    L[i,(i-1)%N]=1;L[i,i]=-2;L[i,(i+1)%N]=1
G=np.zeros((N+1,N+1))
G[:N,:N]=0.85*L-0.12*np.eye(N)
G[:N,N]=0.45*(1+0.35*np.cos(np.arange(N)*2*np.pi/N))
G[N,:N]=.50/N
G[N,N]=-.75
# Negative control: a nonlocal stable dynamical system without spatial diffusion.
DYNAMICS=os.environ.get('TOY_DYNAMICS','local_coupled')
if DYNAMICS=='nonlocal':
    rng_nonlocal=np.random.default_rng(891)
    R=rng_nonlocal.normal(size=(N,N))
    G[:N,:N]=-.35*np.eye(N)+.85*(R-R.T)/np.sqrt(2*N)
    G[:N,N]=.4*rng_nonlocal.normal(size=N)
    G[N,:N]=.3*rng_nonlocal.normal(size=N)/N
    G[N,N]=-.6
w,V=np.linalg.eig(G)
Vi=np.linalg.inv(V)

def sample(num,seed,shift=False,uniform=False,obs_stretch=False):
    rng=np.random.default_rng(seed)
    # Per recording distinct state; no overlap between splits
    base=rng.normal(size=(num,3))
    x=np.arange(N)[None,:]
    u0=(base[:,0,None]*np.sin(2*np.pi*x/N)+base[:,1,None]*np.cos(4*np.pi*x/N)
        +0.6*rng.normal(size=(num,N)))
    z0=base[:,2,None]
    y0=np.concatenate([u0,z0],axis=1)
    coeff=y0@Vi.T
    if uniform:
        times=np.repeat(np.arange(1,HIST+1,dtype=float)[None,:]*0.18,num,axis=0)
    else:
        gaps=rng.uniform(0.05,0.75 if obs_stretch else 0.38,size=(num,HIST))
        times=np.cumsum(gaps,axis=1)
    dt=rng.uniform(.35,.95,size=num) if shift else rng.uniform(.08,.30,size=num)
    all_t=np.concatenate([times,(times[:,-1]+dt)[:,None]],axis=1)
    expw=np.exp(all_t[:,:,None]*w[None,None,:])
    full=np.einsum('btk,bk,lk->btl',expw,coeff,V).real
    obs=full[:,:HIST,:N]+rng.normal(0,.07,size=(num,HIST,N))
    target=full[:,-1,:N]
    return tuple(torch.tensor(a,dtype=torch.float32) for a in (obs,times,dt[:,None],target))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Linear(N+2,WIDTH)
        self.attn=nn.MultiheadAttention(WIDTH,4,batch_first=True)
        self.norm=nn.LayerNorm(WIDTH)
        self.den=nn.Linear(WIDTH,N)
        self.zinit=nn.Linear(WIDTH,1)
    def forward(self,x,t):
        diff=torch.cat([t[:,:1],t[:,1:]-t[:,:-1]],dim=1)
        a=self.emb(torch.cat([x,t[:,:,None],diff[:,:,None]],dim=-1))
        att,_=self.attn(a,a,a,need_weights=False)
        h=self.norm(a+att)[:,-1]
        u=x[:,-1]+.12*torch.tanh(self.den(h))
        z=self.zinit(h)
        return u,z,h

class Toy(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind=kind
        self.enc=Encoder()
        if kind=='attention':
            # A stronger, horizon-aware attention baseline with more parameters.
            self.net=nn.Sequential(nn.Linear(N+WIDTH+2,64),nn.Tanh(),nn.Linear(64,N))
        else:
            if kind=='neural_ode':
                self.dense=nn.Sequential(nn.Linear(N+1,32),nn.Tanh(),nn.Linear(32,N+1))
            self.conv=nn.Conv1d(1,1,3,padding=0,bias=False)
            self.rawd=nn.Parameter(torch.tensor(-0.2))
            self.rawdecay=nn.Parameter(torch.tensor(-1.4))
            self.coupling=nn.Parameter(torch.ones(N)*.15)
            self.zdecay=nn.Parameter(torch.tensor(-.7))
            self.zdrive=nn.Parameter(torch.tensor(.25))
    def dyn(self,u,z):
        if self.kind=='neural_ode':
            dz=self.dense(torch.cat([u,z],dim=-1))
            return dz[:,:N],dz[:,N:]
        lap=torch.roll(u,-1,-1)-2*u+torch.roll(u,1,-1)
        # Periodic 3-point convolution, with no zero-padding boundary artifact.
        a=self.conv.weight[0,0,0]*torch.roll(u,1,-1)+self.conv.weight[0,0,1]*u+self.conv.weight[0,0,2]*torch.roll(u,-1,-1)
        du=(F.softplus(self.rawd) if self.kind!='ccnn_nodiff' else 0)*lap-F.softplus(self.rawdecay)*u + self.coupling*(z if self.kind!='ccnn_noglobal' else 0) +.15*torch.tanh(a)
        dz=-F.softplus(self.zdecay)*z+self.zdrive*u.mean(-1,keepdim=True)
        return du,dz
    def forward(self,x,t,dt):
        u,z,h=self.enc(x,t)
        if self.kind=='attention':
            return u+dt*self.net(torch.cat([u,h,dt,dt.square()],dim=-1))
        if self.kind=='euler':
            du,dz=self.dyn(u,z)
            return u+dt*du
        # 4-step full-horizon RK4 integration, z persists.
        hstep=dt/4
        for k in range(4):
            a,b=self.dyn(u,z)
            c,d=self.dyn(u+.5*hstep*a,z+.5*hstep*b)
            e,f=self.dyn(u+.5*hstep*c,z+.5*hstep*d)
            g,j=self.dyn(u+hstep*e,z+hstep*f)
            u=u+hstep*(a+2*c+2*e+g)/6
            z=z+hstep*(b+2*d+2*f+j)/6
        return u

def train(kind,seed):
    torch.manual_seed(seed)
    m=Toy(kind)
    opt=torch.optim.AdamW(m.parameters(),lr=.004)
    train=sample(900,seed+11)
    val=sample(220,seed+101)
    train_batches=10
    best=1e9;beststate=None
    start=time.perf_counter()
    for epoch in range(24):
        m.train()
        idx=torch.randperm(len(train[0]))
        for inds in idx.split(120):
            opt.zero_grad()
            pred=m(*(a[inds] for a in train[:3]))
            loss=F.mse_loss(pred,train[3][inds]);loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),2)
            opt.step()
        m.eval()
        with torch.no_grad():
            pred=m(*val[:3]);v=F.mse_loss(pred,val[3]).item()
        if v<best:
            best=v;beststate={k:p.detach().clone() for k,p in m.state_dict().items()}
    m.load_state_dict(beststate)
    out={'kind':kind,'seed':seed,'val':best,'train_seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in m.parameters())}
    m.eval()
    with torch.no_grad():
        for domain,kwargs in [('iid',{}),('long_gap',{'shift':True}),('regular',{'uniform':True}),('stretched_observations',{'obs_stretch':True})]:
            ex=sample(320,seed+1001,**kwargs)
            start_inf=time.perf_counter()
            out[domain]=F.mse_loss(m(*ex[:3]),ex[3]).item()
            if domain=='long_gap':out['inference_ms_320']=1000*(time.perf_counter()-start_inf)
    return out

if __name__=='__main__':
 results=[]
 for seed in [1,2,3]:
  for kind in (('attention','euler','ccnn','neural_ode','ccnn_nodiff','ccnn_noglobal') if DYNAMICS=='local_coupled' else ('attention','euler','ccnn','neural_ode')):
   res=train(kind,seed)
   results.append(res)
   print(res,flush=True)
 with open('/mnt/data/ccnn_attention_toy_results_'+DYNAMICS+'.json','w') as f:json.dump(results,f,indent=2)
 print('\nSUMMARY:')
 for k in (('attention','euler','ccnn','neural_ode','ccnn_nodiff','ccnn_noglobal') if DYNAMICS=='local_coupled' else ('attention','euler','ccnn','neural_ode')):
  r=[x for x in results if x['kind']==k]
  print(k,{m:(round(float(np.mean([x[m] for x in r])),6),round(float(np.std([x[m] for x in r])),6)) for m in ('iid','long_gap','regular','stretched_observations','train_seconds','inference_ms_320')})