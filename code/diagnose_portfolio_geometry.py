"""How does cascade depth depend on portfolio geometry?
Fixed schemas, quasi-static beta sweep (the Fig.1 protocol), varying aspect
ratio M/d and coherence chi.  Depth = # covariance eigenvalues above the floor."""
import numpy as np
gamma,sobs,sdyn,dt,N = 1.0,0.6,0.25,0.01,1500
floor = sdyn**2/(2*gamma)

def equilibrate(G,d,beta,rng,steps=2500,om=None):
    M=G.shape[0]
    if om is None: om=0.01*rng.standard_normal((N,d))
    for _ in range(steps):
        perm=rng.permutation(N); p=np.empty(N,int)
        p[perm[::2]]=perm[1::2]; p[perm[1::2]]=perm[::2]
        S=(om-om[p])@G.T + sobs*rng.standard_normal((N,M))
        fb=(2*(S>0)-1.0)@G
        om+=(-gamma*om*(1+np.sum(om**2,axis=1,keepdims=True))+beta*fb)*dt \
            + sdyn*np.sqrt(dt)*rng.standard_normal((N,d))
    return om

def portfolio(M,d,chi,u,rng):
    if chi<=0:                                   # near-orthogonal (needs M<=d)
        Q,_=np.linalg.qr(rng.standard_normal((d,d))); dirs=Q[:,:min(M,d)].T
        if M>d:                                  # pad with random extras
            ex=rng.standard_normal((M-d,d)); ex/=np.linalg.norm(ex,axis=1,keepdims=True)
            dirs=np.vstack([dirs,ex])
    else:
        v=rng.standard_normal(d); v/=np.linalg.norm(v)
        b=rng.standard_normal((M,d))
        dirs=(1-chi)*b + chi*np.abs(rng.standard_normal((M,1)))*v
        dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    return dirs*np.sqrt(u[:M])[:,None]

def depth(G,d,rng,beta=3.0):
    om=equilibrate(G,d,beta,rng)
    C=(om.T@om)/N
    ev=np.sort(np.linalg.eigvalsh(C))[::-1]
    return int((ev>1.6*floor).sum()), ev/floor

u_geo = 2.6*np.exp(-np.arange(12)/2.2)           # geometric, Fig-1-like hierarchy
print(f"strengths: {np.round(u_geo[:8],2)}   (floor-crossing threshold = 1.6x floor)")
print(f"{'geometry':<34}{'depth':>7}   top-5 lambda/floor")
print("-"*74)
for M,d,chi,lab in [(3,3,0.0,'M=3  d=3   near-orthogonal'),
                    (6,6,0.0,'M=6  d=6   near-orthogonal'),
                    (8,8,0.0,'M=8  d=8   near-orthogonal'),
                    (6,6,0.0001,'M=6  d=6   random'),
                    (8,6,0.0001,'M=8  d=6   random (overcomplete)'),
                    (12,6,0.0001,'M=12 d=6   random (overcomplete)'),
                    (6,6,0.4,'M=6  d=6   coherent chi=0.4'),
                    (6,6,0.7,'M=6  d=6   coherent chi=0.7')]:
    rng=np.random.default_rng(11)
    G=portfolio(M,d,chi,u_geo,rng)
    k,ev=depth(G,d,rng)
    print(f"{lab:<34}{k:>5}     {np.round(ev[:5],2)}")
