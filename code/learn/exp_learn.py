"""Learning experiments. Usage: python exp_learn.py {sweep,sign,es} [steps]
sweep : shared myopic PG from k0=0.05 at beta in {2,6,12}, eps in {1,5}  (H2, H3)
sign  : shared myopic PG from k0=0 (10 seeds) and k0=-0.5 (3 seeds), beta=6 (H4)
es    : evolution strategy on a scalar shared k across replica populations, k0=+-0.5 (H4)
"""
import os, sys, numpy as np
import core as c

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "learn"); os.makedirs(OUT, exist_ok=True)
REC = 50  # record every REC steps


def learn_pg(k0, beta, eps, steps, seed, kmax=50.0):
    rng = np.random.default_rng(seed)
    om = 0.05 * rng.standard_normal((c.N, c.d))
    k = np.full(c.M, float(k0)); th = np.zeros(c.M)
    K, V, R, TH = [], [], [], []
    racc = 0.0
    for t in range(steps):
        om, info = c.step(om, k, beta, rng, theta=th)
        gk, gth = c.pg_grad(info, k, th)
        k = np.clip(k + eps * c.DT * gk, -kmax, kmax)
        th = th + eps * c.DT * gth
        racc += info["r"].mean()
        if (t + 1) % REC == 0:
            K.append(k.copy()); TH.append(th.copy()); V.append(c.proj_var(om)[0]); R.append(racc / REC); racc = 0.0
    return np.array(K), np.array(TH), np.array(V), np.array(R)


def W_of_k(k, beta, rng, n=500, burn=500, avg=1000):
    om = 0.05 * rng.standard_normal((n, c.d))
    kv = np.full(c.M, k); W = 0.0
    for t in range(burn + avg):
        om, info = c.step(om, kv, beta, rng)
        if t >= burn:
            W += info["r"].mean()
    return W / avg


if __name__ == "__main__":
    which = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    if which == "sweep":
        out = {}
        for beta in [2.0, 6.0, 12.0]:
            for eps in [1.0, 5.0]:
                for seed in range(2):
                    K, TH, V, R = learn_pg(0.05, beta, eps, steps, 100 + seed)
                    out[f"b{beta:g}_e{eps:g}_s{seed}"] = np.stack([K, TH, V], 1)
                    out[f"R_b{beta:g}_e{eps:g}_s{seed}"] = R
                    print(f"beta={beta} eps={eps} seed={seed}: final k={K[-1].round(2)} theta={TH[-1].round(3)} "
                          f"v={V[-1].round(3)} payoff={R[-200:].mean():.3f}", flush=True)
        np.savez(os.path.join(OUT, "learn_sweep.npz"), rec=REC, dt=c.DT, steps=steps, **out)
    elif which == "sign":
        out = {}
        for k0, seeds in [(0.0, range(10)), (-0.5, range(3))]:
            for seed in seeds:
                K, TH, V, R = learn_pg(k0, 6.0, 1.0, steps, 200 + seed)
                out[f"k0{k0:g}_s{seed}"] = np.stack([K, TH, V], 1)
                out[f"R_k0{k0:g}_s{seed}"] = R
                print(f"k0={k0} seed={seed}: final k={K[-1].round(2)} v={V[-1].round(3)} payoff={R[-200:].mean():.3f}",
                      flush=True)
        np.savez(os.path.join(OUT, "learn_sign.npz"), rec=REC, dt=c.DT, steps=steps, **out)
    elif which == "es":
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 25
        delta, eps_es = 0.2, 2.0
        out = {}
        for k0 in [-0.5, 0.5]:
            k = k0; traj = [k]; Wt = []
            for it in range(iters):
                s = 1000 * it + (0 if k0 < 0 else 500000)
                Wp = W_of_k(k + delta, 6.0, np.random.default_rng(s))   # common random numbers
                Wm = W_of_k(k - delta, 6.0, np.random.default_rng(s))
                k = k + eps_es * (Wp - Wm) / (2 * delta)
                traj.append(k); Wt.append(0.5 * (Wp + Wm))
                print(f"ES k0={k0} it={it}: k={k:.3f} W={Wt[-1]:.3f}", flush=True)
            out[f"k0{k0:g}"] = np.array(traj); out[f"W_k0{k0:g}"] = np.array(Wt)
        np.savez(os.path.join(OUT, "learn_es.npz"), delta=delta, eps_es=eps_es, **out)
