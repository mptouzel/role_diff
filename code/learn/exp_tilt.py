"""Correlated asymmetry: (Go,Go) pays -1 + c*tanh(Delta/D0); pair welfare unchanged.
Usage:
  python exp_tilt.py basin          exact myopic gradient dk/dt at fixed k (k<0 to 0.5) for each c, beta=6
  python exp_tilt.py learn [steps]  shared PG from k0 in {0, -0.5, -3} for c in C_LIST, beta=6
"""
import os, sys, numpy as np
import core as c
from exp_learn import learn_pg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "learn")
C_LIST = [0.0, 0.25, 0.5, 1.0, 2.0]
BETA = 6.0

if sys.argv[1] == "basin":
    ks = np.r_[-np.geomspace(0.05, 8, 18)[::-1], 0.0, np.geomspace(0.05, 0.5, 5)]
    rng = np.random.default_rng(21)
    grad = np.zeros((len(C_LIST), len(ks), c.M)); v = np.zeros((len(ks), c.M))
    for j, k in enumerate(ks):
        kv = np.full(c.M, k); z = np.zeros(c.M)
        om = 0.05 * rng.standard_normal((c.N, c.d))
        for _ in range(800):
            om, _ = c.step(om, kv, BETA, rng)
        n = 0
        for t in range(600):
            om, info = c.step(om, kv, BETA, rng)
            if t % 3 == 0:
                for i, ct in enumerate(C_LIST):
                    grad[i, j] += c.analytic_grad(info, kv, z, c=ct)
                v[j] += c.proj_var(om)[0]; n += 1
        grad[:, j] /= n; v[j] /= n
        print(f"k={k:7.3f} v1={v[j,0]:.3f} grad ctx1 by c: {grad[:, j, 0].round(4)}", flush=True)
    np.savez(os.path.join(OUT, "tilt_basin.npz"), ks=ks, c_list=C_LIST, grad=grad, v=v, beta=BETA, D0=c.D0)

elif sys.argv[1] == "learn":
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    out = {}
    plan = [(0.0, 10, [0.1, 0.25, 0.5, 1.0]), (-0.5, 3, [0.5, 1.0, 2.0]), (-3.0, 3, [0.5, 1.0, 2.0])]
    for k0, nseed, cs in plan:
        for ct in cs:
            for s in range(nseed):
                K, TH, V, R = learn_pg(k0, BETA, 1.0, steps, seed0 + s, c_tilt=ct)
                out[f"k0{k0:g}_c{ct:g}_s{s}"] = np.stack([K, TH, V], 1)
                out[f"R_k0{k0:g}_c{ct:g}_s{s}"] = R
                print(f"k0={k0} c={ct} seed={s}: final k={K[-1].round(2)} v={V[-1].round(3)} "
                      f"payoff={R[-100:].mean():.3f}", flush=True)
    np.savez(os.path.join(OUT, f"tilt_learn_seed{seed0}.npz"), rec=50, dt=c.DT, steps=steps, beta=BETA, **out)

elif sys.argv[1] == "fresh":
    # fresh-seed replications of the orientation counts, seeds disjoint from learn_sign (200-209)
    # and tilt_learn_seed300 (300-309): (k0, c, first seed, n seeds, steps)
    plan = [(0.0, 0.0, 400, 12, 20000), (0.0, 0.1, 410, 10, 15000),
            (-0.5, 1.0, 420, 1, 15000), (-0.5, 2.0, 420, 1, 15000)]
    out = {}
    for k0, ct, seed0, nseed, steps in plan:
        for s in range(nseed):
            K, TH, V, R = learn_pg(k0, BETA, 1.0, steps, seed0 + s, c_tilt=ct)
            out[f"k0{k0:g}_c{ct:g}_s{s}"] = np.stack([K, TH, V], 1)
            out[f"R_k0{k0:g}_c{ct:g}_s{s}"] = R
            print(f"k0={k0} c={ct} seed={seed0 + s}: final k={K[-1].round(2)} payoff={R[-100:].mean():.3f}",
                  flush=True)
    np.savez(os.path.join(OUT, "tilt_fresh.npz"), rec=50, dt=c.DT, beta=BETA, **out)
