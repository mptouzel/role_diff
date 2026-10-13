"""Phase 1 checks: sign mode vs sim_fig1 data, and fixed-k sweeps (no learning).
Saves data/learn/fixed_k.npz with, per (beta, k): projected variances v_m, eigenvalues of C,
mean payoff, and the analytic small-k gradient."""
import os, sys, numpy as np
import core as c

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "learn"); os.makedirs(OUT, exist_ok=True)
burn, avg = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (1200, 1500)


def run(k, beta, mode, rng, om=None):
    om = 0.05 * rng.standard_normal((c.N, c.d)) if om is None else om
    kv = np.full(c.M, k)
    for _ in range(burn):
        om, _ = c.step(om, kv, beta, rng, mode=mode)
    V, E, R, D2 = 0, 0, 0, 0
    GR = np.zeros(c.M); ng = 0
    for _ in range(avg):
        om, info = c.step(om, kv, beta, rng, mode=mode)
        v, e = c.proj_var(om)
        V, E, R = V + v, E + e, R + info["r"].mean()
        D2 = D2 + (info["Delta"] ** 2).mean(0)
        if mode != "sign" and abs(k) <= 0.5 and _ % 5 == 0:
            GR = GR + c.analytic_grad(info, kv, np.zeros(c.M)); ng += 1
    return om, V / avg, E / avg, R / avg, GR / max(ng, 1), D2 / avg


rng = np.random.default_rng(11)
# 1. sign mode vs committed fig1 data
f1 = np.load(os.path.join(ROOT, "data", "fig1_data.npz"))
sign_rows = []
for beta in [2.0, 6.0, 12.0]:
    _, v, e, r, _, _ = run(0, beta, "sign", rng)
    i = np.argmin(abs(f1["betas"] - beta))
    sign_rows.append((beta, e, f1["eigs"][i], r))
    print(f"sign beta={beta}: eigs {e.round(3)}  fig1 {f1['eigs'][i].round(3)}  payoff {r:.3f}")

# 2. fixed-k sweeps, both signs
ks = np.r_[-np.geomspace(0.05, 20, 10)[::-1], 0.0, np.geomspace(0.05, 20, 16)]
betas = [2.0, 6.0, 12.0]
res = {b: dict(v=[], e=[], r=[], gr=[], d2=[]) for b in betas}
for beta in betas:
    om = None
    for k in ks:
        om, v, e, r, gr, d2 = run(k, beta, "tanh", rng, None)
        for key, val in zip(["v", "e", "r", "gr", "d2"], [v, e, r, gr, d2]):
            res[beta][key].append(val)
    print(f"beta={beta} done; payoff at k=+20 {res[beta]['r'][-1]:.3f}, k=-20 {res[beta]['r'][0]:.3f}")

np.savez(os.path.join(OUT, "fixed_k.npz"), ks=ks, betas=np.array(betas),
         v=np.array([res[b]["v"] for b in betas]), e=np.array([res[b]["e"] for b in betas]),
         r=np.array([res[b]["r"] for b in betas]), gr=np.array([res[b]["gr"] for b in betas]),
         d2=np.array([res[b]["d2"] for b in betas]),
         sign_beta=np.array([s[0] for s in sign_rows]), sign_eigs=np.array([s[1] for s in sign_rows]),
         fig1_eigs=np.array([s[2] for s in sign_rows]), sign_payoff=np.array([s[3] for s in sign_rows]),
         U=c.U, burn=burn, avg=avg, seed=11)
print("saved fixed_k.npz")
