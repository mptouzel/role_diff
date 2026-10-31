"""Role splitting under a linear vs a saturating score (SM Sec. S6, "Why the score saturates").

One context on one axis (d = M = 1), harmonic restoring force, random pairing. Each
step an agent meets K random partners and the read noise is averaged analytically
(2 Phi - 1), so shot noise is small and the run sits near the dense-encounter limit.
For each beta we equilibrate (warm start) and record, over the averaging window, the
mean bimodality coefficient BC = (skew^2 + 1) / kurtosis of the projection x = w g.
BC is 1 for two point masses, 5/9 for a uniform law, 1/3 for a Gaussian.

Prediction tested: once two roles at +-x* have formed, a member's within-role deviation
sees the loop at gain Lambda/2 * s'(x*), with s the score. The linear score (s' = 1)
loses the two-role state above Lambda ~ 2 and spreads into a uniform ranking; the tanh
score (s' = sech^2 x*) keeps it.

Writes data/role_splitting.npz.  Run:  python code/sim_role_splitting.py
"""
import os
import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(HERE), "data", "role_splitting.npz")

rng = np.random.default_rng(11)
N, K = 2000, 64
gamma, sigma_obs, sigma_dyn, dt = 1.0, 0.6, 0.25, 0.01
g = 0.533                                        # |g_1| of the Fig. 1 repertoire
gain = np.sqrt(2 / np.pi) / sigma_obs            # read gain of the single-sample sign rule
beta_c = gamma / (gain * g**2)                   # Lambda = beta / beta_c
Lams = np.linspace(0.25, 6.0, 24)
burn, avg, every = 1500, 1500, 50


def bc(x):
    x = x - x.mean(); m2 = (x**2).mean()
    return ((x**3).mean()**2 / m2**3 + 1) / ((x**4).mean() / m2**2)


def run(score):
    w = 0.05 * rng.standard_normal(N)
    out = np.zeros((len(Lams), 2))                # BC, variance of x
    for i, L in enumerate(Lams):
        beta = L * beta_c
        acc = []
        for t in range(burn + avg):
            x = w * g
            s = np.tanh(x) if score == "tanh" else x
            a = erf((s[:, None] - s[rng.integers(0, N, (N, K))]) / (sigma_obs * np.sqrt(2))).mean(1)
            w = w + (-gamma * w + beta * a * g) * dt + sigma_dyn * np.sqrt(dt) * rng.standard_normal(N)
            if t >= burn and t % every == 0:
                acc.append([bc(w * g), (w * g).var()])
        out[i] = np.mean(acc, 0)
        print(f"{score:6s} Lambda={L:4.2f}  BC={out[i, 0]:.3f}  var x={out[i, 1]:.3f}", flush=True)
    return out


res = {s: run(s) for s in ["linear", "tanh"]}

# two-role state: half the partners in the other role (read ~ +-1), half in the same (read ~ 0),
# so x* = beta g^2 / (2 gamma); within-role loop gain = (Lambda / 2) * s'(x*)
xstar = Lams * beta_c * g**2 / (2 * gamma)
within = {"linear": Lams / 2, "tanh": Lams / 2 / np.cosh(xstar)**2}
np.savez(DATA_PATH, Lams=Lams, beta_c=beta_c, xstar=xstar,
         bc_linear=res["linear"][:, 0], var_linear=res["linear"][:, 1],
         bc_tanh=res["tanh"][:, 0], var_tanh=res["tanh"][:, 1],
         within_linear=within["linear"], within_tanh=within["tanh"])

print("\n Lambda | BC linear  BC tanh | within-role gain linear  tanh")
for i, L in enumerate(Lams):
    print(f"  {L:4.2f}  |   {res['linear'][i, 0]:.3f}    {res['tanh'][i, 0]:.3f}  |"
          f"        {within['linear'][i]:.2f}            {within['tanh'][i]:.3f}")
il = np.argmax(res["linear"][:, 0])
print(f"\nlinear score: BC peaks at {res['linear'][il, 0]:.3f}, Lambda = {Lams[il]:.2f}; "
      f"at Lambda = {Lams[-1]:.1f} BC = {res['linear'][-1, 0]:.3f} (uniform 0.556)")
print(f"tanh score: BC at Lambda = {Lams[-1]:.1f} is {res['tanh'][-1, 0]:.3f}; "
      f"max within-role gain over Lambda >= 1: {within['tanh'][Lams >= 1].max():.3f}")
