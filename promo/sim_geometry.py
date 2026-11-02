"""Three-dimensional geometry of one schema (not part of the submission).

The other promo runs show the cascade as a spectrum. This one shows the objects
the cascade acts on: the cloud of omega-identities, the schema that scores them,
and the split of the cloud into two role lobes along the axis that schema reads.

Repertoire. d = M = 3, exactly orthogonal, with the strength profile of Fig. 1
(code/sim_fig1.py, |g_m| = 0.533, 0.434, 0.302), so the clip traces the same
cascade the figure does: beta_c = 2.65, 3.99, 8.25. The dynamics is that of
code/sim_fig1.py: tanh statuses, harmonic restoring force, read noise averaged
over K random partners per context.

What the run shows. Measured here (N = 600), the marginal dip (density at the
origin over peak density) on axis 1 is 0.40 at beta = 3 and 0 from beta = 6; on
axis 2 it is 0.24 at beta = 6 and 0 from beta = 10; on axis 3 it is 0.46 at
beta = 10 and 0 by beta = 20. Each schema divides the population in turn, so the
cloud ends in eight groups, the corners of a box aligned with the three schemas.

Saves a subcritical snapshot for the staged encounter of the clip's second act,
then the ramp frames.

Writes promo/geometry_data.npz.
"""
import os
import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "geometry_data.npz")

rng = np.random.default_rng(17)

N, d, M = 600, 3, 3          # N small: individual points must stay readable in 3D
gamma = 1.0
sigma_obs = 0.6
sigma_dyn = 0.25
dt = 0.01                       # dense-encounter limit: no shot noise to resolve
K = 32                          # partners averaged per context per step
#
rho_pair = 0.0

dirs = np.eye(d)
strength = np.array([0.533, 0.434, 0.302])          # |g_m|, the Fig. 1 profile
G = dirs * strength[:, None]
u = strength**2

gain = 2.0 / (sigma_obs * np.sqrt(2.0 * np.pi))     # g = 2 pibar'(0)
beta_c = gamma / ((1.0 - rho_pair) * gain * u)       # Lambda_k = 1
floor = sigma_dyn**2 / (2.0 * gamma)

beta_sub = 1.6                                       # below beta_c^(1) = 2.6
beta_lo, beta_hi = 1.6, 20.0
n_frames = 900
steps_per_frame = 40
sub_steps = 4000


def step(omega, beta, rng):
    """Identical in form to promo/sim_ramp.py: K partners per context, redrawn
    per context, so role assignments do not correlate across axes."""
    score = np.tanh(omega @ G.T)                         # (N, M) statuses
    partner = rng.integers(0, N, size=(N, K, M))         # K partners per context
    gaps = score[:, None, :] - score[partner, np.arange(M)]   # status differences
    a = erf(gaps / (np.sqrt(2) * sigma_obs)).mean(1)     # noise-averaged sign decision
    drift = -gamma * omega + beta * (a @ G)
    return omega + drift * dt + sigma_dyn * np.sqrt(dt) * rng.standard_normal((N, d))


# --- subcritical snapshot, for the staged encounter -------------------------
omega = 0.05 * rng.standard_normal((N, d))
for _ in range(sub_steps):
    omega = step(omega, beta_sub, rng)
omega_sub = omega.copy()
print(f"subcritical snapshot at beta={beta_sub}: rms={omega_sub.std(axis=0).round(3)}")

# --- ramp -------------------------------------------------------------------
# Exponential in beta, as sim_ramp.py: strengths compound exponentially, so a
# constant-speed marker on a log axis still reads as a clock.
betas = np.geomspace(beta_lo, beta_hi, n_frames)
cloud = np.zeros((n_frames, N, d), dtype=np.float32)
eigs = np.zeros((n_frames, d))
dips = np.zeros((n_frames, d))


def marginal_dip(x):
    """Density at the origin over peak density: 1 unimodal, 0 fully split."""
    h, e = np.histogram(x, bins=31)
    c = 0.5 * (e[1:] + e[:-1])
    return float(h[np.argmin(np.abs(c))] / max(h.max(), 1))


for i, beta in enumerate(betas):
    for _ in range(steps_per_frame):
        omega = step(omega, beta, rng)
    cloud[i] = omega
    C = (omega.T @ omega) / N
    eigs[i] = np.sort(np.linalg.eigvalsh(C))[::-1]
    dips[i] = [marginal_dip(omega[:, k]) for k in range(d)]
    if i % 150 == 0:
        print(f"frame {i:4d}  beta={beta:6.2f}  dips={dips[i].round(2)}")

np.savez(DATA_PATH, omega_sub=omega_sub.astype(np.float32), cloud=cloud,
         betas=betas, eigs=eigs, dips=dips, G=G, strength=strength, u=u,
         beta_c=beta_c, beta_sub=beta_sub, floor=floor, gamma=gamma,
         sigma_obs=sigma_obs, sigma_dyn=sigma_dyn, rho_pair=rho_pair)
print("done ->", DATA_PATH)
