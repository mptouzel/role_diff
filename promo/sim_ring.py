"""Ramped sweep for the degenerate-pair animation (not part of the submission).

promo/sim_ramp.py uses a repertoire whose leading gap is wide (u_2/u_1 = 0.25),
so axis 1 splits well before axis 2. This script runs the opposite case: axes 1
and 2 are nearly degenerate (u_2/u_1 = 0.91, thresholds 2.0 and 2.2), the case
of Sec. III where "the axes ignite ... together where the spectrum is
degenerate".

The two modes lift off the noise floor at the same beta, and each divides the
population on its own. The dynamics is that of code/sim_fig1.py: tanh statuses,
harmonic restoring force, read noise averaged over K random partners per
context. The harmonic force separates into one term per axis, so the two
divisions are independent and the status plane ends in four groups, one per
role combination. Measured here: each quadrant holds 0.24-0.26 of the agents,
the two statuses are uncorrelated (r = 0.02), and both bimodality coefficients
reach 1.00 by beta = 13.

Writes promo/ring_data.npz (status clouds, status variance per axis, and the
angular contrast of the (z_1, z_2) density).
"""
import os
import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "ring_data.npz")

rng = np.random.default_rng(7)

N, d, M = 1500, 8, 8
gamma = 1.0
sigma_obs = 0.6
sigma_dyn = 0.25
dt = 0.01                       # dense-encounter limit: no shot noise to resolve
K = 32                          # partners averaged per context per step
rho_pair = 0.0

dirs = np.eye(d)[:M]                                    # exactly orthogonal, as sim_ramp
piprime = 1.0 / (sigma_obs * np.sqrt(2 * np.pi))
gain = 2 * piprime

# Axes 1 and 2 near-degenerate (thresholds 2.0 and 2.2, u_2/u_1 = 0.91) so they
# ignite together. Axes 3-8 are pushed to thresholds well above the top of the
# ramp, so they stay under the floor for the whole clip and the inset stays
# readable: two eigenvalues rise, six do not.
beta_c_target = np.concatenate([[2.0, 2.2], np.geomspace(45.0, 150.0, M - 2)])
u = gamma / ((1.0 - rho_pair) * gain * beta_c_target)
G = dirs * np.sqrt(u)[:, None]
mu = np.sort(np.linalg.eigvalsh(G @ G.T))[::-1]
beta_c = gamma / ((1.0 - rho_pair) * gain * mu)

beta_lo, beta_hi = 1.0, 30.0                            # stops well below axis 3
n_frames = 600
steps_per_frame = 40
floor = sigma_dyn**2 / (2 * gamma)

def step_dynamics(omega, beta, rng):
    """Identical to promo/sim_ramp.py: one independent partner per context."""
    score = np.tanh(omega @ G.T)                         # (N, M) statuses
    partner = rng.integers(0, N, size=(N, K, M))         # K partners per context
    gaps = score[:, None, :] - score[partner, np.arange(M)]   # status differences
    a = erf(gaps / (np.sqrt(2) * sigma_obs)).mean(1)     # noise-averaged sign decision
    drift = -gamma * omega + beta * (a @ G)
    return omega + drift * dt + sigma_dyn * np.sqrt(dt) * rng.standard_normal((N, d))

betas = np.geomspace(beta_lo, beta_hi, n_frames)
eigs = np.zeros((n_frames, d))
Ghat = G / np.linalg.norm(G, axis=1, keepdims=True)
z_cloud = np.zeros((n_frames, N, 2), dtype=np.float32)
s_cloud = np.zeros((n_frames, N, 2), dtype=np.float32)   # statuses tanh(omega.g)
svar = np.zeros((n_frames, M))                        # status variance per axis
contrast = np.zeros(n_frames)                           # angular max/min, ring diagnostic

def angular_contrast(z):
    r = np.hypot(z[:, 0], z[:, 1])
    keep = r > np.median(r) * 0.5
    if keep.sum() < 50:
        return 1.0
    th = np.arctan2(z[keep, 1], z[keep, 0])
    h, _ = np.histogram(th, bins=72, range=(-np.pi, np.pi))
    h = np.convolve(np.r_[h, h, h], np.ones(7) / 7, "same")[72:144]
    return h.max() / max(h.min(), 1.0)           # an empty bin counts as one agent

omega = 0.05 * rng.standard_normal((N, d))
for i, beta in enumerate(betas):
    for _ in range(steps_per_frame):
        omega = step_dynamics(omega, beta, rng)
    C = (omega.T @ omega) / N
    eigs[i] = np.sort(np.linalg.eigvalsh(C))[::-1]
    z = omega @ Ghat.T
    z_cloud[i] = z[:, :2]
    st = np.tanh(omega @ G.T)
    s_cloud[i] = st[:, :2]
    svar[i] = st.var(axis=0)
    contrast[i] = angular_contrast(z[:, :2])
    if i % 100 == 0:
        print(f"frame {i:4d}  beta={beta:6.2f}  "
              f"N_active={(eigs[i] > 2*floor).sum():2d}  contrast={contrast[i]:5.2f}")

np.savez(DATA_PATH, betas=betas, eigs=eigs, beta_c=np.sort(beta_c), u=np.sort(u)[::-1],
         mu=mu, gamma=gamma, sigma_dyn=sigma_dyn, sigma_obs=sigma_obs,
         rho_pair=rho_pair, floor=floor, z_cloud=z_cloud, s_cloud=s_cloud, svar=svar, contrast=contrast)
print("done ->", DATA_PATH)
print(f"final contrast {contrast[-1]:.2f}  (about 2 for a ring, >8 for four groups)   "
      f"quadrants {[round(float(x),3) for x in [np.mean((z_cloud[-1][:,0]>0)&(z_cloud[-1][:,1]>0)), np.mean((z_cloud[-1][:,0]<0)&(z_cloud[-1][:,1]>0)), np.mean((z_cloud[-1][:,0]<0)&(z_cloud[-1][:,1]<0)), np.mean((z_cloud[-1][:,0]>0)&(z_cloud[-1][:,1]<0))]]}")
