"""Ramped sweep for the promotional animation (not part of the submission).

Fig. 1 equilibrates the population at each beta. Here beta instead rises
linearly in time, beta(t), as in Sec. V of the main text, and the INSTANTANEOUS
covariance is recorded every frame. That is what makes the eigenvalues visibly
fluctuate and then lift off the noise floor one at a time.

The dynamics is the one of code/sim_fig1.py, unchanged. Only the schedule
(a ramp rather than per-beta equilibration) and the repertoire size differ:
M = d = 16 modes on a geometric strength profile, so the thresholds
beta_c^(k) = gamma / ((1-rho_pair) g u_k) are spaced logarithmically and the
cascade is class (ii) of Sec. III B.

Writes promo/ramp_data.npz.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "ramp_data.npz")

rng = np.random.default_rng(11)

N, d, M = 1200, 16, 16          # smaller N: visible finite-population jitter
gamma = 1.0
sigma_obs = 0.6
sigma_dyn = 0.25
dt = 0.02
rho_pair = 0.0                      # random pairing, factor (1-rho_pair) = 1

# --- repertoire: near-orthogonal directions, geometric strengths ------------
base = np.eye(d)
dirs = np.array([base[m] + 0.12 * rng.standard_normal(d) for m in range(M)])
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

piprime = 1.0 / (sigma_obs * np.sqrt(2 * np.pi))
gain = 2 * piprime                                      # read gain g = 2 pibar'(0)
# Thresholds are compressed to ~2..5: the quartic term is isotropic, so once
# early modes saturate the shared |omega|^2 lifts the effective threshold of
# every remaining axis by (1 + <|omega|^2>). A wide linear hierarchy therefore
# stalls its own tail long before the sweep ends.
k0 = 16.4                                               # log-spacing of thresholds
beta_c_target = 2.0 * np.exp(np.arange(M) / k0)         # ~2 ... ~5
u = gamma / ((1.0 - rho_pair) * gain * beta_c_target)   # u_k = |g_k|^2
strength = np.sqrt(u)
G = dirs * strength[:, None]
beta_c = gamma / ((1.0 - rho_pair) * gain * u)          # = beta_c_target

# --- ramp schedule ---------------------------------------------------------
beta_lo, beta_hi = 0.5, 42.0
n_frames = 900
steps_per_frame = 32                                    # 22400 steps total
floor = sigma_dyn**2 / (2 * gamma)                      # noise floor sigma_w^2/2gamma

def step_dynamics(omega, beta, rng):
    perm = rng.permutation(N)
    partner = np.empty(N, dtype=int)
    partner[perm[::2]] = perm[1::2]
    partner[perm[1::2]] = perm[::2]
    gaps = omega - omega[partner]
    S = gaps @ G.T + sigma_obs * rng.standard_normal((N, M))
    fb = ((2 * (S > 0).astype(float) - 1.0) @ G)
    drift = -gamma * omega * (1.0 + np.sum(omega**2, axis=1, keepdims=True)) + beta * fb
    return omega + drift * dt + sigma_dyn * np.sqrt(dt) * rng.standard_normal((N, d))

betas = np.linspace(beta_lo, beta_hi, n_frames)
eigs = np.zeros((n_frames, d))
Ghat = G / np.linalg.norm(G, axis=1, keepdims=True)
n_cloud = 400
cloud_idx = rng.choice(N, size=n_cloud, replace=False)
z_cloud = np.zeros((n_frames, n_cloud, 2))              # top two axes, for optional use

omega = 0.05 * rng.standard_normal((N, d))
for i, beta in enumerate(betas):
    for _ in range(steps_per_frame):
        omega = step_dynamics(omega, beta, rng)
    C = (omega.T @ omega) / N                           # instantaneous, not averaged
    eigs[i] = np.sort(np.linalg.eigvalsh(C))[::-1]
    z = omega[cloud_idx] @ Ghat.T
    z_cloud[i] = z[:, :2]
    if i % 100 == 0:
        print(f"frame {i:4d}  beta={beta:6.2f}  N_active={(eigs[i] > 2*floor).sum():2d}")

np.savez(DATA_PATH, betas=betas, eigs=eigs, beta_c=np.sort(beta_c),
         u=np.sort(u)[::-1], gamma=gamma, sigma_dyn=sigma_dyn, sigma_obs=sigma_obs,
         rho_pair=rho_pair, floor=floor, z_cloud=z_cloud)
print("done ->", DATA_PATH)
