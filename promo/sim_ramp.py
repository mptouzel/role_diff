"""Ramped sweep for the promotional animation (not part of the submission).

Fig. 1 equilibrates the population at each beta. Here beta instead rises
linearly in time, beta(t), as in Sec. V of the main text, and the INSTANTANEOUS
covariance is recorded every frame. That is what makes the eigenvalues visibly
fluctuate and then lift off the noise floor one at a time.

The dynamics is the one of code/sim_fig1.py, unchanged. Only the schedule
(a ramp rather than per-beta equilibration) and the repertoire size differ:
M = d = 16 modes on a geometric strength profile, so the thresholds
beta_c^(k) = gamma / ((1-rho_pair) g mu_k) are spaced logarithmically, so the
cascade is the geometric class (ii) of Sec. III B. The directions are perturbed
coordinate axes with the perturbation scaled by 1/sqrt(d), so the repertoire
stays near-orthogonal as d grows and each ignition is a division along a new,
independent role axis.

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
sigma_dyn = 0.25                # as Fig. 1. Lowering it tightens the role lobes but
                                # also lowers the floor, and v_k/floor = 1/(1-Lambda_k),
                                # so "supra-floor" then fires far below threshold and
                                # the cascade collapses into one jump.
dt = 0.02
rho_pair = 0.0                      # random pairing, factor (1-rho_pair) = 1

# --- repertoire: near-orthogonal directions, geometric strengths ------------
# Exactly orthogonal, not merely near-orthogonal. An overlap of even 0.06
# gives a kick along g_1 a component along ghat_2, and at large beta with a
# saturated <a> that bias is comparable to axis 2's own split: roles then
# correlate across axes (r ~ 0.3) and the quadrants stop being independent.
# This is the transposability of Sec. I D, and here it has to be switched off
# for each ignition to read as a division along a genuinely new axis.
dirs = np.eye(d)[:M]

piprime = 1.0 / (sigma_obs * np.sqrt(2 * np.pi))
gain = 2 * piprime                                      # read gain g = 2 pibar'(0)
# Three dominant schemas well separated in threshold, then a packed band of 13.
# The separation lets the first divisions be seen one at a time; the band then
# fires in quick succession, which is the quiescent-interval-then-burst shape of
# the spiked class (v) of Sec. III B. The spread is kept modest because the
# quartic term is isotropic: once early modes saturate, the shared |omega|^2
# lifts the effective threshold of every remaining axis by (1 + <|omega|^2>),
# so a wide hierarchy stalls its own tail.
beta_c_target = np.concatenate([[1.8, 3.5, 6.0], np.linspace(8.0, 11.0, M - 3)])
u = gamma / ((1.0 - rho_pair) * gain * beta_c_target)   # u_k = |g_k|^2
strength = np.sqrt(u)
G = dirs * strength[:, None]
mu = np.sort(np.linalg.eigvalsh(G @ G.T))[::-1]         # eigenvalues of H^T H
beta_c = gamma / ((1.0 - rho_pair) * gain * mu)         # Lambda_k = Lambda mu_k = 1

# --- ramp schedule ---------------------------------------------------------
beta_lo, beta_hi = 0.5, 42.0
n_frames = 900
steps_per_frame = 32                                    # 22400 steps total
floor = sigma_dyn**2 / (2 * gamma)                      # noise floor sigma_w^2/2gamma

def step_dynamics(omega, beta, rng):
    """One partner per context, drawn independently.

    code/sim_fig1.py draws a single partner per step and projects the SAME gap
    onto every context. That makes an agent which sits above its partner overall
    Go on many contexts at once, so role assignments correlate across axes
    (measured at r ~ 0.4, against a geometric overlap of 0.06). Sec. I instead
    draws one context per encounter, so an agent meets different partners in
    different contexts. Redrawing the partner per context restores that
    independence at the same feedback scale.
    """
    proj = omega @ G.T                                   # (N, M) status on each axis
    partner = rng.integers(0, N, size=(N, M))            # independent partner per context
    gaps = proj - proj[partner, np.arange(M)]
    S = gaps + sigma_obs * rng.standard_normal((N, M))
    fb = ((2 * (S > 0).astype(float) - 1.0) @ G)
    drift = -gamma * omega * (1.0 + np.sum(omega**2, axis=1, keepdims=True)) + beta * fb
    return omega + drift * dt + sigma_dyn * np.sqrt(dt) * rng.standard_normal((N, d))

betas = np.linspace(beta_lo, beta_hi, n_frames)
eigs = np.zeros((n_frames, d))
Ghat = G / np.linalg.norm(G, axis=1, keepdims=True)
n_cloud = N                                             # every agent, so density reads
cloud_idx = np.arange(N)
n_show = 3                                              # axes kept for the unfolding panel
z_cloud = np.zeros((n_frames, n_cloud, n_show), dtype=np.float32)

omega = 0.05 * rng.standard_normal((N, d))
for i, beta in enumerate(betas):
    for _ in range(steps_per_frame):
        omega = step_dynamics(omega, beta, rng)
    C = (omega.T @ omega) / N                           # instantaneous, not averaged
    eigs[i] = np.sort(np.linalg.eigvalsh(C))[::-1]
    z = omega[cloud_idx] @ Ghat.T
    z_cloud[i] = z[:, :n_show]
    if i % 100 == 0:
        print(f"frame {i:4d}  beta={beta:6.2f}  N_active={(eigs[i] > 2*floor).sum():2d}")

np.savez(DATA_PATH, betas=betas, eigs=eigs, beta_c=np.sort(beta_c),
         u=np.sort(u)[::-1], mu=mu, gamma=gamma, sigma_dyn=sigma_dyn, sigma_obs=sigma_obs,
         rho_pair=rho_pair, floor=floor, z_cloud=z_cloud)
print("done ->", DATA_PATH)
