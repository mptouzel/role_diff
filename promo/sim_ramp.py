"""Ramped sweep for the promotional animation (not part of the submission).

Fig. 1 equilibrates the population at each beta. Here beta instead rises
linearly in time, beta(t), as in Sec. V of the main text, and the INSTANTANEOUS
covariance is recorded every frame. That is what makes the eigenvalues visibly
fluctuate and then lift off the noise floor one at a time.

The dynamics is the one of code/sim_fig1.py: tanh statuses, harmonic restoring
force, read noise averaged over K random partners per context (the
dense-encounter limit). Only the schedule (a ramp rather than per-beta
equilibration) and the repertoire size differ:
M = d = 16 modes on a geometric strength profile, so the thresholds
beta_c^(k) = gamma / ((1-rho_pair) g mu_k) are spaced logarithmically and the
cascade is the geometric class (ii) of Sec. III B. Under an exponential ramp
that puts the liftoffs at even intervals in frame number. The directions are
exactly orthogonal coordinate axes, so each ignition is a division along a new,
independent role axis.

Measured on this run (bimodality coefficient of the status on axes 1-3): axis 1
is split by beta = 5 (0.96), axes 2 and 3 by beta = 20 (0.99, 0.98), and all
sixteen are above the floor by beta = 40. Each axis divides the population
independently, so the joint of axes 1 and 2 ends in four groups.

Writes promo/ramp_data.npz.
"""
import os
import numpy as np
from scipy.special import ndtr, erf          # standard normal CDF

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
dt = 0.01                       # dense-encounter limit: no shot noise to resolve
K = 32                          # partners averaged per context per step
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
# One well-separated leading schema, then fifteen geometric from 7.2 to 60: one
# onset, a quiescent interval, then a burst, which is the spiked class (v) of
# Sec. III B. The top of the ramp (beta = 100) leaves the last mode only just
# above its own threshold, so an axis is still igniting on screen at the end.
#
# The wide first gap (u_2/u_1 = 0.25) gives panel (a) a stretch with one clean
# binary division before axis 2 divides each lobe again.
beta_c_target = np.concatenate([[1.8], np.geomspace(7.2, 60.0, M - 1)])
u = gamma / ((1.0 - rho_pair) * gain * beta_c_target)   # u_k = |g_k|^2
strength = np.sqrt(u)
G = dirs * strength[:, None]
mu = np.sort(np.linalg.eigvalsh(G @ G.T))[::-1]         # eigenvalues of H^T H
beta_c = gamma / ((1.0 - rho_pair) * gain * mu)         # Lambda_k = Lambda mu_k = 1

# --- ramp schedule ---------------------------------------------------------
# beta stops at 100, past the last threshold (60). The ramp is exponential so that a constant-speed marker on a log-beta axis
# still reads as a clock, and because strengths compound exponentially anyway.
beta_lo, beta_hi = 1.0, 100.0
n_frames = 900
steps_per_frame = 48                                    # 0.48 time units per frame
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
    score = np.tanh(omega @ G.T)                         # (N, M) statuses
    partner = rng.integers(0, N, size=(N, K, M))         # K partners per context
    gaps = score[:, None, :] - score[partner, np.arange(M)]   # status differences
    a = erf(gaps / (np.sqrt(2) * sigma_obs)).mean(1)     # noise-averaged sign decision
    drift = -gamma * omega + beta * (a @ G)
    return omega + drift * dt + sigma_dyn * np.sqrt(dt) * rng.standard_normal((N, d))

betas = np.geomspace(beta_lo, beta_hi, n_frames)
eigs = np.zeros((n_frames, d))
Ghat = G / np.linalg.norm(G, axis=1, keepdims=True)
n_cloud = N                                             # every agent, so density reads
cloud_idx = np.arange(N)
n_show = 3                                              # axes kept for the unfolding panel
z_cloud = np.zeros((n_frames, n_cloud, n_show), dtype=np.float32)
s_cloud = np.zeros((n_frames, n_cloud, n_show), dtype=np.float32)   # statuses tanh(omega.g)

# What the cascade is worth, measured on the encounters themselves. For a pair
# meeting in context m the role-inference error is P_e = Phi(-|Delta|/sread),
# so the encounter yields I = 1 - H_bin(P_e) bits of role information and a
# coordination payoff W/w0 = (1-2P_e)^2 above Nash. Both go to 1 per context as
# the roles sharpen, so summed over contexts each saturates at N: the same axis
# as the cascade staircase, and the gap between them is the axes that have
# activated without yet separating cleanly.
bits = np.zeros(n_frames)
payoff = np.zeros(n_frames)

def _hbin(p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def extracted(omega, rng):
    score = np.tanh(omega @ G.T)
    partner = rng.integers(0, N, size=(N, M))
    gaps = score - score[partner, np.arange(M)]
    pe = ndtr(-np.abs(gaps) / sigma_obs)
    return (1.0 - _hbin(pe)).mean(axis=0).sum(), ((1.0 - 2.0 * pe) ** 2).mean(axis=0).sum()

omega = 0.05 * rng.standard_normal((N, d))
for i, beta in enumerate(betas):
    for _ in range(steps_per_frame):
        omega = step_dynamics(omega, beta, rng)
    C = (omega.T @ omega) / N                           # instantaneous, not averaged
    eigs[i] = np.sort(np.linalg.eigvalsh(C))[::-1]
    z = omega[cloud_idx] @ Ghat.T
    z_cloud[i] = z[:, :n_show]
    s_cloud[i] = np.tanh(omega[cloud_idx] @ G.T)[:, :n_show]
    bits[i], payoff[i] = extracted(omega, rng)
    if i % 100 == 0:
        print(f"frame {i:4d}  beta={beta:6.2f}  N_active={(eigs[i] > 2*floor).sum():2d}")

np.savez(DATA_PATH, betas=betas, eigs=eigs, beta_c=np.sort(beta_c),
         u=np.sort(u)[::-1], mu=mu, gamma=gamma, sigma_dyn=sigma_dyn, sigma_obs=sigma_obs,
         rho_pair=rho_pair, floor=floor, z_cloud=z_cloud, s_cloud=s_cloud, bits=bits, payoff=payoff)
print("done ->", DATA_PATH)
