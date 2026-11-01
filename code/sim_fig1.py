"""Quasi-static beta sweep of the role cascade (Fig. 1b,c).

For each FIXED beta we equilibrate the agent dynamics and average the order
parameter -- a stationary/adiabatic object, so the x-axis is unambiguously beta
(NOT time). The schema repertoire is held fixed: orthogonal axes with a
hierarchical strength profile (Model 4), so the covariance eigenvalues
lambda_k(beta) rise from the noise floor one at a time at the critical couplings
beta_c^(k) = gamma/((1-rho_pair) mu_k), mu_k = 2 pibar'(0) u_k with u_k the
eigenvalues of H^T H, beta_c^(k) ~ [2.66, 4.01, 8.26]. Model 6 (dynamic schemas)
selects this same ordering endogenously via resource feedback (main-text Fig. 3);
here we fix the schemas to display the cascade cleanly.

Model of main-text Eq. (1): status difference = difference of saturating scores,
tanh(w_i.g_m) - tanh(w_j.g_m); harmonic restoring force -gamma w_i. Dense-encounter
limit: each step every agent meets K random partners in every context and its
single-sample sign decision is replaced by its average over the read noise,
erf(Delta / (sqrt 2 sigma_obs)), averaged over the K partners. The write shot
noise of a finite encounter rate is therefore absent, and the only noise is the
isotropic omega-identity noise sigma_dyn, giving the flat floor sigma_dyn^2/2gamma.
"""
import os
import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)                  # .../culture_engine_PRX
DATA_PATH = os.path.join(PKG_ROOT, "data", "fig1_data.npz")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

rng = np.random.default_rng(5)

N, d, M = 2500, 3, 3
gamma = 1.0
sigma_obs = 0.6
sigma_dyn = 0.25
dt = 0.02
K = 32           # partners per agent per step (dense-encounter average)
rho_pair = 0.0   # random pairing: partner correlation (factor (1-rho_pair)=1)

betas = np.linspace(0.1, 21.0, 120)
burn_steps = 1200
avg_steps = 1500

# Fixed hierarchical context portfolio: orthogonal directions with spaced
# strengths |g_m|. The strengths are pinned by the target critical couplings via
# u_k ~ 1/beta_c^(k), beta_c^(k) = gamma sigma_obs sqrt(2pi)/(2 (1-rho_pair) u_k).
dirs = np.eye(d)
strength = np.array([0.533, 0.434, 0.302]) * np.sqrt(sigma_obs / 0.6)   # |g_m|
G = dirs * strength[:, None]                      # fixed throughout

piprime = 1.0 / (sigma_obs * np.sqrt(2 * np.pi))
u_k = np.sort(np.linalg.eigvalsh(G.T @ G))[::-1]  # eigenvalues of H^T H, strongest first
mu = 2 * piprime * u_k
beta_c = gamma / ((1.0 - rho_pair) * mu)   # gamma/((1-rho_pair) mu_k); random -> gamma/mu_k

def step_dynamics(omega, beta, rng):
    score = np.tanh(omega @ G.T)                  # N x M saturating scores
    partners = rng.integers(0, N, (N, K))         # random matching, rho_pair = 0
    gaps = score[:, None, :] - score[partners]    # status differences Delta_im(j)
    a = erf(gaps / (np.sqrt(2) * sigma_obs)).mean(1)   # noise-averaged sign decision
    drift = -gamma * omega + beta * (a @ G)
    return omega + drift * dt + sigma_dyn * np.sqrt(dt) * rng.standard_normal((N, d))

eigs_C = np.zeros((len(betas), d))
Ghat = G / np.linalg.norm(G, axis=1, keepdims=True)

# panel-b scatter: instantaneous z-snapshots of a large agent cloud at a coarse
# beta grid (time-averaging would wash out the bimodality, so we snapshot).
n_snap_beta = 26
snap_ib = np.linspace(0, len(betas) - 1, n_snap_beta).astype(int)
n_cloud = 260
cloud_idx = rng.choice(N, size=n_cloud, replace=False)
snap_betas = betas[snap_ib]
z_snap = np.full((n_snap_beta, n_cloud, M), np.nan)
si = 0

omega = 0.05 * rng.standard_normal((N, d))       # adiabatic warm start
for ib, beta in enumerate(betas):
    for _ in range(burn_steps):
        omega = step_dynamics(omega, beta, rng)
    accC = np.zeros((d, d))
    for _ in range(avg_steps):
        omega = step_dynamics(omega, beta, rng)
        accC += (omega.T @ omega) / N
    eigs_C[ib] = np.sort(np.linalg.eigvalsh(accC / avg_steps))[::-1]
    if ib in snap_ib:                              # instantaneous cloud snapshot
        z_snap[si] = omega[cloud_idx] @ Ghat.T
        si += 1
    if ib % 18 == 0:
        print(f"beta={beta:.2f}  eigs={eigs_C[ib].round(3)}")

# schema strengths are fixed; report them (constant) for the dashed overlay
g_norms = np.tile(u_k, (len(betas), 1))

np.savez(DATA_PATH,
         betas=betas, eigs=eigs_C, gnorm=g_norms,
         z_snap=z_snap, snap_betas=snap_betas,
         gamma=gamma, sigma_dyn=sigma_dyn, beta_c=beta_c, rho_pair=rho_pair, mu=mu)
print("done. beta_c^(k) =", beta_c.round(3))
print("final eigs:", eigs_C[-1].round(3))
# report onset betas
floor = sigma_dyn**2 / (2 * gamma)
for k in range(3):
    idx = np.argmax(eigs_C[:, k] > 3 * floor)
    hit = eigs_C[idx, k] > 3 * floor
    print(f"lam{k+1} onset (3x floor): beta={betas[idx]:.2f}" if hit else f"lam{k+1}: no onset")
