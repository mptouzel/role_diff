"""Three-dimensional geometry of one schema (not part of the submission).

The other promo runs show the cascade as a spectrum. This one shows the objects
the cascade acts on: the cloud of omega-identities, the schema that scores them,
and the split of the cloud into two role lobes along the axis that schema reads.

Repertoire. d = M = 3, exactly orthogonal, with the strength profile of Fig. 1
(code/sim_fig1.py, |g_m| = 0.533, 0.434, 0.302), so the clip traces the same
cascade the figure does: beta_c = 2.6, 4.0, 8.2. Directions are exactly
orthogonal rather than the 0.12-jittered ones of sim_fig1.py, so each axis is a
coordinate of the box and an ignition reads as a division along a drawn arrow.

What the run is for. Measured at N = 3000 on this repertoire, the axis-1
marginal dip (density at the origin over peak density) falls 0.96 -> 0.30 ->
0.04 at beta = 3, 6, 12, while the axis-2 dip is still 0.97 at beta = 12 and
0.55 at beta = 20, and the angular contrast of the (z_1, z_2) density stays at
34-43 throughout. The population therefore holds TWO lobes, not four groups and
not the ring of sim_ring.py: axis 2 crosses its threshold at beta_c = 4.0 and
broadens the cloud long before it divides it. The animation is written to say
that rather than to imply a second role.

Saves a subcritical snapshot for the staged encounter of the clip's second act,
then the ramp frames.

Writes promo/geometry_data.npz.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "geometry_data.npz")

rng = np.random.default_rng(17)

N, d, M = 600, 3, 3          # N small: individual points must stay readable in 3D
gamma = 1.0
sigma_obs = 0.6
sigma_dyn = 0.25
dt = 0.002
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
steps_per_frame = 200
sub_steps = 20000


def step(omega, beta, rng):
    """One encounter per context per agent, partner redrawn per context.

    Identical in form to promo/sim_ramp.py: the independent partner draw is what
    keeps role assignments from correlating across axes.
    """
    proj = omega @ G.T
    partner = rng.integers(0, N, size=(N, M))
    gaps = proj - proj[partner, np.arange(M)]
    S = gaps + sigma_obs * rng.standard_normal((N, M))
    fb = (2.0 * (S > 0).astype(float) - 1.0) @ G
    drift = -gamma * omega * (1.0 + np.sum(omega**2, axis=1, keepdims=True)) + beta * fb
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
