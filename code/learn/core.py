"""Learnable role-following policy on top of the sim_fig1.py dynamics.

Policy per context m:  P(Go | s) = 1/2 (1 + tanh(k_m (s - theta_m)))
k = 0 -> 50/50 play (the game's mixed Nash); k -> inf -> sign rule; k < 0 -> anti-follow.

Conventions copied from code/sim_fig1.py (not imported, so that file is untouched):
N=2500, d=M=3, gamma=1, sigma_obs (=sigma_read)=0.6, sigma_dyn (=sigma_omega)=0.25,
dt=0.02, all M contexts played every step against one random partner,
feedback beta * sum_m a_m g_m, quartic restoring force.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

N, d, M = 2500, 3, 3
GAMMA = 1.0
SIGMA_READ = 0.6
SIGMA_OM = 0.25
DT = 0.02
FLOOR = SIGMA_OM**2 / (2 * GAMMA)

# payoff[a_i, a_j], index 1 = Go, 0 = Defer: (G,G)=-1, (G,D)=6, (D,G)=2, (D,D)=3
PAY = np.array([[3.0, 2.0], [6.0, -1.0]])
NASH = 2.5

GH_X, GH_W = hermegauss(40)
GH_W = GH_W / GH_W.sum()


def schema_matrix(seed=5):
    """Same fixed hierarchical repertoire as sim_fig1.py (same rng draw order)."""
    rng = np.random.default_rng(seed)
    base = np.eye(d)
    dirs = np.array([base[m] + 0.12 * rng.standard_normal(d) for m in range(M)])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    strength = np.array([0.533, 0.434, 0.302])
    return dirs * strength[:, None], dirs, strength**2


G, GHAT, U = schema_matrix()


# ---------------------------------------------------------------- theory
def g_eff(k, sigma=SIGMA_READ):
    """Slope at Delta=0 of E_xi[tanh(k(Delta + sigma xi))]."""
    # substitute y = k sigma xi:  g_eff = sign(k)/sigma * int sech^2(y) phi(y/(|k| sigma)) dy
    # (accurate for all k; Gauss-Hermite in xi fails once sech^2 is narrower than the nodes)
    k = np.asarray(k, float)
    y = np.linspace(-25, 25, 20001)
    w = (1 / np.cosh(y) ** 2) * (y[1] - y[0])
    ak = np.abs(k)[..., None] * sigma + 1e-300
    phi = np.exp(-0.5 * (y / ak) ** 2) / np.sqrt(2 * np.pi)
    return np.sign(k) / sigma * (w * phi).sum(-1)


G_SIGN = np.sqrt(2 / np.pi) / SIGMA_READ  # g_eff(inf) = sign rule read gain


def Lambda_m(k, beta, u=U, rho=0.0, gamma=GAMMA):
    return beta * (1 - rho) * np.multiply.outer(g_eff(k), u) / gamma


def k_c(beta, u=U, rho=0.0, gamma=GAMMA):
    """k at which Lambda_m(k)=1, nan if unreachable (g_eff saturates at G_SIGN)."""
    from scipy.optimize import brentq
    out = []
    for um in np.atleast_1d(u):
        target = gamma / (beta * (1 - rho) * um)
        out.append(np.nan if target >= G_SIGN else brentq(lambda k: g_eff(k) - target, 1e-9, 1e4))
    return np.array(out)


def v_lin(k, beta):
    lam = Lambda_m(k, beta)
    return np.where(lam < 1, FLOOR / (1 - lam), np.nan)


# ---------------------------------------------------------------- dynamics
def pairing(rng):
    perm = rng.permutation(N)
    partner = np.empty(N, dtype=int)
    partner[perm[::2]] = perm[1::2]
    partner[perm[1::2]] = perm[::2]
    return partner


def step(omega, k, beta, rng, theta=None, mode="tanh", n=None):
    """One encounter round. k, theta: arrays of length M (shared policy).
    Returns new omega and info dict (s, a in {-1,+1}, t, r, Delta, partner)."""
    n = omega.shape[0]
    theta = np.zeros(M) if theta is None else theta
    perm = rng.permutation(n)
    partner = np.empty(n, dtype=int)
    partner[perm[::2]] = perm[1::2]
    partner[perm[1::2]] = perm[::2]
    Delta = (omega - omega[partner]) @ G.T            # n x M
    s = Delta + SIGMA_READ * rng.standard_normal((n, M))
    if mode == "sign":
        go = s > 0
        t = np.sign(s)
    else:
        t = np.tanh(k[None, :] * (s - theta[None, :]))
        go = rng.random((n, M)) < 0.5 * (1 + t)
    a = 2.0 * go - 1.0
    gi = go.astype(int)
    r = PAY[gi, gi[partner]]
    drift = -GAMMA * omega * (1.0 + np.sum(omega**2, axis=1, keepdims=True)) + beta * (a @ G)
    omega = omega + drift * DT + SIGMA_OM * np.sqrt(DT) * rng.standard_normal(omega.shape)
    return omega, dict(s=s, a=a, t=t, r=r, Delta=Delta, partner=partner)


def pg_grad(info, k, theta):
    """Myopic per-agent REINFORCE estimate of d E[r_i]/d(k_m, theta_m), batch-mean baseline."""
    adv = info["r"] - info["r"].mean(0, keepdims=True)
    da = info["a"] - info["t"]
    gk = (adv * (info["s"] - theta) * da).mean(0)
    gth = (adv * (-k) * da).mean(0)
    return gk, gth


def analytic_grad(info, k, theta):
    """Exact expectation over read noise of d E[r_i]/dk_i (partner's k fixed), averaged over agents."""
    D = info["Delta"][..., None]                       # n x M x H
    x = SIGMA_READ * GH_X
    kk, th = k[None, :, None], theta[None, :, None]
    zi = D - th + x
    zj = -D - th + x
    q = 0.5 * (1 + (GH_W * np.tanh(kk * zj)).sum(-1))  # partner Go prob
    dp = 0.5 * (GH_W * zi / np.cosh(kk * zi) ** 2).sum(-1)
    return (dp * (3 - 6 * q)).mean(0)


def proj_var(omega):
    C = omega.T @ omega / omega.shape[0]
    return np.einsum("md,de,me->m", GHAT, C, GHAT), np.sort(np.linalg.eigvalsh(C))[::-1]
