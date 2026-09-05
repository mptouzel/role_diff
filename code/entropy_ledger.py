"""Entropy and information ledger of Appendix F, on one active axis.

Reduces the identity dynamics at fixed schemas to the top schema axis
y = w.g1_hat, where the drift is the gradient of the 1-D section of
main-text Eq. (SM-eq:sm-energy) and the stationary density is Boltzmann,
p*(y) ~ exp(-2E(y)/sigma_idn^2).  For a grid of loop gains Lambda_1 it
prints the four quantities the appendix quotes:

  h(p*)               differential entropy of the identity marginal, in bits
  J                   negentropy against a Gaussian of the same variance
  2 E_b/sigma_idn^2   barrier height in units of the identity temperature
  I(S;D) - I(a;D)     information the sign policy discards per encounter

with D = (1-rho_pair) |g1| y the status difference, S = D + xi the
observation through the channel of noise sigma_read, and a = sign(S).
I(a;D) is the Szilard I of Eq. (3) averaged over the population.

Parameters are those of Fig. 1 (Appendix B).  This script generates no
figure; it is the source of the numbers quoted in Appendix F.
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid

GAMMA, S_READ, S_IDN, RHO_PAIR = 1.0, 0.6, 0.25, 0.0   # Fig. 1 parameters
G1 = 0.533                                             # |g_1|, strongest schema
GAIN = np.sqrt(2 / np.pi) / S_READ                     # eta = 2 pibar'(0)
BETA_C = GAMMA / ((1 - RHO_PAIR) * GAIN * G1**2)

NGRID, SPAN = 24001, 8.0
NCONV = 4001                   # subsampled grid for the O(n^2) convolution


def _h(p, x):                  # differential entropy in bits
    return -np.trapz(p * np.log2(np.clip(p, 1e-300, None)), x)


def _hbin(p):
    p = np.clip(p, 1e-300, 1 - 1e-300)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def ledger(lam):
    """Return the ledger entries at loop gain Lambda_1 = lam."""
    y = np.linspace(-SPAN, SPAN, NGRID)
    beta = lam * BETA_C
    psi = cumulative_trapezoid(
        G1 * (2 * norm.cdf((1 - RHO_PAIR) * G1 * y / S_READ) - 1), y, initial=0.0)
    E = GAMMA * (y**2 / 2 + y**4 / 4) - beta * psi
    p = np.exp(-2 * (E - E.min()) / S_IDN**2)
    p /= np.trapz(p, y)

    var = np.trapz(p * y**2, y) - np.trapz(p * y, y)**2
    h = _h(p, y)
    j = 0.5 * np.log2(2 * np.pi * np.e * var) - h
    i0 = np.argmin(np.abs(y))
    barrier = 2 * (E[i0] - E[i0 + np.argmax(p[i0:])]) / S_IDN**2

    # status difference, observation, and the loss at the sign policy
    d = (1 - RHO_PAIR) * G1 * y
    pd = p / ((1 - RHO_PAIR) * G1)
    step = max(1, NGRID // NCONV)
    dc, pc = d[::step], pd[::step]
    pc = pc / np.trapz(pc, dc)
    ps = norm.pdf((dc[:, None] - dc[None, :]) / S_READ) / S_READ @ pc * (dc[1] - dc[0])
    ps /= np.trapz(ps, dc)
    i_obs = _h(ps, dc) - 0.5 * np.log2(2 * np.pi * np.e * S_READ**2)
    i_act = 1.0 - np.trapz(pd * _hbin(norm.cdf(-np.abs(d) / S_READ)), d)
    return var, h, j, barrier, i_obs, i_act


if __name__ == "__main__":
    print(f"beta_c = {BETA_C:.3f}\n")
    print(f"{'Lam_1':>7}{'var':>9}{'h(p*)':>9}{'J':>8}{'2E_b/s^2':>10}"
          f"{'I(S;D)':>9}{'I(a;D)':>9}{'loss':>9}")
    for lam in [0.5, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 30.0]:
        var, h, j, b, i_obs, i_act = ledger(lam)
        print(f"{lam:>7.1f}{var:>9.3f}{h:>9.3f}{j:>8.3f}{b:>10.1f}"
              f"{i_obs:>9.3f}{i_act:>9.3f}{i_obs - i_act:>9.4f}")
