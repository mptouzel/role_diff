"""
Figures for Sec. 'The functional form of the bifurcation cascade'.

Fig 1 (fig:cascade-classes): (a) counting function N(beta)/R for the six
spectral classes; (b) bifurcation rate dN/dbeta on log-log axes with slope
refs; (c) the rank-ordered gain spectra those two transform.
Fig 2 (fig:regime-diagram): schematic phase diagram of endogenous regimes
E1-E4 in the (nu/g, condensation-regulation) plane, with miniature cascade
glyphs in each region.
Fig 3 (fig:monitor-view): the monitor's view -- N(t) under a linear Lambda
ramping of the loop gain, showing why early observation windows are non-identifying.

All sweeps in the loop gain Lambda (spectra normalized to mu_1 = 1, so
bifurcation points b_k = 1/mu_k).
"""

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(os.path.dirname(_HERE), "figures")
def _out(name):
    return os.path.join(_FIG_DIR, name)
import numpy as np
import matplotlib as mpl
_MPL_PINNED = "3.8.4"          # produced the committed figures (see requirements.txt)
if mpl.__version__ != _MPL_PINNED:
    print(f"warning: matplotlib {mpl.__version__} != {_MPL_PINNED}; text rendering will differ")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Spectra (normalized: mu_1 = 1). R modes each.
# ----------------------------------------------------------------------
R = 50   # modes plotted per class; M >= R, and M = 2R for the q=2 Marchenko-Pastur case
k = np.arange(1, R + 1)

k0 = 8.0
spec = {}
spec["degenerate"] = np.ones(R)
spec["geometric"] = np.exp(-(k - 1) / k0)
spec["zipf_steep"] = k ** (-2.0)      # a = 2  (> 1, decelerating)
spec["zipf_shallow"] = k ** (-0.5)    # a = 1/2 (< 1, accelerating)

# Marchenko-Pastur: P^T P with M = 2 d  (y = d/M = 1/2), normalized top edge
y_mp = 0.5
lam_minus, lam_plus = (1 - np.sqrt(y_mp)) ** 2, (1 + np.sqrt(y_mp)) ** 2
d_mp, M_mp = 400, 800
P = rng.normal(size=(M_mp, d_mp)) / np.sqrt(M_mp)
mp_eigs_full = np.sort(np.linalg.eigvalsh(P.T @ P))[::-1]
mp_eigs_full /= mp_eigs_full[0]
# subsample R of them (quantiles) for the step curve
qs = (np.arange(R) + 0.5) / R
spec["mp"] = np.quantile(mp_eigs_full, 1 - qs)

# Spiked: one coherent outlier + narrow bulk (uniform coherence example)
chi, Msp = 0.20, R
mu_sp = 1.0
mu_bulk = (1 - chi) / (1 + (Msp - 1) * chi)      # ~ 0.0784 -> gap ratio ~ 12.7
bulk = mu_bulk * (1 + 0.05 * rng.normal(size=R - 1))
spec["spiked"] = np.sort(np.concatenate([[mu_sp], bulk]))[::-1]

styles = {
    "degenerate":   dict(color="0.35",       label=r"(i) single avalanche"),
    "geometric":    dict(color="tab:blue",   label=r"(ii) logarithmic ($k_0{=}8$)"),
    "zipf_steep":   dict(color="tab:green",  label=r"(iii) decelerating ($a{=}2$)"),
    "zipf_shallow": dict(color="tab:red",    label=r"(iii) accelerating ($a{=}\frac{1}{2}$)"),
    "mp":           dict(color="tab:orange", label=r"(iv) quiet, then burst ($q{=}2$)"),
    "spiked":       dict(color="tab:purple", label=r"(v) one, gap, then all ($\chi{=}0.2$)"),
}


def counting_curve(mu):
    """Empirical N(beta)/R as a step function; returns (beta_grid, N/R)."""
    b = np.sort(1.0 / mu)
    bg = np.concatenate([[0.7], np.repeat(b, 2), [1e4]])
    Ng = np.concatenate([[0.0], np.repeat(np.arange(1, len(b) + 1), 2)[:-1] / len(b), [1.0], [1.0]])
    # build proper step: before b1 -> 0; after b_k -> k/R
    bg, Ng = [0.5], [0.0]
    for i, bb in enumerate(b):
        bg += [bb, bb]
        Ng += [i / len(b), (i + 1) / len(b)]
    bg.append(1e4)
    Ng.append(1.0)
    return np.array(bg), np.array(Ng)


# ----------------------------------------------------------------------
# Figure 1: gain spectra + counting functions + log-log rates
# ----------------------------------------------------------------------
fig, (axA, axB, axS) = plt.subplots(1, 3, figsize=(8.8, 2.9))

ORDER = ["degenerate", "geometric", "zipf_steep", "zipf_shallow", "mp", "spiked"]
BMAX = 400

# ---- panel (a): the rank-ordered spectra the other two panels transform
for name in ORDER:
    mu = np.sort(spec[name])[::-1]
    axS.plot(k, mu / mu[0], lw=1.6, **styles[name])

axS.set_xscale("log")
axS.set_yscale("log")
axS.set_xlim(0.9, R * 1.15)
axS.set_ylim(3e-4, 2.2)
axS.set_xlabel(r"mode index $k$")
axS.set_ylabel(r"$\mu_k/\mu_1$")
axS.set_title(r"(c)  gain spectrum")

for name in ORDER:
    bg, Ng = counting_curve(spec[name])
    axA.plot(bg, Ng, lw=1.8, **styles[name])

axA.set_xscale("log")
axA.set_xlim(0.7, BMAX)
axA.set_ylim(-0.02, 1.05)
axA.set_xlabel(r"loop gain  $\Lambda(t)$")
axA.set_ylabel(r"activated fraction  $N(t)/R$")
axA.axvline(1.0, color="0.8", lw=0.8, zorder=0)
axA.set_title(r"(a)  bifurcation count")

# no in-panel callouts: at this width they collide with the curves, and the legend
# plus the caption already name every class and its signature feature.

# ---- panel (b): rates dN/dbeta (per mode), log-log, analytic where possible
b = np.logspace(np.log10(0.75), np.log10(BMAX), 800)

# geometric: dN/db = k0 / b  on [1, e^{(R-1)/k0}]
mask = (b >= 1) & (b <= np.exp((R - 1) / k0))
axB.plot(b[mask], (k0 / b[mask]) / R, color="tab:blue", lw=1.8)

# Zipf: dN/db = (1/a) b^{(1-a)/a}
for a, c in [(2.0, "tab:green"), (0.5, "tab:red")]:
    mask = (b >= 1) & (b <= R ** a)
    axB.plot(b[mask], ((1 / a) * b[mask] ** ((1 - a) / a)) / R, color=c, lw=1.8)

# MP: dN/db = rho_norm(1/b) / b^2 ; rho_norm(mu) = lam_plus * rho_MP(lam_plus*mu)
def rho_mp(lam):
    out = np.zeros_like(lam)
    m = (lam > lam_minus) & (lam < lam_plus)
    out[m] = np.sqrt((lam_plus - lam[m]) * (lam[m] - lam_minus)) / (2 * np.pi * y_mp * lam[m])
    return out

mu_b = 1.0 / b
rate_mp = lam_plus * rho_mp(lam_plus * mu_b) / b ** 2
axB.plot(b[rate_mp > 0], rate_mp[rate_mp > 0], color="tab:orange", lw=1.8)

# spiked bulk: Gaussian bulk in mu at mu_bulk, rel. width 5%
s_b = 0.05 * mu_bulk
rate_sp = ((R - 1) / R) * np.exp(-0.5 * ((mu_b - mu_bulk) / s_b) ** 2) / (np.sqrt(2 * np.pi) * s_b) / b ** 2
axB.plot(b, rate_sp, color="tab:purple", lw=1.8)

# deltas: degenerate at b=1, spike at b=1 -> arrows
for x0, c, dy in [(1.0, "0.35", 1.0)]:
    axB.annotate("", xy=(x0, 2.0), xytext=(x0, 0.15),
                 arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
axB.text(1.12, 1.05, r"$\delta$ (avalanche)", fontsize=8, color="0.35")

# slope guides
def slope_guide(x0, x1, y0, p, txt, dx=1.0, dy=1.4):
    xs = np.array([x0, x1])
    axB.plot(xs, y0 * (xs / x0) ** p, color="0.6", lw=0.8, ls=":")
    axB.text(x1 * dx, y0 * (x1 / x0) ** p * dy, txt, fontsize=8, color="0.45")

slope_guide(6, 60, 0.0229, -1.0, r"slope $-1$", dx=1.14, dy=1.28)
slope_guide(6, 60, 0.0035, -0.5, r"$-\frac{1}{2}$", dx=1.05, dy=0.74)
slope_guide(1.6, 4.5, 0.055, +1.0, r"$+1$", dx=1.05, dy=1.0)

axB.set_xscale("log")
axB.set_yscale("log")
axB.set_xlim(0.75, BMAX)
axB.set_ylim(3e-4, 3)
axB.set_xlabel(r"loop gain  $\Lambda(t)$")
axB.set_ylabel(r"bifurcation rate  $R^{-1}\,dN/d\Lambda$")
axB.set_title(r"(b)  bifurcation rate")

for ax in (axA, axB, axS):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# no legend: the class table in the manuscript float carries the colour key
fig.tight_layout()
fig.savefig(_out("fig_cascade_classes.pdf"))
fig.savefig(_out("fig_cascade_classes.png"), dpi=300)
plt.close(fig)

print("done")
