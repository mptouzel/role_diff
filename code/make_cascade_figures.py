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
axA.set_xlabel(r"loop gain  $\Lambda$")
axA.set_ylabel(r"activated fraction  $N(\beta)/R$")
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
axB.set_xlabel(r"loop gain  $\Lambda$")
axB.set_ylabel(r"bifurcation rate  $R^{-1}\,dN/d\beta$")
axB.set_title(r"(b)  bifurcation rate")

for ax in (axA, axB, axS):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# no legend: the class table in the manuscript float carries the colour key
fig.tight_layout()
fig.savefig(_out("fig_cascade_classes.pdf"))
fig.savefig(_out("fig_cascade_classes.png"), dpi=300)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: endogenous regime phase diagram (schematic)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.set_xscale("log")
ax.set_xlim(0.08, 12)
ax.set_ylim(0, 1)

bands = {  # (y0, y1)
    "E4": (0.00, 0.26),
    "E3": (0.26, 0.58),
    "E2": (0.58, 0.80),
    "E1": (0.80, 1.00),
}
col = {"E4": "#e6d4f0", "E3L": "#d9ecd9", "E3R": "#f5d2cc", "E2": "#d4e4f5", "E1": "#e3e3e3"}

def rect(x0, x1, y0, y1, c):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transData,
                           facecolor=c, edgecolor="none", zorder=0))

rect(0.08, 12, *bands["E4"], col["E4"])
rect(0.08, 1.0, *bands["E3"], col["E3L"])
rect(1.0, 12, *bands["E3"], col["E3R"])
rect(0.08, 12, *bands["E2"], col["E2"])
rect(0.08, 12, *bands["E1"], col["E1"])
for y in [0.26, 0.58, 0.80]:
    ax.axhline(y, color="w", lw=2.0)
    ax.axhline(y, color="0.55", lw=0.8)

# a = 1 boundary inside E3
ax.plot([1, 1], list(bands["E3"]), color="0.2", lw=1.2, ls="--")
ax.text(1.0, 0.585, r"$a=1$ (Zipf boundary)", ha="center", va="bottom", fontsize=8.5)

# labels
ax.text(0.095, 0.90, "E1  hard saturation / per-schema caps\n"
        r"$\Rightarrow$ (near-)degenerate spectrum", va="center", fontsize=9)
ax.text(3.0, 0.90, "single avalanche\n(no warning)", va="center", ha="center",
        fontsize=9, color="#8a1f1f")
ax.text(0.095, 0.69, "E2  budget slack, slow resources ($c\\ll\\gamma$),\n"
        r"age-ordered compounding $\Rightarrow$ geometric, $k_0=\nu/g$",
        va="center", fontsize=9)
ax.text(3.0, 0.69, "logarithmic cascade\n(constant per e-fold;\nmost monitorable)",
        va="center", ha="center", fontsize=9, color="#1f4f8a")
ax.text(0.095, 0.475, "E3  binding bandwidth,\nshare-proportional\n"
        r"reinforcement $\Rightarrow$ Zipf, $a\sim g/\nu$", va="center", fontsize=9)
ax.text(0.5, 0.315, "entrenchment: $a>1$\ndecelerating, front-loaded",
        ha="center", va="center", fontsize=9, color="#1f6b1f")
ax.text(2.8, 0.315, "churn: $a<1$\naccelerating (deceptive)",
        ha="center", va="center", fontsize=9, color="#8a1f1f")
ax.text(0.095, 0.16, "E4  unregulated (autocatalytic) condensation;\n"
        r"fringe pinned at invasion margin $F=(\kappa/\alpha)u$"
        "\n" r"$\Rightarrow$ spiked spectrum", va="center", fontsize=9)
ax.text(3.0, 0.13, "spike, quiescent interval,\nclustered avalanche",
        va="center", ha="center", fontsize=9, color="#5a2d82")

# miniature N(beta) glyphs
def glyph(cx, cy, kind, color):
    w, h = 0.10, 0.085  # in axes fraction
    ia = ax.inset_axes([cx, cy, w, h], transform=ax.transAxes)
    x = np.linspace(0, 1, 200)
    if kind == "step":
        y_ = (x > 0.35).astype(float)
    elif kind == "log":
        y_ = np.log1p(20 * x) / np.log(21)
    elif kind == "concave":
        y_ = x ** 0.5
    elif kind == "convex":
        y_ = x ** 2.5
    elif kind == "spiked":
        y_ = 0.08 * (x > 0.12) + 0.92 * (x > 0.82)
    ia.plot(x, y_, color=color, lw=1.6)
    ia.set_xticks([]); ia.set_yticks([])
    for s in ia.spines.values():
        s.set_color("0.6"); s.set_linewidth(0.6)
    ia.set_ylim(-0.08, 1.1)
    ia.set_facecolor("white")

glyph(0.88, 0.855, "step", "0.25")
glyph(0.88, 0.615, "log", "tab:blue")
glyph(0.05, 0.28, "concave", "tab:green")
glyph(0.88, 0.30, "convex", "tab:red")
glyph(0.88, 0.045, "spiked", "tab:purple")

# axes labels / arrows
ax.set_xlabel(r"schema innovation / incumbent compounding,  $\nu/g$"
              "\n" r"(raised by: protocol churn, fast resource decay $c$;"
              r" lowered by: rate limits, inertia)")
ax.set_yticks([])
ax.annotate("", xy=(0.02, 0.99), xytext=(0.02, 0.01),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.1))
ax.text(-0.075, 0.5, "regulation of the condensation instability\n"
        "(weak / autocatalytic  $\\rightarrow$  strong / saturated)",
        transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=9.5)
ax.set_title("Endogenous cascade classes selected by the schema--resource dynamics (Model 6)",
             fontsize=10.5)

fig.tight_layout()
fig.savefig(_out("fig_regime_diagram.pdf"))
fig.savefig(_out("fig_regime_diagram.png"), dpi=300)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 3: the monitor's view (linear ramp in loop gain)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.0))

BLIN = 15
bg, Ng = counting_curve(spec["geometric"]); ax.plot(bg, Ng, lw=1.8, color="tab:blue",
                                                    label=r"geometric (extrapolable)")
bg, Ng = counting_curve(spec["zipf_shallow"]); ax.plot(bg, Ng, lw=1.8, color="tab:red",
                                                       label=r"Zipf $a{=}\frac{1}{2}$ (accelerating)")
bg, Ng = counting_curve(spec["mp"]); ax.plot(bg, Ng, lw=1.8, color="tab:orange",
                                             label="random MP (quiet start)")
bg, Ng = counting_curve(spec["spiked"]); ax.plot(bg, Ng, lw=1.8, color="tab:purple",
                                                 label="spiked (gap, then avalanche)")

ax.set_xlim(0, BLIN)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"loop gain  $\Lambda(t)$   (linear sweep, $\dot\Lambda$ const.)")
ax.set_ylabel(r"$N(t)/R$")

# early observation window
ax.axvspan(0, 3, color="0.92", zorder=0)
ax.text(1.5, 0.995, "early observation\nwindow: classes\nnearly indistinguishable",
        ha="center", va="top", fontsize=8.5, color="0.35")

ax.annotate("avalanche of pinned fringe", xy=(12.6, 0.50), xytext=(8.3, 0.30),
            fontsize=8.5, color="tab:purple",
            arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=0.9))
ax.annotate(r"quiescent interval $\simeq \frac{1+(M{-}1)\chi}{1-\chi}$",
            xy=(6.5, 0.045), xytext=(5.1, 0.20), fontsize=8.5, color="tab:purple",
            arrowprops=dict(arrowstyle="-", color="tab:purple", lw=0.8))
ax.annotate(r"$N \propto (\Lambda-1)^{3/2}$", xy=(2.1, 0.075), xytext=(2.8, 0.33),
            fontsize=8.5, color="tab:orange",
            arrowprops=dict(arrowstyle="-", color="tab:orange", lw=0.8))
ax.legend(loc="upper left", bbox_to_anchor=(0.42, 0.99), frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=8.5)
ax.set_title("The monitor's view under linear ramping of loop gain", fontsize=10.5)

fig.tight_layout()
fig.savefig(_out("fig_monitor_view.pdf"))
fig.savefig(_out("fig_monitor_view.png"), dpi=300)
plt.close(fig)

print("done")
