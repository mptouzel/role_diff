"""Supplemental Fig. S1 (learned policy), four panels. Reads data/learn/*.npz only;
the simulations are code/learn/exp_*.py (`make learn-data`).
   (a) small-slope growth rate of the myopic payoff gradient against 3 u_m v_m;
   (b) learned slopes through the cascade, with the fixed-slope covariance as inset;
   (c) orientation of the convention from ell_0 = 0, without and with the collision tilt;
   (d) edge of the anti-follow basin against the tilt."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_HERE, "learn"))
D = os.path.join(_ROOT, "data", "learn")
def _out(name):
    return os.path.join(_ROOT, "figures", name)
import numpy as np
import matplotlib as mpl
_MPL_PINNED = "3.8.4"          # produced the committed figures (see requirements.txt)
if mpl.__version__ != _MPL_PINNED:
    print(f"warning: matplotlib {mpl.__version__} != {_MPL_PINNED}; text rendering will differ")
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import core as c

plt.rcParams.update({'font.size': 8.5, 'axes.linewidth': 0.8, 'font.family': 'sans-serif',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'legend.fontsize': 7.5})
cols = ['#1f77b4', '#ff7f0e', '#2ca02c']   # contexts 1-3, as the axes of Fig. 1
MUTED = '0.55'
LMAX = 4.0                                  # 40-node quadrature is unreliable beyond |ell| = 4

f = np.load(os.path.join(D, "fixed_k.npz")); ks = f["ks"]; betas = list(f["betas"])
sw = np.load(os.path.join(D, "learn_sweep.npz"))
sg = np.load(os.path.join(D, "learn_sign.npz"))
tl = np.load(os.path.join(D, "tilt_learn_seed300.npz"))
tb = np.load(os.path.join(D, "tilt_basin.npz"))
bi = betas.index(6.0)

fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8))

# ---------- (a) small-ell growth rate: exact gradient / ell against 3 u_m v_m ----------
a = ax[0, 0]
sel = (np.abs(ks) > 0) & (np.abs(ks) <= 0.12)
mk = {2.0: 'o', 6.0: 's', 12.0: '^'}
for j, b in enumerate(betas):
    for m in range(3):
        x = 3 * c.U[m] * f["v"][j, sel, m]; y = f["gr"][j, sel, m] / ks[sel]
        a.plot(x, y, mk[b], ms=3.5, color=cols[m], alpha=0.85, mec='none')
lim = [0.005, 0.3]; a.plot(lim, lim, '-', lw=0.9, color=MUTED)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel(r'$3\,u_m v_m$')
a.set_ylabel(r'$\ell^{-1}\,\partial_\ell \bar U_i$')
for b in betas:
    a.plot([], [], mk[b], ms=3.5, color=MUTED, mec='none', label=rf'$\beta={b:g}$')
for m in range(3):
    a.plot([], [], '-', lw=1.5, color=cols[m], label=f'context {m+1}')
a.legend(loc='upper left')

# ---------- (b) learned ell_m(t), beta = 6 (solid) and 12 (dashed); inset v_m vs ell ----------
a = ax[0, 1]
t = np.arange(1, sw["b6_e1_s0"].shape[0] + 1) * int(sw["rec"]) * float(sw["dt"])
for m in range(3):
    a.plot(t, sw["b6_e1_s0"][:, 0, m], lw=1.5, color=cols[m])
    a.plot(t, sw["b12_e1_s0"][:, 0, m], '--', lw=1.1, color=cols[m])
for m, lc in enumerate(c.k_c(6.0)):
    if np.isfinite(lc):
        a.axhline(lc, color=cols[m], lw=0.7, alpha=0.6, ls=':')
a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(0.03, 12)
a.set_xlabel(r'time ($1/\gamma$)'); a.set_ylabel(r'learned slope $\ell_m$')
ins = inset_axes(a, width='36%', height='34%', loc='upper left', borderpad=1.6)
X = sw["b6_e1_s0"]; pos = ks > 0
for m in range(3):
    ins.plot(ks[pos], f["v"][bi, pos, m], '-', lw=3, alpha=0.3, color=cols[m])
    ins.plot(X[:, 0, m], X[:, 2, m], '.', ms=1.2, color=cols[m])
ins.set_xscale('log'); ins.set_yscale('log'); ins.set_xlim(0.04, 12)
ins.set_xlabel(r'$\ell$', fontsize=7, labelpad=0); ins.set_ylabel(r'$v_m$', fontsize=7, labelpad=0)
ins.tick_params(labelsize=6)
ins.spines[['top', 'right']].set_visible(True)

# ---------- (c) orientation from ell_0 = 0, 10 seeds: c_col = 0 against c_col = 0.1 ----------
a = ax[1, 0]
tt = np.arange(1, tl["k00_c0.1_s0"].shape[0] + 1) * int(tl["rec"]) * float(tl["dt"])
for s in range(10):
    a.plot(tt, sg[f"k00_s{s}"][:tt.size, 0, 0], color=cols[1], lw=0.9, alpha=0.7)
    a.plot(tt, tl[f"k00_c0.1_s{s}"][:, 0, 0], color=cols[0], lw=0.9, alpha=0.8)
a.plot([], [], color=cols[1], lw=1.5, label=r'$c_{\mathrm{col}}=0$')
a.plot([], [], color=cols[0], lw=1.5, label=r'$c_{\mathrm{col}}=0.1$')
a.axhline(0, color=MUTED, lw=0.7)
a.set_yscale('symlog', linthresh=0.1); a.set_xscale('log')
a.set_xlabel(r'time ($1/\gamma$)'); a.set_ylabel(r'learned slope $\ell_1$')
a.legend(loc='upper left')

# ---------- (d) anti-follow basin edge |ell*| against the tilt ----------
a = ax[1, 1]
CL = list(tb["c_list"]); G = tb["grad"]; kb = tb["ks"]
neg = (kb <= 0) & (kb >= -LMAX)
for m, mark in zip(range(3), ['o', 's', '^']):
    lst = []
    for i in range(1, len(CL)):
        g, kk = G[i, neg, m], kb[neg]
        j = [q for q in range(len(kk) - 1) if g[q] < 0 <= g[q + 1]][-1]
        lst.append(kk[j] - g[j] * (kk[j + 1] - kk[j]) / (g[j + 1] - g[j]))
    a.plot(CL[1:], -np.array(lst), mark, ms=5, color=cols[m], label=f'context {m+1}')
cc = np.linspace(0, 2.1, 50)
a.plot(cc, cc / (6 * c.D0), '-', lw=0.9, color=MUTED, label=r'$c_{\mathrm{col}}/6\Delta_0$')
a.set_xlabel(r'collision tilt $c_{\mathrm{col}}$'); a.set_ylabel(r'basin edge $|\ell^\ast|$')
a.legend(loc='upper left')

for a, lab in zip(ax.flat, 'abcd'):
    a.text(-0.2, 1.02, f'({lab})', transform=a.transAxes, fontsize=10, fontweight='bold',
           va='bottom')
fig.tight_layout()
fig.savefig(_out("fig_s_learning.pdf"))
print("wrote figures/fig_s_learning.pdf")
