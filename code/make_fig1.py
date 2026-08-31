import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)                  # .../culture_engine_PRX
DATA_PATH = os.path.join(PKG_ROOT, "fig1_data.npz")
FIG_DIR = os.path.join(PKG_ROOT, "figures")

plt.rcParams.update({'font.size': 8.5, 'axes.linewidth': 0.8,
                     'font.family': 'sans-serif'})

d = np.load(DATA_PATH)
betas, eigs, gnorm = d['betas'], d['eigs'], d['gnorm']
floor = d['sigma_dyn']**2 / (2 * d['gamma'])

fig = plt.figure(figsize=(7.2, 3.5))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1, 1],
                       hspace=0.12, wspace=0.28, left=0.02, right=0.965,
                       top=0.955, bottom=0.09)

# ---------------- (a) state / signal-flow schematic ---------------------
axA = fig.add_subplot(gs[:, 0]); axA.set_xlim(-0.7, 10.5); axA.set_ylim(0.55, 11.05)   # top extended by 0.30 to
# drop the diagram slightly clear of the ignition text above it
axA.axis('off')
GRN = '#3f7a55'; PUR = '#7a3fa0'

def sbox(x, y, text, fc, title, w=3.0, h=1.55):
    axA.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                 fc=fc, ec='none', lw=1.0, zorder=3))
    axA.text(x, y+h/2-0.34, title, ha='center', va='center', fontsize=9.5,
             fontweight='bold', zorder=4)
    axA.text(x, y-0.25, text, ha='center', va='center', fontsize=6.5, zorder=4)

def fbox(x, y, title, sub, w=2.6, h=1.15):
    axA.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                 fc='#efe6f7', ec='none', lw=1.0, zorder=3))
    axA.text(x, y+h/2-0.32, title, ha='center', va='center', fontsize=8.5,
             fontweight='bold', zorder=4)
    axA.text(x, y-h/2+0.44, sub, ha='center', va='center', fontsize=6.5, zorder=4)

def A(p0, p1, color='0.15', lw=1.9, dashed=False, ms=15, sA=2, sB=6, rad=0.0):
    axA.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                 arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color,
                 shrinkA=sA, shrinkB=sB, zorder=2,
                 linestyle='--' if dashed else '-'))

def smerge(s1, e1, s2, e2, J, T, color, dashed=False, lw=1.7, kin=1.2,
           kout1=1.6, kout2=1.6):
    J = np.array(J, float); T = np.array(T, float); u = T-J; u = u/np.linalg.norm(u)
    P2 = J - kin*u
    for S, e, ko in ((s1, e1, kout1), (s2, e2, kout2)):
        S = np.array(S, float); e = np.array(e, float); e = e/np.linalg.norm(e)
        P1 = S + ko*e
        axA.add_patch(PathPatch(Path([tuple(S), tuple(P1), tuple(P2), tuple(J)],
            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]), fill=False,
            lw=lw, edgecolor=color, ls='--' if dashed else '-', zorder=2,
            capstyle='round'))
    A(tuple(J), tuple(T), color=color, lw=lw, dashed=dashed, sA=0, sB=6)

def bez(P0, P1, P2, P3, color, dashed=True, lw=1.7):
    axA.add_patch(PathPatch(Path([P0, P1, P2, P3],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]), fill=False,
        lw=lw, edgecolor=color, ls='--' if dashed else '-', zorder=2,
        capstyle='round'))

# LEFT: the three state variables; RIGHT: the fast signal-flow chain
sbox(1.7, 8.4, r"$M$ contexts $\mathbf{g}_m=\sqrt{u_m}\,\hat{\mathbf{g}}_m$" "\n"
     r"$\dot u_m=2u_m(\alpha\lambda_m-\kappa)$" "\n"
     r"$\lambda_m=\hat{\mathbf{g}}_m^{\top}\mathbf{C}_r\hat{\mathbf{g}}_m$",
     '#dbe9f6', 'Schemas', w=3.9, h=1.95)
sbox(2.9, 5.0, r"$N_{\mathrm{a}}$ identity states $\boldsymbol{\omega}_i$" "\n"
     r"$\dot{\boldsymbol{\omega}}_i\approx-\gamma\boldsymbol{\omega}_i+\beta\langle a\rangle_m\mathbf{g}_m+\boldsymbol{\eta}_i$",
     '#e7f2e4', 'Identities', w=4.3)
sbox(2.45, 1.5, r"agent wealth $r_i$" "\n"
     r"$\dot r_i=\sum_m f_m W_{im}-c\,r_i$", '#fdeede', 'Resources', w=3.2)
SX = 8.4
fbox(SX, 8.4, 'Role Signal',
     r"$\Delta_{im}(j)=(\boldsymbol{\omega}_i{-}\boldsymbol{\omega}_j)\cdot\mathbf{g}_m$" "\n"
     r"$s_{im}=\Delta_{im}(j)+\sigma_{\rm obs}\xi$", w=3.4, h=1.55)
fbox(SX, 5.0, 'Action', r"$a_{im}=\pi(s_{im})$", w=3.1, h=1.15)
fbox(SX, 1.5, 'Reward',
     r"$W_{im}=w_0(1{-}2P_{e,im})^2$" "\n"
     r"$P_{e,im}=\Phi(-|\Delta_{im}|/\sigma_{\rm obs})$", w=3.4, h=1.55)

# band behind Role Signal + Action: both are averages over the drawn partner
axA.add_patch(FancyBboxPatch((6.45, 3.70), 3.90, 5.70, boxstyle="round,pad=0.05",
             fc='#f0f0f0', ec='none', zorder=1))
axA.text(10.62, 6.55, r'average over encounters with many $j$', rotation=90,
         ha='center', va='center', fontsize=5.9, color='0.35',
         zorder=6, clip_on=False)
axA.text(10.22, 3.86, 'encounter', ha='right', va='bottom',
         fontsize=6.5, color='0.35', zorder=6)

# SOLID feed-forward read-out: (Schema,Identity)->Signal->Action->Reward->Resources
smerge((3.25, 8.4), (1, 0), (3.35, 5.75), (0.55, 1), (5.75, 8.4), (SX-1.5, 8.4),
       '0.15', kin=1.4, kout1=0.5, kout2=2.5)
A((SX, 7.63), (SX, 5.58)); A((SX, 4.42), (SX, 2.28)); A((6.7, 1.5), (4.10, 1.5))
# GREEN reinforce (slow): Resources + Identity -> Schema, nested up the left
bez((0.85, 1.55), (-0.55, 1.55), (0.25, 6.25), (0.90, 6.75), GRN)
bez((1.10, 5.55), (0.65, 6.05), (0.75, 6.55), (0.90, 6.75), GRN)
A((0.90, 6.75), (1.50, 7.42), color=GRN, dashed=False, sA=0, sB=6)
# PURPLE write (slow): Action + Schema merge, then one trunk up into Identity's base
bez((3.25, 8.4), (4.8, 8.4), (5.8, 4.30), (5.8, 3.70), PUR)
bez((SX-1.3, 5.00), (6.3, 4.55), (5.8, 4.30), (5.8, 3.70), PUR)
bez((5.8, 3.70), (5.9, 2.45), (3.2, 2.40), (2.90, 3.40), PUR)
A((2.90, 3.40), (2.90, 4.22), color=PUR, dashed=False, sA=0, sB=6)

axA.text(4.9, 8.72, "read", fontsize=6.6, ha='center', color='0.3', zorder=6)
axA.text(4.35, 3.42, "write  " r"$\beta\langle a\rangle_m\mathbf{g}_m$", fontsize=6.6,
         ha='center', color=PUR, zorder=6,
         bbox=dict(fc='white', ec='none', alpha=0.9, pad=0.4))
axA.text(-0.05, 6.35, "reinforce\n" r"$\mathbf{C}_r$", fontsize=6.6, ha='center',
         color=GRN, zorder=6)
axA.text(5.5, 1.86, "accumulate", fontsize=6.5, ha='center', color='0.3', zorder=6)
axA.text(-0.02, 1.01, 'a', transform=axA.transAxes, fontsize=12, fontweight='bold')
axA.text(4.9, 10.90, 'ignition at', ha='center', va='center', fontsize=9.5, zorder=6)
axA.text(4.9, 10.22,
         r'critical loop gain  $\Lambda=\dfrac{2\beta}{\gamma}\,\dfrac{\eta_{\rm agent}(1-\rho_{\rm pair})}{\sigma_{\rm obs}}=1$',
         ha='center', va='center', fontsize=9.5, zorder=6)

# ------- (b) merged: agent-status swarm + order-parameter branches + histogram -------
_bpos = gs[:, 1].get_position(fig); _bm = _bpos.height*0.11
axM = fig.add_axes([_bpos.x0, _bpos.y0+_bm, _bpos.width, _bpos.height-_bm])
cols = ['#1f77b4', '#ff7f0e', '#2ca02c']
rho_pair = 0.0                                # random pairing; factor (1-rho_pair)=1
mu_k = d['mu']
beta_c = d['gamma'] / ((1.0 - rho_pair) * mu_k)
z_snap = d['z_snap']; snap_betas = d['snap_betas']
gamma = float(d['gamma']); gamma_eff = gamma
gain_k = (1.0 - rho_pair) * mu_k
XMAX = 20.0
famp = np.sqrt(floor)                          # noise-floor amplitude
YLIM = 2.7

# individual-agent status swarm (jittered columns at snapshot betas) -- background
rng_b = np.random.default_rng(0)
off, jw = 0.34, 0.12
ax_off = (np.arange(3) - 1) * off
_valid = [i for i in range(len(snap_betas))
          if betas[0] + off + jw <= snap_betas[i] <= XMAX - off - jw]
ib_sel = np.array(_valid[::2])
for k in range(3):
    for ib in ib_sel:
        zz = z_snap[ib, :, k]; zz = zz[~np.isnan(zz)]
        bx = snap_betas[ib] + ax_off[k] + rng_b.uniform(-jw, jw, zz.size)
        axM.scatter(bx, zz, s=4, alpha=0.18, color=cols[k], edgecolors='none',
                    rasterized=True, zorder=2)

# order-parameter branches: solid +/- sqrt(v_k) (simulation), dotted linear theory
for k in range(3):
    amp = np.sqrt(eigs[:, k])
    axM.plot(betas,  amp, color=cols[k], lw=1.7, zorder=4)
    axM.plot(betas, -amp, color=cols[k], lw=1.7, zorder=4)
    bb = np.linspace(betas[0], beta_c[k] * 0.985, 200)
    amp_lin = np.sqrt(floor * gamma / (gamma_eff - bb * gain_k[k]))
    axM.plot(bb,  amp_lin, color=cols[k], lw=0.9, ls=(0, (1, 1)), alpha=0.9, zorder=3)
    axM.plot(bb, -amp_lin, color=cols[k], lw=0.9, ls=(0, (1, 1)), alpha=0.9, zorder=3)

# noise band with vertically centred label
axM.axhspan(-famp, famp, color='0.86', zorder=0)
axM.axhline(0.0, color='0.6', lw=0.5, ls=':', zorder=1)
axM.text(13.5, 0.0, r'noise floor $\pm\sqrt{\sigma_{\rm dyn}^2/2\gamma}$',
         fontsize=6.4, va='center', ha='center', color='k', zorder=5)

# beta_c guides + labels (same annotations as the original panel)
_trx = axM.get_xaxis_transform()          # x in data coords, y in axes coords
for k in range(3):
    axM.axvline(beta_c[k], color=cols[k], lw=0.9, ls='--', alpha=0.5, zorder=1)
    axM.text(beta_c[k], -0.022, rf'$\beta_c^{{({k+1})}}$', color=cols[k],
             ha='center', va='top', fontsize=8.5, transform=_trx,
             clip_on=False, zorder=6)

# cascade: hopping arrows along the top spine, beta_c^(1) -> (2) -> (3).
# rad is scaled by 1/span so both hops rise to the same height.
_span = np.diff(beta_c[:3])
_apex = 0.20 * _span[1]
for k in (0, 1):
    axM.add_patch(FancyArrowPatch(
        (beta_c[k], 1.0), (beta_c[k+1], 1.0), transform=_trx,
        connectionstyle=f'arc3,rad={-_apex/_span[k]:.3f}', arrowstyle='-|>',
        mutation_scale=7, lw=0.9, color='0.35',
        shrinkA=0.5, shrinkB=0.5, clip_on=False, zorder=7))
axM.text(0.5*(beta_c[0]+beta_c[2]), 1.060, 'cascade', transform=_trx,
         ha='center', va='bottom', fontsize=7.5, color='0.35',
         clip_on=False, zorder=7)

axM.set_xlim(betas[0], XMAX); axM.set_ylim(-YLIM, YLIM)
axM.set_xticks([5, 10, 15, 20])           # keep the tick at 5, drop its label:
axM.set_xticklabels(['', '10', '15', '20'])   # beta_c^(2)=3.99 sits where '5' would print
axM.set_xlabel(r'feedback strength $\beta$  (equilibrated at each $\beta$; $\Lambda\propto\beta$)')
axM.set_ylabel(r'differentiation order parameter  $\pm\sqrt{v_k}$')
axM.text(-0.13, 1.01, 'b', transform=axM.transAxes, fontsize=12, fontweight='bold')
for k, lb in enumerate([r'$k=1$', r'$k=2$', r'$k=3$']):
    axM.plot([], [], 'o', color=cols[k], ms=4, label=lb)
axM.text(0.015, 0.02, 'solid: sim\ndotted: linear theory', transform=axM.transAxes,
         va='bottom', ha='left', fontsize=6.8, color='0.3', linespacing=1.35, zorder=6)
axM.legend(frameon=False, fontsize=7, loc='lower right', ncol=3, handlelength=0.6,
           columnspacing=0.9, handletextpad=0.3, borderaxespad=0.3)

# right-side histogram strip = numeric status axis (RHS), shared scale
divider = make_axes_locatable(axM)
axH = divider.append_axes("right", size="17%", pad=0.06, sharey=axM)
ib = int(np.argmin(np.abs(snap_betas - XMAX)))
yy = np.linspace(-YLIM, YLIM, 400)
for k in (2, 1, 0):
    zz = z_snap[ib, :, k]; zz = zz[~np.isnan(zz)]
    dens = gaussian_kde(zz)(yy); dens = dens / dens.max() * 0.95
    axH.fill_betweenx(yy, 0, dens, color=cols[k], alpha=0.32, lw=0)
    axH.plot(dens, yy, color=cols[k], lw=1.2)
axH.set_xlim(0, 1.08); axH.set_xticks([])
axH.axhline(0.0, color='0.6', lw=0.5, ls=':')
for _sp in ('top', 'bottom', 'right'):
    axH.spines[_sp].set_visible(False)
axH.tick_params(axis='y', which='both', left=False, right=False,
                labelleft=False, labelright=False)
axH.yaxis.set_label_position('right')
axH.set_ylabel(r'agent status  $\boldsymbol{\omega}_i\cdot\hat{\mathbf{g}}_{m(k)}$'
               '\n' r'(individual agents shown as dots at left)', fontsize=8)
axH.set_title(rf'$\beta={snap_betas[ib]:.0f}$', fontsize=7)

# Save under the final name; also refresh fig_loop_ignition, which is the name
# the manuscript's Figure 1 \includegraphics resolves to (main.tex), so the
# article compiles against this final version without a source edit.
for stem in ('fig_loop_ignition',):
    plt.savefig(os.path.join(FIG_DIR, stem + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, stem + '.pdf'), dpi=600, bbox_inches='tight')   # rasterized scatter
print('saved:', 'fig_loop_ignition')
