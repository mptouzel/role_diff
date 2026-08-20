import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)                  # .../culture_engine_PRX
DATA_PATH = os.path.join(PKG_ROOT, "fig1_data.npz")
FIG_DIR = os.path.join(PKG_ROOT, "figures")

plt.rcParams.update({'font.size': 8.5, 'axes.linewidth': 0.8,
                     'font.family': 'sans-serif'})

d = np.load(DATA_PATH)
betas, eigs, gnorm = d['betas'], d['eigs'], d['gnorm']
floor = d['sigma_dyn']**2 / (2 * d['gamma'])

fig = plt.figure(figsize=(7.2, 5.6))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1, 1],
                       hspace=0.12, wspace=0.28, left=0.02, right=0.965,
                       top=0.955, bottom=0.09)

# ---------------- (a) self-consistent loop schematic --------------------
axA = fig.add_subplot(gs[:, 0]); axA.set_xlim(-0.4, 10); axA.set_ylim(0, 10)
axA.axis('off')

def box(x, y, w, h, text, fc, title, tfs=9.5, bfs=6.9, lw=1.0):
    axA.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.12",
                 fc=fc, ec='k', lw=lw, zorder=3))
    axA.text(x, y + h/2 - 0.30, title, ha='center', va='center',
             fontsize=tfs, fontweight='bold', zorder=4)
    axA.text(x, y - 0.22, text, ha='center', va='center', fontsize=bfs, zorder=4)

def arrow(p0, p1, rad, color='0.15', lw=2.0, ms=22, shrinkB=8, shrinkA=3):
    axA.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                 arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color,
                 shrinkA=shrinkA, shrinkB=shrinkB, zorder=2))

def lab(x, y, t, fs=6.1, ha='center', color='k'):
    axA.text(x, y, t, fontsize=fs, ha=ha, va='center', color=color,
             bbox=dict(fc='white', ec='none', alpha=0.9, pad=0.5), zorder=5)

# Directed information-engine ring (clockwise): Roles -> Policy -> Resources ->
# Schemas -> Roles. Policy & Schemas share a vertical level. Ring arcs bow OUTWARD
# (convex) to open the centre for the ignition box; the fast ignition loop is the
# inner chord Policy -> Roles (action feedback).
box(4.6, 8.35, 3.7, 1.95, r"role states $\boldsymbol{\omega}_i$" "\n"
    r"$\dot{\boldsymbol{\omega}}_i=-\gamma\boldsymbol{\omega}_i(1{+}|\boldsymbol{\omega}_i|^2)$" "\n"
    r"$+\beta\sum_m f_m\langle a\rangle_m\mathbf{g}_m+\boldsymbol{\eta}_i$", '#e7f2e4', 'Roles')
box(8.0, 5.32, 2.5, 1.5, "decision rule" "\n" r"$a_{im}=\mathbb{1}[s_{im}{>}0]$",
    '#efe6f7', r'Policy $\pi$', tfs=8.5, bfs=6.4)
box(4.6, 1.5, 3.5, 1.5, r"agent wealth $r_i$" "\n"
    r"$\dot{r}_i=\sum_m W_{im}-c\,r_i$", '#fdeede', 'Resources')
box(1.75, 5.32, 3.5, 1.6, r"context directions $\mathbf{g}_m$" "\n"
    r"$\dot{\mathbf{g}}_m=-\kappa\mathbf{g}_m+\alpha\mathbf{C}_r\mathbf{g}_m$", '#dbe9f6', 'Schemas')

# ring arrows (clockwise, bowing OUTWARD; labels pushed outside each arc)
arrow((6.00, 7.30), (7.35, 6.18), -0.22)
lab(8.15, 7.25, "measure\n" r'$s_{im}=(\boldsymbol{\omega}_i{-}\boldsymbol{\omega}_j)\!\cdot\!\mathbf{g}_m+\xi$', color='0.15')
arrow((7.55, 4.45), (5.75, 2.42), -0.22)
lab(8.35, 2.85, "extract work\n" r'$W_{im}=w_0(1{-}2P_{e,im})^2$', color='0.15')
arrow((3.00, 2.42), (1.98, 4.40), -0.22)
lab(1.20, 2.85, "reweight\n"
    r'$\mathbf{C}_r=\dfrac{\sum_i r_i\boldsymbol{\omega}_i\boldsymbol{\omega}_i^{\!\top}}{\sum_i r_i}$', color='0.15')
arrow((2.28, 6.24), (3.45, 7.24), -0.22)
lab(1.40, 7.20, "frame " r'$\mathbf{g}_m$', color='0.25')

# inner feedback chord: Policy -> Roles (fast ignition loop)
arrow((6.65, 6.15), (5.55, 7.24), 0.0, color='#a03a00', lw=2.3, ms=18, shrinkB=6)
lab(4.95, 6.55, "feedback " r'$a_{im}$', fs=6.0, color='#a03a00')

# ignition condition -- open centre of the ring
axA.text(4.75, -99, "ignition:\n"
         r'$\Lambda=\dfrac{2\beta}{\gamma}\dfrac{\eta_{\rm agent}\phi_{\rm pair}}{\sigma_{\rm obs}}>1$',
         ha='center', va='center', fontsize=8.3, zorder=6,
         bbox=dict(fc='#fff8dc', ec='0.4', boxstyle='round,pad=0.42'))
axA.text(4.6, 0.28, 'Go/Defer anti-coordination game:  '
         r'$(G,G)\,{-}1$   $(G,D)\;6{,}2$   $(D,D)\;3$', ha='center', fontsize=6.4, color='0.4')
axA.text(-0.2, 9.75, 'a', fontsize=12, fontweight='bold')

# ---------------- (b) two-mode scatter ----------------------------------
axB = fig.add_subplot(gs[0, 1])
cols = ['#1f77b4', '#ff7f0e', '#2ca02c']
beta_c = d['beta_c']
z_snap = d['z_snap']            # (n_snap_beta, n_cloud, M)
snap_betas = d['snap_betas']
# Give each (beta, axis) column real width so the two-mode DENSITY is visible:
# subsample to a set of clean, well-spaced beta columns, offset the three axes,
# and jitter agents horizontally within each column (otherwise all 260 agents of
# a column land on one x and overprint into a vertical smear that hides the split).
rng_b = np.random.default_rng(0)
off = 0.36                                   # horizontal separation of the three axes
jw = 0.13                                    # within-column horizontal jitter half-width
ax_off = (np.arange(3) - 1) * off
# Keep only columns whose offset triplet fits inside the shared x-range (else blue
# clips off the left edge, green off the right), then take every other one so the
# spacing is uniform -- picking evenly spaced INDICES, not nearest matches to
# evenly spaced betas, which rounds unevenly and leaves a gap mid-range.
_valid = [i for i in range(len(snap_betas))
          if betas[0] + off + jw <= snap_betas[i] <= betas[-1] - off - jw]
ib_sel = np.array(_valid[::2])
for k in range(3):
    for ib in ib_sel:
        zz = z_snap[ib, :, k]
        zz = zz[~np.isnan(zz)]
        bx = snap_betas[ib] + ax_off[k] + rng_b.uniform(-jw, jw, zz.size)
        axB.scatter(bx, zz, s=5, alpha=0.32, color=cols[k],
                    edgecolors='none', rasterized=True, zorder=2)
for k, lab in enumerate(['axis 1', 'axis 2', 'axis 3']):
    axB.plot([], [], 'o', color=cols[k], ms=4, label=lab)
# mark where each mode splits (its critical coupling)
for k in range(3):
    axB.axvline(beta_c[k], color=cols[k], lw=0.8, ls='--', alpha=0.45)
axB.axhline(0, color='0.6', lw=0.6, zorder=0)
axB.set_ylabel(r'agent status  $z_{im}=\boldsymbol{\omega}_i\!\cdot\!\hat{\mathbf{g}}_m$')
axB.legend(frameon=False, fontsize=7, loc='upper left', ncol=3,
           handlelength=0.6, columnspacing=0.9, borderaxespad=0.2,
           handletextpad=0.3)
axB.set_xlim(betas[0], betas[-1]); axB.tick_params(labelbottom=False)
axB.set_ylim(-2.4, 2.4)
axB.text(-0.16, 1.02, 'b', transform=axB.transAxes, fontsize=12,
         fontweight='bold')
axB.text(0.97, 0.06, 'splitting is ordered by axis strength',
         transform=axB.transAxes, fontsize=6.6, color='0.4', ha='right')

# ---------------- (c) the cascade (quasi-static), normalized ------------
axC = fig.add_subplot(gs[1, 1], sharex=axB)
gamma = float(d['gamma'])
f_pair = float(d['f_pair'])
mu_k = d['mu'] if 'mu' in d.files else gamma / beta_c
gain_k = f_pair * mu_k                        # effective gain under random pairing
gamma_eff = gamma                             # linearization: cubic term -> gamma at omega->0

# linear mean-field theory (random pairing), normalized by the noise floor:
# lambda_k / floor = gamma / (gamma_eff - beta f_pair mu_k), diverging exactly
# at the random-pairing threshold beta_c^(k) = gamma_eff/(f_pair mu_k).
for k in range(3):
    bb = np.linspace(betas[0], beta_c[k] * 0.985, 200)
    lam_lin = gamma / (gamma_eff - bb * gain_k[k])   # = (sig2/2(gamma_eff-...))/floor
    axC.plot(bb, lam_lin, color=cols[k], lw=0.9, ls=(0, (1, 1)), alpha=0.85)

# simulated equilibrated eigenvalues (solid), normalized by the noise floor
for k in range(3):
    axC.plot(betas, eigs[:, k] / floor, color=cols[k], lw=1.6,
             label=rf'$\lambda_{k+1}(\mathbf{{C}})$')

axC.axhline(1.0, color='k', ls=':', lw=1.0)
axC.text(0.30, 0.052, r'noise floor $\sigma_{\rm dyn}^2/2\gamma$',
         transform=axC.transAxes, fontsize=6.8, va='center', ha='center', color='0.35')

# exact critical couplings beta_c^(k) = gamma_eff/(f_pair mu_k) as vertical guides
for k in range(3):
    axC.axvline(beta_c[k], color=cols[k], lw=0.9, ls='--', alpha=0.55)
    axC.text(beta_c[k], 150, rf'$\beta_c^{{({k+1})}}$', color=cols[k],
             ha='center', va='top', fontsize=8.5, rotation=0,
             bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))

axC.set_yscale('log')
axC.set_xlabel(r'feedback strength $\beta$  (equilibrated at each $\beta$; $\Lambda\propto\beta$)')
axC.set_ylabel(r'order parameter  $\lambda_k(\mathbf{C})/(\sigma_{\rm dyn}^2/2\gamma)$')
axC.legend(frameon=False, fontsize=7.5, loc='lower right', handlelength=1.4)
axC.set_ylim(0.5, 220)
axC.set_xlim(betas[0], betas[-1])
axC.text(-0.16, 1.02, 'c', transform=axC.transAxes, fontsize=12,
         fontweight='bold')
axC.text(0.03, 0.79, 'dotted: linear theory\n'
         r'$\lambda_k=\dfrac{\sigma_{\rm dyn}^2}{2(\gamma_{\rm eff}-\beta \phi_{\rm pair}\mu_k)}$',
         transform=axC.transAxes, fontsize=6.4, color='0.4',
         ha='left', va='top',
         bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))

# Save under the final name; also refresh fig_loop_ignition, which is the name
# the manuscript's Figure 1 \includegraphics resolves to (main.tex), so the
# article compiles against this final version without a source edit.
for stem in ('fig_loop_full', 'fig_loop_ignition'):
    plt.savefig(os.path.join(FIG_DIR, stem + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, stem + '.pdf'), bbox_inches='tight')
print('saved:', ', '.join(('fig_loop_full', 'fig_loop_ignition')))
