"""Status-plane animation for the degenerate pair (promotional; not part of the
submission).

One panel, square, for a social feed: the status plane (tanh(omega.g_1),
tanh(omega.g_2)) with a marginal on each axis, and the status variance of every
axis inset in the centre, which the role groups leave empty.

What it shows. Axes 1 and 2 are nearly degenerate (sim_ring.py), so both leave
the noise floor at the same beta. Each then divides the population on its own:
both marginals go bimodal and the plane fills its four corners, one group per
role combination. Two schemas of equal strength ignite together and give four
roles. The readout counts the groups as 2 to the number of bimodal axes
(bimodality coefficient above 5/9, the value of a uniform law).

Statuses are confined to (-1, 1), so the frame is fixed for the whole clip: the
noise-floor cloud at the start and the saturated corners at the end share it.

Writes promo/ring.mp4.
"""
import os
import shutil
import subprocess
import tempfile
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "ring_data.npz"))
betas, eigs, z = D["betas"], D["eigs"], D["s_cloud"]   # statuses
svar = D["svar"]
floor, contrast = float(D["floor"]), D["contrast"]
M = eigs.shape[1]
ACTIVE = 2.0 * floor

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13})

zl = 1.08                                   # statuses lie in (-1, 1)


def smooth(H):
    return gaussian_filter(H, sigma=2.0, mode="constant")


def density_stack():
    """Per-frame density, smoothed in space then in time (as make_ramp_animation)."""
    NB = 60
    e = np.linspace(-zl, zl, NB + 1)
    out = np.zeros((len(betas), NB, NB))
    ema = None
    for i in range(len(betas)):
        H = smooth(np.histogram2d(z[i, :, 0], z[i, :, 1], bins=[e, e])[0])
        ema = H if ema is None else 0.92 * ema + 0.08 * H
        out[i] = ema
    return out, e


DENS, EDGE = density_stack()
MX, MY = DENS.sum(axis=2), DENS.sum(axis=1)
CEN = 0.5 * (EDGE[1:] + EDGE[:-1])


def _ema(v, a=0.06):
    out = np.empty_like(v, dtype=float)
    e = float(v[0])
    for i, x in enumerate(v):
        e = (1 - a) * e + a * float(x)
        out[i] = e
    return out


MXMAX, MYMAX = _ema(MX.max(axis=1)) * 1.2, _ema(MY.max(axis=1)) * 1.2
VMAX = _ema(np.array([np.percentile(d, 99.5) for d in DENS]), 0.06)
EIGS_S = np.vstack([_ema(svar[:, k], 0.10) for k in range(M)]).T
ACTIVE_K = 2.0 * floor * np.asarray(D["u"])              # status floor per axis, |g_k|^2 floor


def bimod(x):
    x = x - x.mean(); m2 = (x**2).mean()
    return ((x**3).mean()**2 / m2**3 + 1) / ((x**4).mean() / m2**2)


BC = np.array([[bimod(z[i, :, c]) for c in range(2)] for i in range(len(betas))])
BC = np.vstack([_ema(BC[:, c], 0.10) for c in range(2)]).T
GROUPS = 2 ** (BC > 5 / 9).sum(axis=1)

fig = plt.figure(figsize=(9.6, 9.6))
gs = fig.add_gridspec(2, 2, width_ratios=[5.0, 0.8], height_ratios=[0.8, 5.0],
                      wspace=0.05, hspace=0.05,
                      left=0.10, right=0.96, top=0.965, bottom=0.08)
axT = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[1, 0])
axR = fig.add_subplot(gs[1, 1], sharey=axM)

im = axM.imshow(DENS[0].T, origin="lower", cmap="Blues", aspect="auto",
                extent=[-zl, zl, -zl, zl], vmin=0.0, vmax=VMAX[0],
                interpolation="bicubic")
axM.set_xlabel(r"status on axis 1   $\tanh(\omega_i\cdot g_1)$")
axM.set_ylabel(r"status on axis 2   $\tanh(\omega_i\cdot g_2)$")
axM.set_xlim(-zl, zl)
axM.set_ylim(-zl, zl)

(ltop,) = axT.plot(CEN, MX[0], lw=2.0, color="#1f77b4")
(lrig,) = axR.plot(MY[0], CEN, lw=2.0, color="#1f77b4")
for a in (axT, axR):
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
axT.set_xlim(-zl, zl)
axR.set_ylim(-zl, zl)

# The four role groups leave the centre empty, so the spectrum sits there.
# Transparent patch: the density behind it stays visible.
kk = np.arange(1, M + 1)
axI = axM.inset_axes([0.355, 0.375, 0.29, 0.25])
axI.patch.set_alpha(0.0)
sc = axI.scatter(kk, EIGS_S[0], s=40, color="#3b78b5", zorder=3)
axI.set_xlabel(r"role axis $k$", fontsize=10.5, labelpad=1.0, color="0.25")
axI.set_title(r"status variance", fontsize=11.5, pad=3.0, color="0.2")
axI.set_xticks([1, 4, 8])
axI.set_yticks([0, 0.5, 1])
axI.tick_params(labelsize=9.5, colors="0.35", length=2.5, pad=1.5)
axI.set_xlim(0.4, M + 0.6)
axI.set_ylim(0.0, 1.05)                                 # fixed: a status variance is at most 1
for _s in ("top", "right"):
    axI.spines[_s].set_visible(False)
for _s in ("left", "bottom"):
    axI.spines[_s].set_color("0.45")
readout = fig.text(0.975, 0.975, "", fontsize=13, va="top", ha="right",
                   color="0.25", linespacing=1.5)            # free corner above the right marginal

def update(i):
    im.set_data(DENS[i].T)
    im.set_clim(0.0, VMAX[i])
    ltop.set_ydata(MX[i])
    lrig.set_xdata(MY[i])
    axT.set_ylim(0, MXMAX[i])
    axR.set_xlim(0, MYMAX[i])
    sc.set_offsets(np.c_[kk, EIGS_S[i]])
    sc.set_color(["#3b78b5" if v > a else "#bdbdbd" for v, a in zip(EIGS_S[i], ACTIVE_K)])
    readout.set_text(rf"$\beta={betas[i]:.1f}$" + f"\n{GROUPS[i]} role group" + ("s" if GROUPS[i] > 1 else ""))
    return im, ltop, lrig, sc, readout


ani = animation.FuncAnimation(fig, update, frames=len(betas), interval=33, blit=False)
tmp = tempfile.mkdtemp(prefix="ring_render_")
try:
    scratch = os.path.join(tmp, "ring.mp4")
    ani.save(scratch, writer=animation.FFMpegWriter(fps=30, codec="libopenh264",
                                                    bitrate=3500))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", scratch, "-c", "copy",
                    "-movflags", "+faststart", os.path.join(HERE, "ring.mp4")],
                   check=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
plt.close(fig)
print("done -> promo/ring.mp4")
