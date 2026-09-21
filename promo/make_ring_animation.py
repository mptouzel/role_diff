"""Status-plane animation for the degenerate pair (promotional; not part of the
submission).

One panel, square, for a social feed: the status plane (z_1, z_2) with a
marginal on each axis, and the covariance spectrum on a LINEAR scale inset in
the corner the joint plot leaves free.

What it shows. Axes 1 and 2 are nearly degenerate (sim_ring.py), so their
eigenvalues leave the noise floor together. Both marginals then go bimodal,
which looks like two independent role divisions and is not one: the isotropic
quartic of Eq. (1) leaves an approximate continuous symmetry in the plane and
the density forms a ring. The readout carries the diagnostic that separates the
two, the angular contrast max/min of the density in angle -- about 2:1 for a
ring, above 8:1 for four separated groups.

The spectrum is linear here on purpose. A log axis makes every mode a visible
event, which is what cascade.mp4 wants; a linear axis shows that two modes carry
essentially all the variance and the other six are flat against the floor.

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
betas, eigs, z = D["betas"], D["eigs"], D["z_cloud"]
floor, contrast = float(D["floor"]), D["contrast"]
M = eigs.shape[1]
ACTIVE = 2.0 * floor

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13})

zl = float(np.percentile(np.abs(z[-1]), 99.5)) * 1.25


def smooth(H):
    return gaussian_filter(H, sigma=1.5, mode="constant")


def density_stack():
    """Per-frame density, smoothed in space then in time (as make_ramp_animation)."""
    NB = 34
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
EIGS_S = np.vstack([_ema(eigs[:, k], 0.10) for k in range(M)]).T

fig = plt.figure(figsize=(9.6, 9.6))
gs = fig.add_gridspec(2, 2, width_ratios=[5.0, 0.8], height_ratios=[0.8, 5.0],
                      wspace=0.05, hspace=0.05,
                      left=0.10, right=0.96, top=0.965, bottom=0.08)
axT = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[1, 0])
axR = fig.add_subplot(gs[1, 1], sharey=axM)

im = axM.imshow(DENS[0].T, origin="lower", cmap="Blues", aspect="auto",
                extent=[-zl, zl, -zl, zl], vmin=0.0, vmax=VMAX[0])
axM.set_xlabel(r"status on axis 1   $\omega_i\cdot\hat g_1$")
axM.set_ylabel(r"status on axis 2   $\omega_i\cdot\hat g_2$")
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

# The ring leaves its centre empty, so the spectrum sits there rather than in a
# corner. Transparent patch: the density behind it stays visible.
kk = np.arange(1, M + 1)
# Sized to the clear hole: agents reach in to r = 1.83 of a 3.03 half-range, so
# the inscribed square is 0.43 of the axes. The box is held inside that with
# room left for the title and tick labels, which would otherwise sit on the ring.
axI = axM.inset_axes([0.355, 0.375, 0.29, 0.25])
axI.patch.set_alpha(0.0)
sc = axI.scatter(kk, EIGS_S[0], s=40, color="#3b78b5", zorder=3)
axI.axhline(floor, ls="--", lw=1.0, color="0.45", zorder=2)
axI.set_xlabel(r"role axis $k$", fontsize=10.5, labelpad=1.0, color="0.25")
axI.set_title(r"spectrum $v_k$", fontsize=11.5, pad=3.0, color="0.2")
axI.set_xticks([1, 4, 8])
axI.set_yticks([0, 1, 2])
axI.tick_params(labelsize=9.5, colors="0.35", length=2.5, pad=1.5)
axI.set_xlim(0.4, M + 0.6)
axI.set_ylim(0.0, 2.7)                                  # fixed, so heights compare
for _s in ("top", "right"):
    axI.spines[_s].set_visible(False)
for _s in ("left", "bottom"):
    axI.spines[_s].set_color("0.45")

def update(i):
    im.set_data(DENS[i].T)
    im.set_clim(0.0, VMAX[i])
    ltop.set_ydata(MX[i])
    lrig.set_xdata(MY[i])
    axT.set_ylim(0, MXMAX[i])
    axR.set_xlim(0, MYMAX[i])
    sc.set_offsets(np.c_[kk, EIGS_S[i]])
    sc.set_color(["#3b78b5" if v > ACTIVE else "#bdbdbd" for v in EIGS_S[i]])
    return im, ltop, lrig, sc


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
