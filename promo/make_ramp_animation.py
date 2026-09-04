"""Two-panel animation of the cascade (promotional; not part of the submission).

Left  : the covariance spectrum at the current instant. Sixteen eigenvalues
        fluctuate about the noise floor sigma_w^2 / 2 gamma and lift off it one
        at a time as the role along that axis differentiates.
Right : the cascade N(t), the count of eigenvalues above the floor, with a
        marker riding the curve at the current time.

Reads promo/ramp_data.npz (written by promo/sim_ramp.py), writes promo/cascade.mp4.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "ramp_data.npz"))
betas, eigs, beta_c = d["betas"], d["eigs"], d["beta_c"]
floor, M = float(d["floor"]), eigs.shape[1]

ACTIVE = 2.0 * floor                      # "above the floor" criterion
N_of_t = (eigs > ACTIVE).sum(axis=1)
k = np.arange(1, M + 1)
cmap = plt.get_cmap("viridis")
colors = [cmap(0.08 + 0.84 * i / (M - 1)) for i in range(M)]

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.3), width_ratios=[1.05, 1])

# ---- left: the instantaneous spectrum ------------------------------------
axL.axhspan(eigs.min() * 0.5, ACTIVE, color="0.92", zorder=0)
axL.axhline(ACTIVE, color="0.45", lw=1.0, ls="--", zorder=1)
axL.text(0.55, ACTIVE * 0.72, "noise floor  " r"$\sigma_\omega^2/2\gamma$",
         va="top", ha="left", fontsize=9, color="0.4")
pts = axL.scatter(k, eigs[0], s=70, c=colors, edgecolor="0.25", linewidth=0.6, zorder=3)
axL.set_yscale("log")
axL.set_xlim(0.3, M + 0.7)
axL.set_xticks(np.arange(2, M + 1, 2))
axL.set_ylim(eigs.min() * 0.6, eigs.max() * 1.8)
axL.set_xlabel(r"role axis  $k$")
axL.set_ylabel(r"covariance eigenvalue  $v_k$")
axL.set_title("(a)  the population's role structure", loc="left", fontsize=12)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ---- right: the cascade ---------------------------------------------------
axR.step(betas, N_of_t, where="post", color="0.88", lw=1.6, zorder=1)
trail, = axR.step([], [], where="post", color="tab:blue", lw=2.4, zorder=2)
dot, = axR.plot([], [], "o", ms=9, color="tab:blue", mec="white", mew=1.4, zorder=3)
axR.set_xlim(betas[0], betas[-1])
axR.set_ylim(-0.4, M + 0.6)
axR.set_xlabel(r"feedback strength  $\beta(t)$")
axR.set_ylabel(r"active role axes  $N(t)$")
axR.set_title("(b)  the cascade", loc="left", fontsize=12)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)
readout = axR.text(0.03, 0.94, "", transform=axR.transAxes, fontsize=10,
                   va="top", ha="left", color="0.25")

fig.tight_layout()

def update(i):
    pts.set_offsets(np.c_[k, eigs[i]])
    on = eigs[i] > ACTIVE
    pts.set_sizes(np.where(on, 105, 42))
    pts.set_color([c if o else "0.72" for c, o in zip(colors, on)])
    trail.set_data(betas[: i + 1], N_of_t[: i + 1])
    dot.set_data([betas[i]], [N_of_t[i]])
    readout.set_text(rf"$\beta={betas[i]:.1f}$,   $N={N_of_t[i]}$ of {M}")
    return pts, trail, dot, readout

ani = animation.FuncAnimation(fig, update, frames=len(betas), interval=33, blit=False)
out = os.path.join(HERE, "cascade.mp4")
ani.save(out, writer=animation.FFMpegWriter(fps=30, bitrate=2400))
print("done ->", out)
