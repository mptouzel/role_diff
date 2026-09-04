"""Three-panel animation of the cascade (promotional; not part of the submission).

(a) Role differentiation, unfolding one dimension at a time. Before ignition the
    population is a single cloud on one axis. When axis 1 crosses threshold it
    splits in two; when axis 2 crosses, the second dimension opens and each lobe
    divides again, giving four role combinations; when axis 3 crosses, colour
    carries the third division rather than a third spatial axis.
(b) The instantaneous covariance spectrum, eigenvalues fluctuating about the
    noise floor and lifting off one at a time.
(c) The cascade N(t), with a marker riding the curve. The ramp is linear in
    time, so the marker moves at constant speed and reads as a clock.

Transitions fire on the OBSERVED liftoff of each axis, not on the predicted
beta_c: critical slowing down makes a finite ramp lag its own thresholds.

Rendered twice, since the two vertical scales in (b) say different things:
  cascade.mp4         log v_k    -- every liftoff is a visible event
  cascade_linear.mp4  linear v_k -- the leading axes dominate; the tail is flat
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "ramp_data.npz"))
betas, eigs, z = d["betas"], d["eigs"], d["z_cloud"]
floor, M, n_show = float(d["floor"]), eigs.shape[1], d["z_cloud"].shape[2]

ACTIVE = 2.0 * floor
N_of_t = (eigs > ACTIVE).sum(axis=1)
k = np.arange(1, M + 1)
cmap = plt.get_cmap("viridis")
colors = [cmap(0.08 + 0.84 * i / (M - 1)) for i in range(M)]

# observed liftoff frame of each shown axis: first frame past which it stays up
def liftoff(col):
    up = eigs[:, col] > ACTIVE
    if not up.any():
        return len(betas) + 1
    run = np.flatnonzero(~up)
    return (run[-1] + 1) if len(run) else 0

lift = [liftoff(c) for c in range(n_show)]
OPEN = 55                                  # frames over which a dimension opens
rng = np.random.default_rng(3)
jit = rng.uniform(-1, 1, z.shape[1])       # fixed per-agent jitter for the 1-D phase

# each axis gets its own fixed range: axis 2 is the weaker mode, and on a
# shared scale its split is squeezed to nothing. Units stay absolute, so the
# tick labels still show the hierarchy that panel (b) quantifies.
zlim = [float(np.abs(z[:, :, c]).max()) * 1.08 for c in range(n_show)]

from scipy.ndimage import gaussian_filter


def smooth(H):
    """A 1200-agent histogram is grainy; blur it so the lobes read as lobes."""
    return gaussian_filter(H, sigma=1.5, mode="constant")


def density_stack(zl, jitter):
    """Per-frame density, smoothed in space and then in time.

    A single frame of 1200 agents is a sparse, twinkling histogram. An
    exponential moving average over frames (time constant ~ 0.4 s at 30 fps)
    lets the lobes persist and drift rather than flicker, and a single global
    colour scale keeps the brightness from breathing frame to frame.
    """
    NB = 30
    xe = np.linspace(-zl[0], zl[0], NB + 1)
    ye = np.linspace(-zl[1], zl[1], NB + 1)
    out = np.zeros((len(betas), NB, NB))
    ema = None
    alpha = 0.08
    for i in range(len(betas)):
        w = np.clip((i - lift[1]) / OPEN, 0.0, 1.0) if n_show > 1 else 0.0
        y = (1 - w) * jitter * 0.06 * zl[1] + w * z[i, :, 1]
        H = smooth(np.histogram2d(z[i, :, 0], y, bins=[xe, ye])[0])
        ema = H if ema is None else (1 - alpha) * ema + alpha * H
        out[i] = ema
    return out, xe, ye
mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})

DENS, _XE, _YE = density_stack(zlim, jit)
# Contrast has to adapt: early on the whole population sits in a few bins, later
# it spreads over many. A per-frame scale would flicker, so smooth it in time too.
_pk = np.array([np.percentile(D, 99.5) for D in DENS])
VMAX = np.empty_like(_pk)
_e = _pk[0]
for _i, _v in enumerate(_pk):
    _e = 0.94 * _e + 0.06 * _v
    VMAX[_i] = _e


def render(scale, outname):
    fig, (axP, axS, axC) = plt.subplots(1, 3, figsize=(15.2, 4.3),
                                        width_ratios=[1.05, 1, 1])

    # ---- (a) the population ----------------------------------------------
    im = axP.imshow(DENS[0].T, origin="lower", cmap="Blues", aspect="auto",
                    extent=[-zlim[0], zlim[0], -zlim[1], zlim[1]],
                    interpolation="bicubic", vmin=0, vmax=VMAX[0], zorder=2)
    axP.axhline(0, color="0.85", lw=0.8, zorder=0)
    axP.axvline(0, color="0.85", lw=0.8, zorder=0)
    axP.set_xlim(-zlim[0], zlim[0])
    axP.set_ylim(-zlim[1], zlim[1])
    axP.set_xlabel(r"status on axis 1   $\omega_i \cdot \hat{g}_1$")
    axP.set_ylabel(r"status on axis 2   $\omega_i \cdot \hat{g}_2$")
    axP.set_title("(a)  who takes which role", loc="left", fontsize=12)
    note = axP.text(0.03, 0.965, "", transform=axP.transAxes, fontsize=9.5,
                    va="top", ha="left", color="0.3")
    for s in ("top", "right"):
        axP.spines[s].set_visible(False)

    # ---- (b) the spectrum -------------------------------------------------
    lo = eigs.min() * 0.6 if scale == "log" else 0.0
    axS.axhspan(lo, ACTIVE, color="0.92", zorder=0)
    axS.axhline(ACTIVE, color="0.45", lw=1.0, ls="--", zorder=1)
    axS.text(0.98, 0.96, "dashed: noise floor  " r"$\sigma_\omega^2/2\gamma$",
             transform=axS.transAxes, va="top", ha="right", fontsize=9, color="0.4")
    pts = axS.scatter(k, eigs[0], s=70, c=colors, edgecolor="0.25", linewidth=0.6, zorder=3)
    axS.set_yscale(scale)
    axS.set_xlim(0.3, M + 0.7)
    axS.set_xticks(np.arange(2, M + 1, 2))
    axS.set_ylim(lo, eigs.max() * (1.8 if scale == "log" else 1.05))
    axS.set_xlabel(r"role axis  $k$")
    axS.set_ylabel(r"covariance eigenvalue  $v_k$")
    axS.set_title("(b)  the population's role structure", loc="left", fontsize=12)
    for s in ("top", "right"):
        axS.spines[s].set_visible(False)

    # ---- (c) the cascade ---------------------------------------------------
    axC.step(betas, N_of_t, where="post", color="0.88", lw=1.6, zorder=1)
    trail, = axC.step([], [], where="post", color="tab:blue", lw=2.4, zorder=2)
    dot, = axC.plot([], [], "o", ms=9, color="tab:blue", mec="white", mew=1.4, zorder=3)
    axC.set_xlim(betas[0], betas[-1])
    axC.set_ylim(-0.4, M + 0.6)
    axC.set_xlabel(r"feedback strength  $\beta(t)$")
    axC.set_ylabel(r"active role axes  $N(t)$")
    axC.set_title("(c)  the cascade", loc="left", fontsize=12)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)
    readout = axC.text(0.03, 0.94, "", transform=axC.transAxes, fontsize=10,
                       va="top", ha="left", color="0.25")
    fig.tight_layout()

    def update(i):
        # (a) how far the second dimension has opened, and whether axis 3 colours
        w = np.clip((i - lift[1]) / OPEN, 0.0, 1.0) if n_show > 1 else 0.0
        im.set_data(DENS[i].T)
        im.set_clim(0, VMAX[i])
        if i < lift[1]:
            note.set_text("one role axis:  two roles")
        elif n_show > 2 and i >= lift[2]:
            note.set_text("three role axes split;  two of them shown")
        else:
            note.set_text("two role axes:  four role combinations")
        # (b), (c)
        pts.set_offsets(np.c_[k, eigs[i]])
        on = eigs[i] > ACTIVE
        pts.set_sizes(np.where(on, 105, 42))
        pts.set_color([c if o else "0.72" for c, o in zip(colors, on)])
        trail.set_data(betas[: i + 1], N_of_t[: i + 1])
        dot.set_data([betas[i]], [N_of_t[i]])
        readout.set_text(rf"$\beta={betas[i]:.1f}$,   $N={N_of_t[i]}$ of {M}")
        return im, pts, trail, dot, readout, note

    ani = animation.FuncAnimation(fig, update, frames=len(betas), interval=33, blit=False)
    ani.save(os.path.join(HERE, outname), writer=animation.FFMpegWriter(fps=30, bitrate=3000))
    plt.close(fig)
    print("done ->", outname)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    print("observed liftoff frames (axes 1..%d): %s" % (n_show, lift[:n_show]))
    if which in ("both", "log"):
        render("log", "cascade.mp4")
    if which in ("both", "linear"):
        render("linear", "cascade_linear.mp4")
