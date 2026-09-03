"""Saddle phase portrait of the two-schema replicator, as three small multiples
sharing one (u1,u2) frame in units of the per-schema ceiling u_sat=1: the
budget U_cap grows left to right (condensate -> saturated cluster -> Theta=1),
so the pinned (saddle) point U_cap/2 climbs the u1=u2 diagonal panel to panel,
reaching exactly the u_sat corner (1,1) when Theta=1 (U_cap=M*u_sat)."""
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
import matplotlib.patheffects as pe

# Figure is built at its true final print size (\columnwidth = 246.0pt in this
# revtex4-2 aps,prx two-column layout, from \the\columnwidth) so that a matplotlib
# fontsize of N is literally N pt on the printed page -- no post-hoc scale factor.
# Body text (\normalsize) in this document is 10pt, so panel-title fontsize=10
# matches it exactly.
COLUMNWIDTH_IN = 246.0 / 72.0
_SCALE = COLUMNWIDTH_IN / 10.2  # shrink from the old 10.2in working canvas
# bbox_inches="tight" crops the saved page below the design canvas, so LaTeX's
# width=\columnwidth then scales it back up; correct for that measured shrinkage
# (calibrated empirically: design canvas -> saved page was a 246.0/222.468 undershoot)
# so the *rendered* title size lands on \normalsize (10pt), not ~11pt.
_CORR = 246.0 / 222.468
_SCALE *= _CORR
BODY_PT = 10 * _CORR

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8*_CORR, "axes.labelsize": 9*_CORR, "axes.linewidth": 0.9*_SCALE, "figure.dpi": 300,
    "xtick.labelsize": 8*_CORR, "ytick.labelsize": 8*_CORR})

HALO = [pe.withStroke(linewidth=5*_SCALE, foreground="white")]

u_sat, M = 1.0, 2
XP = 1.3  # shared axis range for every panel, in units of u_sat

g0, n_bar, b = 1.4, 4, 1.5
def vel(u1, u2, S):
    # sdot: fixed intrinsic growth g0 (stand-in for 2(alpha*lambda-kappa), independent
    # of S) away from the origin, arrested only near the ceiling s->S -- a one-sided
    # barrier, not a two-sided spring, so phi'~0 stays negligible while slack.
    s, d = u1 + u2, u1 - u2
    sdot = g0 * (1 - (s / S)**n_bar)
    ddot = b * d * (1 - (d / S)**2)
    return 0.5*(sdot + ddot), 0.5*(sdot - ddot)

U0 = (0.055, 0.040)   # one seeded start, slightly favouring schema 1, shared by all panels

def trajectory(S, u0=U0, dt=0.002, nmax=400000):
    """Integrate the strength dynamics from u0 until the ceiling or a fixed point."""
    u = np.array(u0, float)
    pts = [u.copy()]
    for _ in range(nmax):
        v = np.array(vel(u[0], u[1], S))
        if not np.isfinite(v).all() or np.hypot(*v) < 1e-4:
            break
        u = u + dt * v
        pts.append(u.copy())
        if u[0] >= u_sat or u[1] >= u_sat or u.max() > XP:
            break
    return np.array(pts)

panels = [
    dict(S=0.5, kind="condensate", title=r"(a) $\Theta=4$", sub="low saturation,\ncondensed state"),
    dict(S=1.5, kind="saturated",  title=r"(b) $\Theta=4/3$", sub="moderate saturation,\ncapped state"),
    dict(S=2.0, kind="theta1",     title=r"(c) $\Theta=1$", sub="high saturation,\nnear pinned state"),
]

g = np.linspace(0.02, XP, 80)
G1, G2 = np.meshgrid(g, g)

fig, axes = plt.subplots(1, 3, figsize=(10.2*_SCALE, 4.8*_SCALE), sharex=True, sharey=True)

for ax, p in zip(axes, panels):
    S = p["S"]
    V1, V2 = vel(G1, G2, S)
    # arrowsize sets an absolute point size (not linewidth/axes-relative), so it
    # needs the same _SCALE correction applied explicitly to stay proportionate
    # now that the figure is built at true final print size.
    ax.streamplot(g, g, V1, V2, color="0.78", linewidth=1.0*_SCALE, density=0.75, arrowsize=1.1*_SCALE)

    # excluded region: no schema strength can exceed u_sat=1
    ax.axvspan(u_sat, XP + 0.2, color="0.6", alpha=0.30, zorder=2, lw=0)
    ax.axhspan(u_sat, XP + 0.2, color="0.6", alpha=0.30, zorder=2, lw=0)

    # manifolds (matplotlib clips to the axes automatically)
    ax.plot([0, XP], [0, XP], color="tab:green", lw=2.6*_SCALE, zorder=4)
    ax.plot([0, S], [S, 0], color="tab:red", lw=2.6*_SCALE, ls=(0,(5,2)), zorder=4)

    tr = trajectory(S)
    # stop short of the terminal marker so the arrowhead is not buried under it
    keep = tr[np.linalg.norm(tr - tr[-1], axis=1) > 0.085]
    if len(keep) < 8:
        keep = tr[:-1]
    ax.plot(keep[:, 0], keep[:, 1], color="k", lw=3.0*_SCALE, zorder=5, solid_capstyle="round")
    ax.plot(*tr[0], "o", color="k", ms=3.2*_SCALE, zorder=5)
    # arrowhead on the final segment, giving the trajectory its direction of travel
    ax.annotate("", xy=keep[-1], xytext=keep[max(0, len(keep) - 10)],
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0,
                                mutation_scale=30*_SCALE, shrinkA=0, shrinkB=0), zorder=5)

    sad = S/2
    ax.plot(sad, sad, "o", mfc="white", mec="k", mew=1.9*_SCALE, ms=10*_SCALE, zorder=6)

    if p["kind"] == "condensate":
        ax.plot([S, 0], [0, S], "o", color="tab:red", ms=8.5*_SCALE, zorder=6, clip_on=False)
        # label the budget where it meets the axis, in the x tick-label position
        # under the red dot. A real tick is not usable: sharex=True would put it
        # on all three panels, whose S differ.
        ax.annotate(r"$U_{\mathrm{cap}}$", xy=(S, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -(mpl.rcParams["xtick.major.size"]
                                 + mpl.rcParams["xtick.major.pad"])),
                    textcoords="offset points", ha="center", va="top",
                    color="tab:red", fontsize=mpl.rcParams["xtick.labelsize"],
                    annotation_clip=False, zorder=10)
    elif p["kind"] in ("saturated", "theta1"):
        # ceiling arrest: where this trajectory is stopped by u_sat, i.e. its endpoint
        ax.plot(*tr[-1], "s", mfc="none", mec="tab:red", mew=2.1*_SCALE, ms=11*_SCALE, zorder=7)

    ax.set_xlim(0, XP); ax.set_ylim(0, XP); ax.set_aspect("equal")
    ax.set_xticks([0, u_sat]); ax.set_xticklabels(["0", "1"])
    ax.set_yticks([0, u_sat]); ax.set_yticklabels(["0", "1"])
    ax.set_title(p["title"], fontsize=BODY_PT, pad=50*_SCALE)  # matches \normalsize body text (10pt)
    ax.text(0.5, 1.01, p["sub"], transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.2*_CORR, linespacing=1.2)

axes[0].set_ylabel(r"$u_2$ (units of $u_{\mathrm{cap}}$)", fontsize=9*_CORR)
fig.supxlabel(r"$u_1$ (units of $u_{\mathrm{cap}}$)", fontsize=9*_CORR)
fig.tight_layout()
fig.subplots_adjust(wspace=0.03)
fig.savefig(_out("fig_pinning_saddle.png"), dpi=600, bbox_inches="tight")
fig.savefig(_out("fig_pinning_saddle.pdf"), bbox_inches="tight")
print("done")
