"""Three-dimensional geometry of one schema (promotional; not part of the
submission).

Four acts, 78 s at 30 fps, from promo/geometry_data.npz:

  1. the two sets of objects -- a cloud of omega-identities and the three
     schemas of the Fig. 1 repertoire, drawn as axes whose lengths are their
     strengths |g_m|;
  2. one encounter in slow motion -- two agents scored by their projection onto
     ghat_1, the difference Delta read through channel noise, the roles that
     follow, and the action written back along the axis that was read. The
     second encounter is a misread, which is what the error rate
     P_e = Phi(-|Delta|/sigma_read) of Eq. (3) means;
  3. the beta ramp -- the cloud stretches along ghat_1 and divides into two role
     lobes as beta crosses beta_c^(1) = 2.6;
  4. beta_c^(2) = 4.0 -- the second schema is above threshold and broadens the
     cloud along ghat_2 without dividing it. Measured on this repertoire at
     N = 3000: axis-2 marginal dip 0.97 at beta = 12 and 0.55 at beta = 20,
     against 0.04 for axis 1, with the angular contrast of the (z_1, z_2)
     density at 34-43 throughout. Two lobes, not four groups, and not the ring
     of sim_ring.py.

Projection is orthographic and hand-rolled rather than mplot3d: the camera
orbits at fixed elevation, points are painter-sorted and sized by depth, and the
view zooms out through act 3 so the growing cloud stays in frame.

Renders need /home/mptouzel/miniconda3/bin/python3 (the venv has no matplotlib)
and FFMpegWriter with codec libopenh264 (no libx264 in this environment).

Writes promo/geometry.mp4.
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

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "geometry_data.npz"))
cloud, betas, dips = D["cloud"], D["betas"], D["dips"]
omega_sub = D["omega_sub"]
strength, beta_c = D["strength"], D["beta_c"]
sigma_obs, beta_sub = float(D["sigma_obs"]), float(D["beta_sub"])
N = omega_sub.shape[0]

def _ema(v, a=0.06):
    out = np.empty_like(v, dtype=float)
    e = float(v[0])
    for i, x in enumerate(v):
        e = (1 - a) * e + a * float(x)
        out[i] = e
    return out


DIPS = np.vstack([_ema(dips[:, m]) for m in range(3)]).T     # density at 0 / peak
# Per-frame ceiling for the marginals, smoothed in time: a fixed one is set by
# the tight subcritical peak and flattens every later frame to nothing.
HMAX = 1.25 * _ema(np.array([max(np.histogram(cloud[i][:, m], bins=23,
                                              range=(-2.7, 2.7))[0].max()
                                 for m in (0, 1)) for i in range(len(betas))]), 0.05)

STRIDE = int(os.environ.get("STRIDE", "1"))        # >1 renders a fast preview
FPS = 30

# Act 2 is cut by beat rather than by a fixed fraction of the act: every text has
# to be readable, which is about 3.5 s for a caption plus its line of gloss. The
# first encounter teaches the beats and gets that everywhere; the second repeats
# four of them and can move, so its time is spent on the two that are new, the
# misread and what the misread costs.
PHASES = ("meet", "score", "delta", "read", "roles", "write")
PH_ENC1 = (100, 100, 100, 110, 110, 110)           # 21.0 s
PH_ENC2 = (55, 55, 55, 80, 125, 100)               # 15.7 s
CUM = [np.cumsum(np.array(v)) for v in (PH_ENC1, PH_ENC2)]

A1 = 300                                           # 10.0 s, three texts
A2 = int(CUM[0][-1] + CUM[1][-1])
A3 = len(betas)
TAIL = 60                                          # hold the closing frame 2 s
NF = A1 + A2 + A3 + TAIL

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 13})

INK, DIM = "#1a1a1a", "#9a9a9a"
GO, DEFER = "#D62728", "#1F77B4"                   # Go / Defer, as the class colours
SCHEMA = "#2CA02C"


# --- staged encounter -------------------------------------------------------
# Two agents from the subcritical snapshot, chosen for a legible geometry: a
# clear gap on axis 1 and small components off it. Encounter 2 is drawn until
# the partner's read comes out on the wrong side, so the clip shows a genuine
# sample of the channel noise rather than a drawn cartoon of one.
def pick_pair(rng, want_misread):
    z = omega_sub[:, 0]
    off = np.linalg.norm(omega_sub[:, 1:], axis=1)
    cand_hi = np.where((z > np.percentile(z, 82)) & (off < np.percentile(off, 45)))[0]
    cand_lo = np.where((z < np.percentile(z, 18)) & (off < np.percentile(off, 45)))[0]
    for _ in range(4000):
        i, j = rng.choice(cand_hi), rng.choice(cand_lo)
        delta = (omega_sub[i] - omega_sub[j]) @ (np.eye(3)[0] * strength[0])
        s_i = delta + sigma_obs * rng.standard_normal()
        s_j = -delta + sigma_obs * rng.standard_normal()
        # i should go and j should defer; a misread is j reading Go as well.
        ok = (s_i > 0) and ((s_j > 0) == want_misread)
        if ok:
            return dict(i=int(i), j=int(j), delta=float(delta), s_i=float(s_i),
                        s_j=float(s_j), a_i=int(np.sign(s_i)), a_j=int(np.sign(s_j)))
    raise RuntimeError("no pair found")


_rng = np.random.default_rng(4)
ENCS = [pick_pair(_rng, False), pick_pair(_rng, True)]
print("staged encounters:", ENCS)


# --- camera -----------------------------------------------------------------
def basis(az, el):
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    f = np.array([ce * ca, ce * sa, se])           # toward the camera
    r = np.array([-sa, ca, 0.0])
    u = np.array([-se * ca, -se * sa, ce])
    return r, u, f


def project(P, az, el):
    r, u, f = basis(az, el)
    P = np.atleast_2d(P)
    return P @ r, P @ u, P @ f


def half_extent(k):
    """World half-size in view. Zoom out through act 3 as the cloud grows."""
    if k < A1 + A2:
        return 0.55
    t = (k - A1 - A2) / max(A3 - 1, 1)
    return 0.55 + (2.4 - 0.55) * min(1.0, t / 0.55) ** 0.8


# The camera rocks about az = -66 deg rather than orbiting. At az = -90 the
# world x axis maps exactly onto the screen horizontal; the story is the
# projection onto ghat_1 and the split along it, so that axis must never go
# edge-on. The 24 deg rock keeps parallax without foreshortening it, and leaves
# ghat_2 a screen-x component of cos(az) = 0.25-0.65.
def azimuth(k):
    return np.deg2rad(-60.0 + 18.0 * np.sin(2 * np.pi * k / 620.0))


def elevation(k):
    return np.deg2rad(17.0 + 5.0 * np.sin(2 * np.pi * k / 870.0 + 1.1))

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor("white")
AXBOX_WIDE, AXBOX_INSET = [0.02, 0.15, 0.94, 0.83], [0.0, 0.15, 0.72, 0.83]
ax = fig.add_axes(AXBOX_INSET)
ax.set_aspect("equal"); ax.axis("off")
axI = fig.add_axes([0.765, 0.30, 0.215, 0.34])     # act 2: the read
axH1 = fig.add_axes([0.765, 0.46, 0.215, 0.26])    # act 3: marginal on axis 1
axH2 = fig.add_axes([0.765, 0.11, 0.215, 0.26])    # act 3: marginal on axis 2
cap = fig.text(0.5, 0.078, "", ha="center", va="center", fontsize=16, color=INK)
sub = fig.text(0.5, 0.032, "", ha="center", va="center", fontsize=12, color=DIM)
hud = fig.text(0.985, 0.955, "", ha="right", va="top", fontsize=13, color=INK)


def arrow(p0, p1, az, el, color, lw, label=None, zorder=5, alpha=1.0):
    x, y, _ = project(np.vstack([p0, p1]), az, el)
    ax.plot(x, y, "-", color=color, lw=lw, zorder=zorder, alpha=alpha,
            solid_capstyle="round")
    dx, dy = x[1] - x[0], y[1] - y[0]
    n = np.hypot(dx, dy)
    if n > 1e-9:
        ax.plot([x[1]], [y[1]], marker=(3, 0, np.degrees(np.arctan2(dy, dx)) - 90),
                ms=6.0 + 3.2 * lw, color=color, zorder=zorder, alpha=alpha)
    if label:
        pad = 0.10 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / 3.0
        ax.text(x[1] + pad * dx / max(n, 1e-9), y[1] + pad * dy / max(n, 1e-9),
                label, color=color, fontsize=15, zorder=zorder + 1, alpha=alpha,
                ha="center", va="center")


def draw_axes(az, el, L, show=(True, True, True), alpha=1.0):
    """The three schemas, as axes whose length is proportional to |g_m|."""
    for m in range(3):
        if not show[m]:
            continue
        e = np.eye(3)[m] * L * 0.82 * (strength[m] / strength[0])
        a = alpha * (1.0 if m == 0 else 0.55)
        x, y, _ = project(np.vstack([-e, np.zeros(3)]), az, el)
        ax.plot(x, y, "-", color=SCHEMA, lw=1.0, alpha=0.30 * a, zorder=2)
        arrow(np.zeros(3), e, az, el, SCHEMA, 2.4 if m == 0 else 1.6,
              label=rf"$\hat g_{m+1}$", zorder=6, alpha=a)


def draw_points(P, az, el, L, color=None, size=26, alpha=0.75, hide=()):
    x, y, dep = project(P, az, el)
    keep = np.ones(len(x), bool)
    for h in hide:
        keep[h] = False
    o = np.argsort(dep[keep])
    xs, ys, ds = x[keep][o], y[keep][o], dep[keep][o]
    dn = (ds - ds.min()) / max(ds.ptp(), 1e-9)
    if isinstance(color, np.ndarray):
        c = color[keep][o]
    elif color is not None:
        c = color
    else:
        c = np.where(P[keep][o][:, 0] > 0, GO, DEFER)
    ax.scatter(xs, ys, s=size * (0.55 + 0.75 * dn), c=c, lw=0,
               alpha=alpha, zorder=3)


def set_view(L, wide):
    box = AXBOX_WIDE if wide else AXBOX_INSET
    ratio = (box[2] * 12.8) / (box[3] * 7.2)
    ax.set_xlim(-ratio * L, ratio * L)
    ax.set_ylim(-L, L)


def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


# --- the four acts ----------------------------------------------------------
def act1(k, az, el, L):
    t = k / A1
    draw_points(omega_sub, az, el, L, color=INK, size=24, alpha=0.70)
    n_ax = 0 if t < 0.36 else (1 if t < 0.67 else 3)
    draw_axes(az, el, L, show=(n_ax > 0, n_ax > 1, n_ax > 2))
    for a in (axI, axH1, axH2):
        a.set_visible(False)
    if t < 0.36:
        cap.set_text(r"Every agent carries an identity: a point $\omega_i$")
        sub.set_text("It is a running summary of the roles that agent has taken.")
    elif t < 0.67:
        cap.set_text(r"A schema is a rule for scoring identities")
        sub.set_text(r"In context 1 it gives agent $i$ the number $\omega_i\cdot g_1$. "
                     r"It reads identities; it is not one of them.")
    else:
        cap.set_text(r"One schema per game context, here three")
        sub.set_text(r"Arrow length is the schema's strength $|g_m|$: "
                     r"how strongly that rule is institutionalized.")
    hud.set_text("")


def act2(k, az, el, L):
    n = 0 if k < CUM[0][-1] else 1
    kk = k if n == 0 else k - int(CUM[0][-1])
    ph = int(np.searchsorted(CUM[n], kk, side="right"))
    ph = min(ph, len(PHASES) - 1)
    start = 0 if ph == 0 else int(CUM[n][ph - 1])
    frac = (kk - start) / max(int(CUM[n][ph]) - start, 1)
    stage = PHASES.index                                  # name -> order
    at = lambda name: ph >= stage(name)

    e = ENCS[n]
    i, j = e["i"], e["j"]
    Pi, Pj = omega_sub[i], omega_sub[j]
    zi, zj = Pi[0], Pj[0]
    faded = ph > 0 or frac > 0.55
    draw_points(omega_sub, az, el, L, color=DIM, size=18,
                alpha=0.16 if faded else 0.55, hide=(i, j))
    draw_axes(az, el, L, show=(True, True, True), alpha=0.5 if faded else 1.0)

    roled = at("roles")
    ci = (GO if e["a_i"] > 0 else DEFER) if roled else INK
    cj = (GO if e["a_j"] > 0 else DEFER) if roled else INK
    writing = at("write")
    shift = 0.30 * frac if writing else 0.0
    Qi = Pi + np.array([shift * e["a_i"], 0, 0])
    Qj = Pj + np.array([shift * e["a_j"], 0, 0])
    for P, Q, c, lab in ((Pi, Qi, ci, r"$i$"), (Pj, Qj, cj, r"$j$")):
        x, y, _ = project(Q, az, el)
        ax.scatter(x, y, s=150, c=c, lw=0, zorder=8)
        ax.text(x[0], y[0] + 0.13 * L, lab, color=c, ha="center", va="bottom",
                fontsize=16, zorder=9)
        if writing:
            arrow(P, Q, az, el, c, 1.8, zorder=7)
    if at("score") and not writing:                # perpendiculars to the axis
        for P, z, c in ((Pi, zi, ci), (Pj, zj, cj)):
            foot = np.array([z, 0.0, 0.0])
            x, y, _ = project(np.vstack([P, foot]), az, el)
            ax.plot(x, y, "--", color=c, lw=1.2, alpha=0.8, zorder=4)
            fx, fy, _ = project(foot, az, el)
            ax.scatter(fx, fy, s=42, facecolor="white", edgecolor=c, lw=1.6, zorder=8)
    if at("delta") and not writing:                # the gap on the axis
        x, y, _ = project(np.vstack([[zj, 0, 0], [zi, 0, 0]]), az, el)
        ax.plot(x, y, "-", color=INK, lw=3.4, alpha=0.9, zorder=7)
        ax.text(x.mean(), y.mean() - 0.11 * L, r"$\Delta$", color=INK,
                ha="center", fontsize=16, zorder=9)

    axH1.set_visible(False); axH2.set_visible(False)
    axI.set_visible(at("read"))
    if at("read"):
        axI.clear()
        xs = np.linspace(e["delta"] - 3.2 * sigma_obs, e["delta"] + 3.2 * sigma_obs, 400)
        axI.plot(xs, gauss(xs, e["delta"], sigma_obs), color=INK, lw=1.6)
        axI.fill_between(xs, 0, gauss(xs, e["delta"], sigma_obs), where=xs < 0,
                         color=GO, alpha=0.30, lw=0)
        axI.axvline(0, color=DIM, lw=1.0)
        axI.axvline(e["delta"], color=INK, lw=1.0, ls=":")
        if ph > stage("read") or frac > 0.55:      # the sample this agent drew
            axI.axvline(e["s_i"], color=ci, lw=2.2)
        axI.set_yticks([]); axI.set_xlabel(r"read $s=\Delta+\sigma_{\rm read}\xi$",
                                           fontsize=11.5)
        axI.set_title(r"$P_e=\Phi(-|\Delta|/\sigma_{\rm read})$", fontsize=11.5, pad=6)
        for sp in ("top", "right", "left"):
            axI.spines[sp].set_visible(False)
        axI.tick_params(labelsize=10)

    name = PHASES[ph]
    if name == "meet":
        cap.set_text(r"One encounter, in context 1")
        sub.set_text("Two agents meet. Neither has been told what to do.")
    elif name == "score":
        cap.set_text(r"The schema scores them both")
        sub.set_text(r"Each identity projects onto $\hat g_1$.")
    elif name == "delta":
        cap.set_text(r"Only the difference matters")
        sub.set_text(r"$\Delta=(\omega_i-\omega_j)\cdot g_1$ is what the encounter turns on.")
    elif name == "read":
        cap.set_text(r"Each agent reads $\Delta$ through a noisy channel")
        sub.set_text(r"The wider the gap, the rarer the misread.")
    elif name == "roles":
        if n == 0:
            cap.set_text(r"Higher goes, lower defers")
            sub.set_text("Complementary roles pay 6 and 2, against 2.5 at the "
                         "mixed Nash equilibrium.")
        else:
            cap.set_text(r"This read failed: both go")
            sub.set_text(r"A collision pays $-1$ each. The payoff above Nash is "
                         r"$W=w_0(1-2P_e)^2$.")
    else:
        cap.set_text(r"The action is written back along the axis that was read")
        sub.set_text(r"Kick $\beta\langle a\rangle_1 g_1$: read axis and write "
                     r"axis are the same axis.")
    hud.set_text("")


def role_colors(P, w):
    """Ink while the population is unimodal, role colours as the axis divides.

    w is driven by the measured marginal dip, so the clip never shows two
    colours before the density has two modes.
    """
    ink = np.array(mpl.colors.to_rgb(INK))
    tgt = np.where((P[:, 0] > 0)[:, None], np.array(mpl.colors.to_rgb(GO)),
                   np.array(mpl.colors.to_rgb(DEFER)))
    return (1 - w) * ink + w * tgt


def marginal_panel(axh, x, k, m, title):
    axh.clear()
    axh.hist(x, bins=23, range=(-2.7, 2.7), color=DIM, alpha=0.9)
    axh.set_ylim(0, HMAX[k])
    axh.set_yticks([])
    axh.set_title(title, fontsize=11.5, pad=4)
    axh.text(0.02, 0.86, f"dip {DIPS[k, m]:.2f}", transform=axh.transAxes,
             fontsize=11, color=INK)
    for sp in ("top", "right", "left"):
        axh.spines[sp].set_visible(False)
    axh.tick_params(labelsize=10)


def act3(k, az, el, L):
    b = betas[k]
    w = float(np.clip(1.0 - DIPS[k, 0] / 0.9, 0.0, 1.0))
    draw_points(cloud[k], az, el, L, color=role_colors(cloud[k], w),
                size=22, alpha=0.78)
    draw_axes(az, el, L, show=(True, True, True))
    axI.set_visible(False)
    axH1.set_visible(True); axH2.set_visible(True)
    marginal_panel(axH1, cloud[k][:, 0], k, 0, r"on axis 1   $\omega_i\cdot\hat g_1$")
    marginal_panel(axH2, cloud[k][:, 1], k, 1, r"on axis 2   $\omega_i\cdot\hat g_2$")
    crossed = int((b > beta_c).sum())
    hud.set_text(rf"$\beta={b:.1f}$" + "\n" +
                 rf"$\beta_c^{{(1,2,3)}}={beta_c[0]:.1f},\,{beta_c[1]:.1f},\,{beta_c[2]:.1f}$"
                 + "\n" + rf"thresholds crossed: {crossed}")
    if b < beta_c[0]:
        cap.set_text(r"Now every agent, every encounter, with $\beta$ rising")
        sub.set_text(r"Below threshold the writes are washed out by identity noise.")
    elif DIPS[k, 0] > 0.45:
        cap.set_text(r"$\beta_c^{(1)}=2.6$: the loop gain on axis 1 reaches one")
        sub.set_text("Each write now survives long enough to bias the next read.")
    elif b < beta_c[1] * 1.7:
        cap.set_text(r"The cloud divides along $\hat g_1$")
        sub.set_text("Two complementary roles, self-sustaining. Nobody assigned them.")
    elif DIPS[k, 1] > 0.62:
        cap.set_text(r"$\beta_c^{(2)}=4.0$: the second schema is over threshold too")
        sub.set_text(r"It widens the cloud along $\hat g_2$ without dividing it.")
    else:
        cap.set_text("A second role axis is not yet a second role")
        sub.set_text(rf"Axis 1 divided at $\beta=2.6$ (dip {DIPS[k, 0]:.2f}); axis 2 "
                     rf"crossed at 4.0 and is at dip {DIPS[k, 1]:.2f} by $\beta={b:.0f}$.")


def draw(k):
    wide = k < A1
    cx = 0.5 if wide else 0.40
    cap.set_position((cx, 0.078)); sub.set_position((cx, 0.032))
    ax.set_position(AXBOX_WIDE if wide else AXBOX_INSET)
    ax.clear(); ax.set_aspect("equal"); ax.axis("off")
    az, el, L = azimuth(k), elevation(k), half_extent(k)
    set_view(L, wide)
    if k < A1:
        act1(k, az, el, L)
    elif k < A1 + A2:
        act2(k - A1, az, el, L)
    else:
        act3(min(k - A1 - A2, A3 - 1), az, el, L)
    return []


DUMP = os.environ.get("DUMP")                       # e.g. DUMP=60,340,900 for stills
if DUMP:
    for k in (int(v) for v in DUMP.split(",")):
        draw(k)
        fig.savefig(os.path.join(HERE, f"_frame_{k:04d}.png"), dpi=100)
        print("still ->", f"_frame_{k:04d}.png")
    raise SystemExit

frames = range(0, NF, STRIDE)
# Render to a scratch directory and remux in, as make_ring_animation.py does:
# writing straight into this Dropbox tree races the sync client, which snapshots
# the half-written file as a conflicted copy.
anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
name = "geometry.mp4" if STRIDE == 1 else "geometry_preview.mp4"
tmp = tempfile.mkdtemp(prefix="geometry_render_")
try:
    scratch = os.path.join(tmp, name)
    anim.save(scratch, writer=animation.FFMpegWriter(fps=FPS, codec="libopenh264",
                                                     bitrate=3000))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", scratch, "-c", "copy",
                    "-movflags", "+faststart", os.path.join(HERE, name)], check=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
plt.close(fig)
print("done -> promo/" + name, f"({len(list(frames))} frames)")
