"""Characterization figure (2 panels), shared axes (c/gamma, M_cap/M):
   (a) normalized role capacity + cascade-morphology glyphs  -- what the endpoint
       CAN sustain (Theta-only, three equal bands);
   (b) observed participation ratio D_obs/M -- what the spectrum REALIZES.
The gap between (a)'s band and (b)'s field is the distance from stationarity."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "axes.linewidth": 0.9, "figure.dpi": 150})

rng = np.random.default_rng(3)
M, S = 48, 1.0
u0 = 1e-4 * S / M
jit_sat = 1 + 0.01 * rng.normal(size=M)
jit_fr = 1 + 0.02 * rng.normal(size=M)

def spectrum(r, Theta):
    u_sat = Theta * S / M
    if Theta <= 1.0:
        L = max(np.log(u_sat / u0), 1e-6)
        n_sat = max(0.0, M - r * L)
        k = np.arange(1, M + 1, dtype=float)
        u = np.where(k <= n_sat, u_sat * jit_sat[:M], u_sat * np.exp(-(k - n_sat) / r))
        return np.maximum(u, u0)
    elif Theta <= M:
        a = 1.0 / r
        k = np.arange(1, M + 1, dtype=float)
        lo, hi = 1e-12 * S, 1e4 * S
        def total(C): return np.sum(np.minimum(u_sat, C * k ** (-a)))
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            hi, lo = (mid, lo) if total(mid) > S else (hi, mid)
        C = np.sqrt(lo * hi)
        return np.maximum(np.minimum(u_sat * jit_sat, C * k ** (-a)), 1e-30)
    else:
        u = np.empty(M); u[0] = S - (M - 1) * u0; u[1:] = u0 * jit_fr[1:M]
        return u

def participation(u): return (u.sum() ** 2) / np.sum(u ** 2)

b_hi, b_lo = 1.0, 1.0 / M
lo, hi = np.log10(b_lo), np.log10(b_hi); gap = hi - lo
ymin, ymax = 10 ** (lo - gap), 10 ** (hi + gap)
rs = np.logspace(np.log10(0.1), np.log10(10), 240)
ys = np.logspace(np.log10(ymin), np.log10(ymax), 240)   # y = M_cap/M

# panel (b) field: D_obs/M at each (c/gamma, M_cap/M);  Theta = 1/y
PRn = np.empty((len(ys), len(rs)))
for iy, y in enumerate(ys):
    Th = 1.0 / y
    for ix, r in enumerate(rs):
        PRn[iy, ix] = participation(spectrum(r, Th)) / M

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.4), sharey=True)

def frame(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(rs[0], rs[-1]); ax.set_ylim(ymin, ymax)
    ax.axhline(b_hi, color="k", lw=1.1, ls="--")
    ax.axhline(b_lo, color="k", lw=1.1, ls="--")
    ax.plot([1, 1], [b_lo, b_hi], color="k", lw=1.0, ls=":")
    ax.set_xlabel(r"activations per compounding time  $r=r_{\mathrm{act}}/g$")

# ---- panel (a): capacity bands + glyphs ----
axA.axhspan(b_hi, ymax, color="#e7efe9", zorder=0)
axA.axhspan(b_lo, b_hi, color="#f3efe6", zorder=0)
axA.axhspan(ymin, b_lo, color="#efe9e9", zorder=0)
frame(axA)
tag = dict(fontsize=11, ha="left", va="center",
           bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92))
axA.text(0.115, b_hi, r"$M_{\mathrm{cap}}=M$  (all contexts)", **tag)
axA.text(0.115, b_lo, r"$M_{\mathrm{cap}}=1$  (single context)", **tag)
axA.text(1.12, b_lo * 2.4, r"$a=1$", fontsize=7.5, ha="left",
         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92))

def glyph(ax, cx, cy, kind, color, label):
    ia = ax.inset_axes([cx, cy, 0.145, 0.095], transform=ax.transAxes)
    x = np.linspace(0, 1, 200)
    y = {"step": (x > 0.42).astype(float), "log": np.log1p(20 * x) / np.log(21),
         "concave": x ** 0.5, "convex": x ** 2.6,
         "spike": 0.08 * (x > 0.12) + 0.90 * (x > 0.82)}[kind]
    ia.plot(x, y, color=color, lw=1.7)
    ia.set_xticks([]); ia.set_yticks([]); ia.set_ylim(-0.12, 1.15)
    for s in ia.spines.values(): s.set_color("0.5"); s.set_linewidth(0.6)
    ia.patch.set_alpha(0.95)
    ax.text(cx + 0.065, cy - 0.028, label, transform=ax.transAxes, ha="center",
            va="top", fontsize=12, fontweight="bold", color="black")

glyph(axA, 0.09, 0.80, "step", "0.25", "avalanche")
glyph(axA, 0.74, 0.80, "log", "tab:blue", "logarithmic")
glyph(axA, 0.09, 0.46, "concave", "tab:green", "decelerating")
glyph(axA, 0.70, 0.46, "convex", "tab:red", "accelerating")
glyph(axA, 0.38, 0.135, "spike", "tab:purple", "spike + gap")

def mixed_glyph(ax, cx, cy, r_ex, Theta_ex, label):
    ia = ax.inset_axes([cx, cy, 0.145, 0.095], transform=ax.transAxes)
    u = spectrum(r_ex, Theta_ex)          # already sorted descending by rank
    beta_proxy = 1.0 / u                  # ascending: activation order matches rank order
    beta_proxy.sort()
    N = np.arange(1, M + 1)
    ia.step(beta_proxy, N, where="post", color="0.2", lw=1.5)
    ia.set_xscale("log")
    ia.set_xticks([]); ia.set_yticks([])
    for s in ia.spines.values(): s.set_color("0.5"); s.set_linewidth(0.6)
    ia.patch.set_alpha(0.95)
    ax.text(cx + 0.0725, cy - 0.028, label, transform=ax.transAxes, ha="center",
            va="top", fontsize=12, fontweight="bold", color="black")

mixed_glyph(axA, 0.40, 0.80, r_ex=3.0, Theta_ex=0.5, label="mixed")

# ---- E1|E2 mixed region: saturated cluster + geometric tail coexist for r < r*(Theta) ----
def r_star(Theta):
    u_sat = Theta * S / M
    L = max(np.log(u_sat / u0), 1e-6)
    return M / L

y_thick = np.logspace(np.log10(b_hi), np.log10(ymax), 200)   # thick band only: Theta<=1
Th_thick = 1.0 / y_thick
r_boundary = np.array([r_star(Th) for Th in Th_thick])

# shade the coexistence region (r < r*(Theta)) with a light hatch, left of the boundary curve
axA.fill_betweenx(y_thick, rs[0], r_boundary, facecolor="none", edgecolor="0.45",
                   hatch="////", linewidth=0.0, alpha=0.55, zorder=1)
axA.plot(r_boundary, y_thick, color="0.3", lw=1.3, ls="-", zorder=2)
axA.text(0.30, 0.955, "mixed: avalanche + logarithmic tail",
         transform=axA.transAxes, fontsize=9.5, ha="center", va="top", style="italic",
         color="0.25", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# arrow from the 'logarithmic' glyph toward the genuinely pure (unshaded) corner past the boundary
axA.annotate("", xy=(0.975, 0.685), xycoords="axes fraction",
             xytext=(0.850, 0.728), textcoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.4,
                              shrinkA=2, shrinkB=2, mutation_scale=14), zorder=6)

# arrow from the 'avalanche' glyph toward r -> 0, where the cascade is a pure avalanche
axA.annotate("", xy=(0.018, 0.8475), xycoords="axes fraction",
             xytext=(0.082, 0.8475), textcoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4,
                              shrinkA=2, shrinkB=2, mutation_scale=14), zorder=6)

axA.set_ylabel(r"normalized context capacity  $M_{\mathrm{cap}}/M=1/\Theta$")
axA.set_title("(a) effective phase diagram: capacity the society can sustain", loc="left", fontsize=13)

# ---- panel (b): observed participation ratio field ----
pc = axB.pcolormesh(rs, ys, PRn, cmap="viridis", vmin=0, vmax=1,
                    shading="auto", rasterized=True)
frame(axB)
fig.colorbar(pc, ax=axB, pad=0.02).set_label(r"fraction of contexts effectively active  $D_{\mathrm{eff}}/M$")
axB.set_title("(b) realization: contexts effectively active", loc="left", fontsize=13)

fig.tight_layout()
fig.savefig("fig_characterization_2panel.pdf"); fig.savefig("fig_characterization_2panel.png", dpi=190)
print("done")
