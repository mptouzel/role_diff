"""Figure 4 (monitoring), two panels:
   (a) monitorability over the replicator-control plane (bifurcation clustering);
   (b) the monitor's view under linear ramping of the loop gain (why early windows
       are non-identifying, and the covariance early-warning remedy)."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 0.9, "figure.dpi": 150})

rng = np.random.default_rng(3)
M, S = 48, 1.0
u0 = 1e-4 * S / M
WIN = 0.15
jit_sat = 1 + 0.01 * rng.normal(size=M)
jit_fr = 1 + 0.02 * rng.normal(size=M)

def spectrum(r, Theta):
    u_sat = Theta * S / M
    if Theta <= 1.0:
        L = max(np.log(u_sat / u0), 1e-6); n_sat = max(0.0, M - r * L)
        k = np.arange(1, M + 1, dtype=float)
        u = np.where(k <= n_sat, u_sat * jit_sat[:M], u_sat * np.exp(-(k - n_sat) / r))
        return np.maximum(u, u0)
    elif Theta <= M:
        a = 1.0 / r; k = np.arange(1, M + 1, dtype=float); lo, hi = 1e-12 * S, 1e4 * S
        def total(C): return np.sum(np.minimum(u_sat, C * k ** (-a)))
        for _ in range(80):
            mid = np.sqrt(lo * hi); hi, lo = (mid, lo) if total(mid) > S else (hi, mid)
        return np.maximum(np.minimum(u_sat * jit_sat, np.sqrt(lo * hi) * k ** (-a)), 1e-30)
    else:
        u = np.empty(M); u[0] = S - (M - 1) * u0; u[1:] = u0 * jit_fr[1:M]; return u

def avalanche_fraction(u):
    mu = u / u.max(); lb = np.sort(np.log(1.0 / mu)); best = j = 0
    for i in range(M):
        while j < M and lb[j] <= lb[i] + WIN: j += 1
        best = max(best, j - i)
    return best / M

# ---------- panel (b): the two repertoires compared ----------
R = 48; kk = np.arange(1, R + 1)

def counting_curve(mu):
    b = np.sort(1.0 / mu); bg, Ng = [0.5], [0.0]
    for i, bb in enumerate(b):
        bg += [bb, bb]; Ng += [i / len(b), (i + 1) / len(b)]
    bg.append(1e4); Ng.append(1.0); return np.array(bg), np.array(Ng)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.3))

# ================= panel (a): monitorability field (M_cap/M axis, matches Fig 3a) =================
b_hi, b_lo = 1.0, 1.0 / M                       # M_cap/M = 1 (all contexts), 1/M (single)
loA, hiA = np.log10(b_lo), np.log10(b_hi); gapA = hiA - loA
ymin, ymax = 10 ** (loA - gapA), 10 ** (hiA + gapA)   # three equal bands (boundaries at 1/3, 2/3)
rs = np.logspace(np.log10(0.1), np.log10(10), 240)
ys = np.logspace(np.log10(ymin), np.log10(ymax), 240)  # y = M_cap/M
Z = np.empty((len(ys), len(rs)))
for iy, y in enumerate(ys):
    Th = 1.0 / y
    for ix, r in enumerate(rs):
        Z[iy, ix] = avalanche_fraction(spectrum(r, Th))
pc = axA.pcolormesh(rs, ys, Z, cmap="RdYlGn_r", vmin=0, vmax=1, shading="auto", rasterized=True)
axA.set_xscale("log"); axA.set_yscale("log")
cb = fig.colorbar(pc, ax=axA, pad=0.015); cb.set_label("bifurcation clustering (unmonitorability)")
axA.axhline(b_hi, color="k", lw=1.1, ls="--"); axA.axhline(b_lo, color="k", lw=1.1, ls="--")
axA.plot([1, 1], [b_lo, b_hi], color="k", lw=1.0, ls=":")   # a=1, middle band
tg = dict(fontsize=10, ha="left", va="center", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
axA.text(0.115, b_hi, r"$M_{\mathrm{cap}}=M$", **tg)
axA.text(0.115, b_lo, r"$M_{\mathrm{cap}}=1$", **tg)
axA.text(1.12, b_lo * 2.4, r"$a=1$", fontsize=10, ha="left",
         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

axA.set_xlim(rs[0], rs[-1]); axA.set_ylim(ymin, ymax)
axA.set_xlabel("speed\n" r"activations per compounding time  $r=r_{\mathrm{act}}/g$"); axA.set_ylabel("capacity\n" r"saturated slots per context  $M_{\mathrm{cap}}/M=1/\Theta$")
axA.set_title("(a) monitorability over the regime plane", loc="left")

# ================= panel (b): the measurement term =================
# Two repertoires whose counting functions coincide until one avalanches: both
# activate one axis at beta_c^(1), then go quiescent. One has its remaining R-1 modes
# just below threshold, the other the same modes 8x further down. Rate data cannot
# separate them, because the two spectra differ only below gamma/beta. On a log axis
# the gap between the two rises equals the gap between the two recovered bulks: the
# same factor of 8, read off the subcritical covariance before either bifurcates.
GAMMA, S2 = 1.0, 1.0                    # gamma, sigma_dyn^2/2 (the scale drops out)
BOBS = 0.95                             # observation coupling, below beta_c^(1)=1

chi_s = 0.20
mu_near = np.concatenate([[1.0], (1-chi_s)/(1+(R-1)*chi_s)*(1+0.03*rng.normal(size=R-1))])
mu_far  = np.concatenate([[1.0], 0.0096*(1+0.03*rng.normal(size=R-1))])
m_near, m_far = mu_near[1:].mean(), mu_far[1:].mean()
b_near, b_far = 1/m_near, 1/m_far
FAC = m_near/m_far                      # the one number the panel is about

CN, CF = "tab:purple", "0.45"
hand = {}
for key, mu, col, lw in [("B", mu_far, CF, 3.0), ("A", mu_near, CN, 1.8)]:
    bg, Ng = counting_curve(mu)
    hand[key], = axB.plot(bg, Ng, lw=lw, color=col, solid_joinstyle="miter")

axB.set_xscale("log"); axB.set_xlim(0.62, 300); axB.set_ylim(-0.02, 1.06)
axB.set_xlabel("ramp\n" r"$\beta/\beta_c^{(1)} \;\propto\; \Lambda(t)$")
axB.set_ylabel(r"$N(t)/R$")
axB.axvspan(0.62, BOBS, color="0.90", zorder=0)
axB.text(0.78, 0.53, "observed here", rotation=90, ha="center", va="center",
         fontsize=8.5, color="0.35")

# the separation between the two rises, in beta
YA = 0.60
axB.annotate("", xy=(b_near, YA), xytext=(b_far, YA),
             arrowprops=dict(arrowstyle="<->", color="k", lw=1.3, shrinkA=0, shrinkB=0))
axB.text(np.sqrt(b_near*b_far), YA + 0.035, r"$\times%.0f$" % FAC, ha="center",
         va="bottom", fontsize=12)
axB.legend([hand["A"], hand["B"]], ["repertoire A", "repertoire B"], loc="upper left",
           bbox_to_anchor=(0.015, 0.99), frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=10)
axB.set_title("(b) identical rate data, different spectra", loc="left")

# inset: the gain spectrum recovered from the subcritical covariance at BOBS, before
# either repertoire has bifurcated. The same factor separates the two bulks.
axI = axB.inset_axes([0.155, 0.205, 0.29, 0.265])   # inside the whitespace between the two rises
bins = np.logspace(np.log10(4e-3), np.log10(2.6), 46)
for mu, col in [(mu_far, CF), (mu_near, CN)]:
    lam = S2/(GAMMA - BOBS*np.sort(mu)[::-1])      # C = (sdyn^2/2)(gamma I - beta G)^{-1}
    mu_rec = (GAMMA - S2/lam)/BOBS                 # the inversion returns the gain spectrum
    axI.hist(mu_rec, bins=bins, color=col, alpha=0.85, edgecolor=col, lw=0.6)
axI.set_xscale("log"); axI.set_yscale("log")
axI.set_xlim(4e-3, 2.6); axI.set_ylim(0.6, 320)
axI.set_xlabel(r"recovered $\mu_k$", fontsize=8.5, labelpad=1.5)
axI.set_ylabel("count", fontsize=8.5, labelpad=1.5)
axI.tick_params(labelsize=8.5, length=2.4, pad=1.4)
axI.set_title(r"from $\mathbf{C}$ at $\beta=0.95\,\beta_c^{(1)}$", fontsize=8.5, pad=3)
YI = 90
axI.annotate("", xy=(m_near, YI), xytext=(m_far, YI),
             arrowprops=dict(arrowstyle="<->", color="k", lw=1.1, shrinkA=0, shrinkB=0))
axI.text(np.sqrt(m_near*m_far), YI*1.35, r"$\times%.0f$" % FAC, ha="center", va="bottom", fontsize=10)
for sp in ("top", "right"): axI.spines[sp].set_visible(False)
axI.patch.set_alpha(0.94)

fig.tight_layout()
fig.savefig("fig4_monitoring.pdf", dpi=600)   # the rasterized colour field needs print dpi
fig.savefig("fig4_monitoring.png", dpi=300)
print("done")
