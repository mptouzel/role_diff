"""Figure 4 (monitoring), two panels:
   (a) monitorability over the replicator-control plane (bifurcation clustering);
   (b) the monitor's view under a linear deployment sweep (why early windows
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

# ---------- panel (b) spectra + counting curves ----------
R = 48; kk = np.arange(1, R + 1)
spec = {"geometric": np.exp(-(kk - 1) / 8.0), "zipf_shallow": kk ** (-0.5)}
Pmp = rng.normal(size=(800, 400)) / np.sqrt(800)
mp = np.sort(np.linalg.eigvalsh(Pmp.T @ Pmp))[::-1]; mp /= mp[0]
spec["mp"] = np.quantile(mp, 1 - (np.arange(R) + 0.5) / R)
chi = 0.20; mu_bulk = (1 - chi) / (1 + (R - 1) * chi)
spec["spiked"] = np.sort(np.concatenate([[1.0], mu_bulk * (1 + 0.05 * rng.normal(size=R - 1))]))[::-1]

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
axA.set_xlabel(r"resource turnover  $c/\gamma$"); axA.set_ylabel(r"normalized context capacity  $M_{\mathrm{cap}}/M$")
axA.set_title("(a) monitorability over the regime plane", loc="left")

# ================= panel (b): monitor's view (sweep) =================
BLIN = 15
for name, col, lab in [("geometric", "tab:blue", "geometric (extrapolable)"),
                       ("zipf_shallow", "tab:red", r"Zipf $a{=}\frac{1}{2}$ (accelerating)"),
                       ("mp", "tab:orange", "random MP (quiet start)"),
                       ("spiked", "tab:purple", "spiked (gap, then avalanche)")]:
    bg, Ng = counting_curve(spec[name]); axB.plot(bg, Ng, lw=1.9, color=col, label=lab)
axB.set_xlim(0, BLIN); axB.set_ylim(-0.02, 1.05)
axB.set_xlabel(r"$\beta/\beta_c^{(1)} \;\propto\; \Lambda(t)$   (linear sweep, $\dot\Lambda$ const.)")
axB.set_ylabel(r"$N(t)/R$")
axB.axvspan(0, 3, color="0.92", zorder=0)
axB.text(1.5, 0.995, "early window:\nclasses nearly\nindistinguishable", ha="center", va="top", fontsize=10, color="0.35")
axB.annotate("avalanche of pinned fringe", xy=(12.6, 0.50), xytext=(7.9, 0.30), fontsize=10, color="tab:purple",
             arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=0.9))
axB.annotate(r"quiescent gap", xy=(6.5, 0.045), xytext=(5.3, 0.20), fontsize=10, color="tab:purple",
             arrowprops=dict(arrowstyle="-", color="tab:purple", lw=0.8))
axB.annotate(r"$N \propto (\beta-\beta_c)^{3/2}$", xy=(2.1, 0.075), xytext=(2.9, 0.33), fontsize=10, color="tab:orange",
             arrowprops=dict(arrowstyle="-", color="tab:orange", lw=0.8))
axB.legend(loc="upper left", bbox_to_anchor=(0.40, 0.99), frameon=True, framealpha=0.92, edgecolor="0.85")
axB.set_title("(b) the monitor's view under a deployment sweep", loc="left")

fig.tight_layout()
fig.savefig("fig4_monitoring.pdf"); fig.savefig("fig4_monitoring.png", dpi=190)
print("done")
