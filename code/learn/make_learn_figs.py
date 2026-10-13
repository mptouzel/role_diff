"""Summary figure for the learned-policy runs. Reads data/learn/*.npz only."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core as c

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "learn"); F = os.path.join(ROOT, "figures", "learn"); os.makedirs(F, exist_ok=True)
COL = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#1f1f1e", "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "lines.linewidth": 2, "legend.frameon": False})

f = np.load(os.path.join(D, "fixed_k.npz")); ks = f["ks"]; betas = list(f["betas"])
sw = np.load(os.path.join(D, "learn_sweep.npz")); sg = np.load(os.path.join(D, "learn_sign.npz"))
es = np.load(os.path.join(D, "learn_es.npz"))
bi = betas.index(6.0)

fig, ax = plt.subplots(2, 3, figsize=(11, 6.4))

# (a) fixed k: status variance along each context vs k, beta=6
a = ax[0, 0]
kk = np.r_[-np.geomspace(0.05, 20, 200)[::-1], np.geomspace(0.05, 20, 200)]
for m in range(3):
    a.plot(ks, f["v"][bi, :, m], "o-", ms=3, lw=1.5, color=COL[m], label=f"context {m+1}")
    a.plot(kk, c.v_lin(kk, 6.0)[:, m], ":", lw=1.2, color=COL[m])
for m, kc in enumerate(c.k_c(6.0)):
    if np.isfinite(kc):
        a.axvline(kc, color=COL[m], lw=0.8, alpha=0.5)
a.set_xscale("symlog", linthresh=0.05); a.set_yscale("log")
a.set_xlabel("policy slope k (fixed)"); a.set_ylabel("status variance $v_m$")
a.set_title("(a) fixed k, β=6: dotted = flat-floor linear theory\nthin lines: $k_c$", fontsize=9, loc="left")
a.legend(fontsize=8)

# (b) H1: measured gradient / k vs 3 u v
a = ax[0, 1]
for j, b in enumerate(betas):
    sel = (np.abs(ks) > 0) & (np.abs(ks) <= 0.12)
    for m in range(3):
        x = 3 * c.U[m] * f["v"][j, sel, m]; y = f["gr"][j, sel, m] / ks[sel]
        a.plot(x, y, "o", ms=4, color=COL[m], alpha=0.8, label=f"context {m+1}" if j == 0 else None)
lim = [0.005, 0.3]; a.plot(lim, lim, "-", lw=1, color=MUTED)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("prediction  3·u_m·v_m"); a.set_ylabel("exact mean gradient / k")
a.set_title("(b) H1: small-k growth rate, |k|≤0.12, β∈{2,6,12}", fontsize=9, loc="left")
a.legend(fontsize=8)

# (c) H2: learned k(t), beta=6 and 12, eps=1
a = ax[0, 2]
t = np.arange(1, sw["b6_e1_s0"].shape[0] + 1) * int(sw["rec"]) * float(sw["dt"])
for m in range(3):
    a.plot(t, sw["b6_e1_s0"][:, 0, m], color=COL[m], label=f"context {m+1}, β=6")
    a.plot(t, sw["b12_e1_s0"][:, 0, m], "--", lw=1.3, color=COL[m])
for m, kc in enumerate(c.k_c(6.0)):
    if np.isfinite(kc):
        a.axhline(kc, color=COL[m], lw=0.8, alpha=0.5)
a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(0.03, 12)
a.set_xlabel("time (1/γ)"); a.set_ylabel("learned k")
a.set_title("(c) H2: myopic PG from k₀=0.05 (dashed β=12)\nthin lines: $k_c$ at β=6", fontsize=9, loc="left")
a.legend(fontsize=8)

# (d) H2 adiabatic: learned (k, v) on top of fixed-k curve
a = ax[1, 0]
X = sw["b6_e1_s0"]
for m in range(3):
    a.plot(ks[ks > 0], f["v"][bi, ks > 0, m], "-", lw=4, alpha=0.25, color=COL[m])
    a.plot(X[:, 0, m], X[:, 2, m], ".", ms=2, color=COL[m])
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.04, 12)
a.set_xlabel("k"); a.set_ylabel("$v_m$")
a.set_title("(d) learning tracks the fixed-k curve (β=6)\nwide = fixed k, dots = during learning", fontsize=9, loc="left")

# (e) H4: sign selection
a = ax[1, 1]
t = np.arange(1, sg["k00_s0"].shape[0] + 1) * int(sg["rec"]) * float(sg["dt"])
for s in range(10):
    a.plot(t, sg[f"k00_s{s}"][:, 0, 0], color=COL[0], lw=1, alpha=0.7, label="PG, k₀=0 (10 seeds)" if s == 0 else None)
for s in range(3):
    a.plot(t, sg[f"k0-0.5_s{s}"][:, 0, 0], color=COL[1], lw=1.2, label="PG, k₀=−0.5" if s == 0 else None)
it = np.arange(len(es["k0-0.5"])) * 30
a.plot(it, es["k0-0.5"], "s-", ms=3, lw=1.2, color=INK, label="ES across populations, k₀=−0.5")
a.plot(it, es["k00.5"], "s--", ms=3, lw=1.2, color=MUTED, label="ES, k₀=+0.5")
a.axhline(0, color=MUTED, lw=0.8)
a.set_yscale("symlog", linthresh=0.1); a.set_xscale("symlog", linthresh=1)
a.set_xlabel("time (1/γ); ES: iteration × 30"); a.set_ylabel("k (context 1)")
a.set_title("(e) H4: sign is a coin flip; nothing flips it back", fontsize=9, loc="left")
a.legend(fontsize=7)

# (f) payoff vs k, both signs
a = ax[1, 2]
for j, b in enumerate(betas):
    a.plot(ks, f["r"][j], "o-", ms=3, lw=1.5, color=["#8a8984", COL[0], COL[1]][j], label=f"β={b:g}")
a.axhline(c.NASH, color=MUTED, lw=0.8, ls=":")
a.set_xscale("symlog", linthresh=0.05)
a.set_xlabel("policy slope k (fixed)"); a.set_ylabel("mean payoff per encounter")
a.set_title("(f) payoff rises with |k| on both sides;\nfollow beats anti-follow, gap opens at ignition", fontsize=9, loc="left")
a.legend(fontsize=8)

TK = [-10, -1, -0.1, 0, 0.1, 1, 10]
for a in [ax[0, 0], ax[1, 2]]:
    a.set_xticks(TK); a.set_xticklabels(["−10", "−1", "−0.1", "0", "0.1", "1", "10"])
ax[1, 1].set_xlim(0, None)
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(F, f"learn_summary.{ext}"), dpi=160)
print("wrote figures/learn/learn_summary.{pdf,png}")
