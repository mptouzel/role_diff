"""Figure for the correlated-asymmetry (tilt) experiments. Reads data/learn/tilt_*.npz and learn_sign.npz."""
import os, sys, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core as c

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "learn"); F = os.path.join(ROOT, "figures", "learn")
SEQ = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # one hue, light -> dark = small -> large c
INK, MUTED, ANTI = "#1f1f1e", "#8a8984", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "lines.linewidth": 2, "legend.frameon": False})

b = np.load(os.path.join(D, "tilt_basin.npz")); ks, G, CL = b["ks"], b["grad"], list(b["c_list"])
L = np.load(os.path.join(D, sys.argv[1] if len(sys.argv) > 1 else "tilt_learn_seed300.npz"))
S = np.load(os.path.join(D, "learn_sign.npz"))
t = np.arange(1, L["k00_c0.1_s0"].shape[0] + 1) * int(L["rec"]) * float(L["dt"])

fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))

a = ax[0]
sel = ks >= -4.5
for i, ct in enumerate(CL):
    a.plot(ks[sel], G[i, sel, 0], "-", color=SEQ[i], label=f"c={ct:g}")
a.axhline(0, color=MUTED, lw=0.8); a.axvline(0, color=MUTED, lw=0.8)
a.set_xscale("symlog", linthresh=0.05); a.set_xticks([-4, -1, -0.1, 0, 0.1, 0.5]); a.set_xticklabels(["−4", "−1", "−0.1", "0", "0.1", "0.5"])
a.set_xlabel("k (fixed, all agents)"); a.set_ylabel("exact myopic dE[r]/dk, context 1")
a.set_title("(a) tilt makes k=0 flow to follow;\ndeep anti-follow still flows away", fontsize=9, loc="left")
a.legend(fontsize=8)

a = ax[1]
neg = ks <= 0
for m, mk in zip(range(3), ["o", "s", "^"]):
    kst = []
    for i, ct in enumerate(CL[1:], 1):
        g, kk = G[i, neg, m], ks[neg]
        j = [q for q in range(len(kk) - 1) if g[q] < 0 <= g[q + 1]][-1]
        kst.append(kk[j] - g[j] * (kk[j + 1] - kk[j]) / (g[j + 1] - g[j]))
    a.plot(CL[1:], -np.array(kst), mk, ms=6, color=["#2a78d6", "#eb6834", "#1baf7a"][m], label=f"context {m+1}")
cc = np.linspace(0, 2.1, 50); a.plot(cc, cc / (6 * c.D0), "-", lw=1.2, color=MUTED, label="c / (6 D₀)")
a.set_xlabel("tilt c"); a.set_ylabel("anti-follow basin edge |k*|")
a.set_title("(b) anti-follow is stable only for k < k*", fontsize=9, loc="left"); a.legend(fontsize=8)

a = ax[2]
for s in range(10):
    a.plot(t, S[f"k00_s{s}"][:t.size, 0, 0], color=ANTI, lw=1, alpha=0.6, label="c=0" if s == 0 else None)
    a.plot(t, L[f"k00_c0.1_s{s}"][:, 0, 0], color=SEQ[2], lw=1, alpha=0.8, label="c=0.1" if s == 0 else None)
a.axhline(0, color=MUTED, lw=0.8)
a.set_yscale("symlog", linthresh=0.1); a.set_xscale("log")
a.set_xlabel("time (1/γ)"); a.set_ylabel("learned k, context 1")
a.set_title("(c) from k₀=0, 10 seeds each", fontsize=9, loc="left"); a.legend(fontsize=8)

a = ax[3]
for k0, ls in [(-0.5, "-"), (-3.0, "--")]:
    for i, ct in enumerate([0.5, 1.0, 2.0]):
        for s in range(3):
            key = f"k0{k0:g}_c{ct:g}_s{s}"
            if key in L:
                a.plot(t, L[key][:, 0, 0], ls, color=SEQ[i + 2], lw=1.2,
                       label=f"c={ct:g}, k₀={k0:g}" if s == 0 else None)
a.axhline(0, color=MUTED, lw=0.8)
a.set_yscale("symlog", linthresh=0.1); a.set_xscale("log")
a.set_xlabel("time (1/γ)"); a.set_ylabel("learned k, context 1")
a.set_title("(d) starting inside anti-follow", fontsize=9, loc="left"); a.legend(fontsize=7)

fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(F, f"tilt_summary.{ext}"), dpi=160)
print("wrote figures/learn/tilt_summary.{pdf,png}")
