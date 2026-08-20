"""Saddle phase portrait of the two-schema replicator: the degenerate (pinned)
fixed point is a saddle -- stable along the magnitude/pinning direction (u1=u2),
unstable along the budget-conserving split (condensation) toward condensates."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 12, "axes.labelsize": 14, "axes.linewidth": 0.9, "figure.dpi": 150})

# flow in (s,d)=(u1+u2, u1-u2):  s relaxes to budget S=1 (pinning);
# d grows from 0 to +-1 (condensation).  Map to (u1,u2).
S, a, b = 1.0, 1.6, 1.5
def vel(u1, u2):
    s, d = u1 + u2, u1 - u2
    sdot = -a * (s - S)
    ddot = b * d * (1 - d**2)
    return 0.5*(sdot + ddot), 0.5*(sdot - ddot)

g = np.linspace(0.02, 1.12, 32)
U1, U2 = np.meshgrid(g, g)
V1, V2 = vel(U1, U2)

fig, ax = plt.subplots(figsize=(6.6, 6.4))
ax.streamplot(g, g, V1, V2, color="0.62", linewidth=0.8, density=1.25, arrowsize=0.9)

# manifolds
ax.plot([0, 1.12], [0, 1.12], color="tab:green", lw=2.2, zorder=4)          # stable: u1=u2 (pinning)
ax.plot([0, 1.0], [1.0, 0.0], color="tab:red", lw=2.2, ls=(0,(5,2)), zorder=4)  # unstable: budget line (condensation)

# fixed points
ax.plot(0.5, 0.5, "o", mfc="white", mec="k", mew=1.8, ms=13, zorder=6)       # saddle (degenerate)
ax.plot([1.0, 0.0], [0.0, 1.0], "o", color="tab:red", ms=13, zorder=6)       # condensates

# manifold direction arrows
for (x,y,dx,dy,c) in [(0.26,0.26, 0.05,0.05,"tab:green"),(0.74,0.74,-0.05,-0.05,"tab:green"),
                      (0.5,0.5, 0.09,-0.09,"tab:red"),(0.5,0.5,-0.09,0.09,"tab:red")]:
    ax.annotate("", xy=(x+dx,y+dy), xytext=(x,y),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0), zorder=5)

# labels
ax.text(0.5, 0.545, "degenerate\n(pinned) state", ha="center", va="bottom", fontsize=10.5)
ax.text(1.0, 0.055, "condensate", ha="center", va="bottom", fontsize=10.5, color="tab:red")
ax.text(0.03, 0.97, "condensate", ha="left", va="center", fontsize=10.5, color="tab:red")
ax.text(0.90, 0.90, "stable manifold\n= pinning", color="tab:green", ha="right", va="center", fontsize=11)
ax.text(0.30, 0.82, "unstable manifold\n= condensation", color="tab:red", ha="center", va="center", fontsize=11)

ax.set_xlim(0, 1.12); ax.set_ylim(0, 1.12); ax.set_aspect("equal")
ax.set_xlabel(r"schema strength  $u_1$"); ax.set_ylabel(r"schema strength  $u_2$")
ax.set_title("Two-schema replicator: pinning vs condensation", fontsize=12.5, loc="left")
fig.tight_layout()
fig.savefig("saddle.png", dpi=190); fig.savefig("saddle.pdf")
print("done")
