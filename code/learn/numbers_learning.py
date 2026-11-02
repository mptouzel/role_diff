"""Every number quoted in Supplement S1 D, recomputed from data/learn/*.npz (no simulation).
Run from code/learn:  python numbers_learning.py"""
import os, numpy as np
import core as c

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "learn")
L = lambda n: np.load(os.path.join(D, n + ".npz"))
f, sw, sg, es, tb, tl = (L(n) for n in ["fixed_k", "learn_sweep", "learn_sign", "learn_es", "tilt_basin",
                                         "tilt_learn_seed300"])
ks, betas = f["ks"], list(f["betas"]); bi = betas.index(6.0)
fresh = L("tilt_fresh") if os.path.exists(os.path.join(D, "tilt_fresh.npz")) else None


def final_l(X):                      # mean slope over the last 10 records, per context
    return X[-10:, 0, :].mean(0)


def tcross(X, rec, dt, lc):          # first time each context's slope exceeds its threshold
    t = np.arange(1, X.shape[0] + 1) * rec * dt
    return [t[np.argmax(X[:, 0, m] > lc[m])] if np.isfinite(lc[m]) and (X[:, 0, m] > lc[m]).any() else np.nan
            for m in range(c.M)]


print("== policy family and thresholds")
print(f"g(inf) = {c.G_SIGN:.4f}  (sqrt(2/pi)/sigma_read)")
for b in [2.0, 6.0, 12.0]:
    print(f"beta={b:g}: sign-rule Lambda_m = {np.round(b * c.U * c.G_SIGN / c.GAMMA, 3)}, ell_c = {np.round(c.k_c(b), 3)}")

print("\n== mixed Nash and small-ell growth rate (fixed_k)")
i0 = np.argmin(abs(ks))
print(f"payoff at ell=0, beta=2/6/12: {np.round(f['r'][:, i0], 4)}")
sel = (np.abs(ks) > 0) & (np.abs(ks) <= 0.12)
ratio = f["gr"][:, sel, :] / ks[None, sel, None] / (1.5 * f["d2"][:, sel, :])
print(f"|ell|<=0.12: gradient/ell over (3/2) E[Delta^2] in [{ratio.min():.3f}, {ratio.max():.3f}], "
      f"max |dev| {np.abs(ratio - 1).max() * 100:.1f}%")
lin = f["gr"][:, sel, :] / ks[None, sel, None] / (3 * c.U[None, None, :] * f["v"][:, sel, :])
print(f"linear-score value 3 u_m v_m: gradient/ell over it in [{lin.min():.3f}, {lin.max():.3f}], "
      f"overestimate up to {(1 / lin.min() - 1) * 100:.0f}% (beta=12 rows: min {lin[-1].min():.3f})")
print(f"floor sigma_w^2/2gamma = {c.FLOOR:.4f}; v_1 at ell=0, beta=2/6/12: {np.round(f['v'][:, i0, 0], 3)}")

print("\n== learning through the cascade (learn_sweep, eps=1, ell0=0.05)")
rec, dt = int(sw["rec"]), float(sw["dt"])
for b in [6, 12]:
    for s in range(2):
        X = sw[f"b{b}_e1_s{s}"]
        print(f"beta={b} seed {s}: crossing t = {np.round(tcross(X, rec, dt, c.k_c(float(b))), 1)}, "
              f"final ell = {np.round(final_l(X), 2)}, final payoff {sw[f'R_b{b}_e1_s{s}'][-200:].mean():.3f}")
X = sw["b6_e1_s0"]; logk = np.log(ks[ks > 0])
dev = []
for m in range(2):
    ok = X[:, 0, m] > ks[ks > 0].min()
    vf = np.exp(np.interp(np.log(X[ok, 0, m]), logk, np.log(f["v"][bi, ks > 0, m])))
    dev.append(np.abs(X[ok, 2, m] - vf) / X[ok, 2, m])
print(f"tracking, beta=6 contexts 1-2: median |v - v_fixed(ell)|/v = {np.median(np.concatenate(dev)) * 100:.1f}%")
th = max(abs(sw[k][:, 1, :]).max() for k in sw.files if not k.startswith("R_") and sw[k].ndim == 3)
print(f"learned offset theta: max |theta| over all sweep runs = {th:.3f}")
print(f"sign-rule payoff beta=6 = {f['sign_payoff'][1]:.3f}")
X = sw["b2_e1_s0"]
print(f"beta=2: final ell = {np.round(final_l(X), 2)}, v_1 final {X[-10:, 2, 0].mean():.3f}, "
      f"payoff {sw['R_b2_e1_s0'][-200:].mean():.3f}, ell=+20 payoff {f['r'][0, -1]:.3f}")

print("\n== orientation without asymmetry (fixed_k, learn_sign, learn_es; beta=6)")
print(f"payoff ell=+20 / ell=-20: {f['r'][bi, -1]:.3f} / {f['r'][bi, 0]:.3f}; "
      f"v_1 at ell=-20 / ell=0: {f['v'][bi, 0, 0]:.3f} / {f['v'][bi, i0, 0]:.3f}")
r6 = f["r"][bi]
drop = max(-np.diff(r6[ks >= 0]).min(), -np.diff(r6[ks <= 0][::-1]).min())
print(f"fixed-ell payoff vs |ell|: largest decrease on either branch {drop:.4f} (Monte Carlo scale)")
fol = np.array([final_l(sg[f"k00_s{s}"]) > 0 for s in range(10)])
print(f"ell0=0: follow in {fol.sum()}/{fol.size} context-runs")
print(f"ell0=-0.5: final ell_1 = {[round(final_l(sg[f'k0-0.5_s{s}'])[0], 2) for s in range(3)]}")
print(f"ES (scalar ell, 20 iterations): from -0.5 -> {es['k0-0.5'][-1]:.2f}, from +0.5 -> {es['k00.5'][-1]:.2f}")
if fresh is not None:
    fol = np.array([final_l(fresh[f"k00_c0_s{s}"]) > 0 for s in range(12)])
    print(f"fresh seeds, ell0=0, c=0: follow in {fol.sum()}/{fol.size} context-runs")

print("\n== correlated asymmetry (tilt_basin, tilt_learn_seed300; beta=6, Delta_0 = 0.5)")
CL, G, kb = list(tb["c_list"]), tb["grad"], tb["ks"]
z = np.argmin(abs(kb))
print(f"gradient at ell=0 by c (context 1): {dict(zip(CL, np.round(G[:, z, 0], 4)))}")
neg = (kb <= 0) & (kb >= -4)
for m in range(c.M):
    e = []
    for i in range(1, len(CL)):
        g, kk = G[i, neg, m], kb[neg]
        j = [q for q in range(len(kk) - 1) if g[q] < 0 <= g[q + 1]][-1]
        e.append(-(kk[j] - g[j] * (kk[j + 1] - kk[j]) / (g[j + 1] - g[j])))
    print(f"context {m + 1}: |ell*| = {np.round(e, 3)}  vs c/6Delta_0 = {np.round(np.array(CL[1:]) / (6 * c.D0), 3)}")
rec, dt = int(tl["rec"]), float(tl["dt"])
lc = c.k_c(6.0)
for ct in [0.1, 0.25, 0.5, 1.0]:
    Xs = [tl[f"k00_c{ct:g}_s{s}"] for s in range(10)]
    fol = np.array([final_l(X) > 0 for X in Xs])
    tc = np.array([tcross(X, rec, dt, lc)[:2] for X in Xs])
    order = np.sum(tc[:, 0] < tc[:, 1])
    print(f"c={ct:g}: follow {fol.sum()}/30 (per context {fol.sum(0)}), ctx1 before ctx2 {order}/10, "
          f"median crossing t {np.round(np.nanmedian(tc, 0), 0)}")
for k0 in [-0.5, -3.0]:
    for ct in [0.5, 1.0, 2.0]:
        fin = [round(final_l(tl[f'k0{k0:g}_c{ct:g}_s{s}'])[0], 2) for s in range(3)]
        print(f"ell0={k0:g}, c={ct:g}: final ell_1 = {fin}")
if fresh is not None:
    Xs = [fresh[f"k00_c0.1_s{s}"] for s in range(10)]
    fol = np.array([final_l(X) > 0 for X in Xs])
    print(f"fresh seeds, ell0=0, c=0.1: follow per context {fol.sum(0)}/10, "
          f"final |ell_3| {np.round(sorted(abs(np.array([final_l(X)[2] for X in Xs]))), 2)}")
    for ct in [1.0, 2.0]:
        print(f"fresh seed, ell0=-0.5, c={ct:g}: final ell_1 = {final_l(fresh[f'k0-0.5_c{ct:g}_s0'])[0]:.2f}")
