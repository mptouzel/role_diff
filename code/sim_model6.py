"""Model 6: the closed schema-role-resource loop, integrated directly.

Three coupled dynamics on separated timescales (gamma >> c >> kappa):

  agents     wdot_i  = -gamma w_i (1+|w_i|^2) + beta sum_m a_im g_m + sigma_dyn eta
  resources  rdot_i  = sum_m W_im - c r_i,          W = w0 (1-2Pe)^2
  schemas    gdot_m  = -kappa g_m + alpha C_r g_m,  C_r = sum_i r_i w_i w_i^T / sum_i r_i

Conventions (fixed in the paper):
  * encounter clock: one random perfect matching per step (dense-encounter limit, nu=1)
  * context frequencies f_m uniform over all M candidates, constant absorbed into beta,
    so every context is played each encounter at weight 1 and beta is the per-context
    write rate.  f_m is NOT renormalized as schemas go extinct.
  * decisions are hard signs of noisy pairwise gaps (no mean-field substitution)

Design choices made explicit (see paper):
  * the budget is not projected; the per-schema ceiling u_sat arrests condensation and
    the realized total sum_m u_m is measured, not imposed.
  * ceiling implemented as a smooth logistic factor (1 - u_m/u_sat)_+ on the growth
    term; `--hard` switches to a clip, to test robustness.
  * N_eff = (sum_i r_i)^2 / sum_i r_i^2 is logged; C_r is only trustworthy for N_eff >> d.

`--radial` restricts the schema dynamics to its strength (radial) sector,
gdot_m = g_m (-kappa + alpha ghat_m^T C_r ghat_m), freezing the directions.  The
full vector form splits exactly into this replicator plus Oja's rule on ghat_m;
the rotation drives every schema onto C_r's leading eigenvector (all pairwise
overlaps -> 1 in test runs), so under `--radial` the context directions are taken
as given by the environment and the loop selects only how strongly each schema is
institutionalized.  `--coherence chi` sets a uniform pairwise overlap
ghat_m . ghat_n = chi in the seeded portfolio (the transposability knob).
"""
import argparse
import numpy as np
from scipy.special import erf


def run(N=3000, d=12, M=24, T=40000, dt=0.01,
        gamma=1.0, c=0.10, kappa=0.010, alpha=None, beta=2.2,
        sigma_obs=0.6, sigma_dyn=0.25, w0=1.5,
        u_sat=0.55, u0=1e-3, seed_spread=0.35, S_bw=None, hard_ceiling=False,
        beta_end=None, orthogonal_seed=False, seed=0, record_every=100,
        radial=False, coherence=0.0):

    rng = np.random.default_rng(seed)
    piprime = 1.0 / (sigma_obs * np.sqrt(2 * np.pi))     # threshold policy slope

    # --- schemas: M candidate directions, seeded at u0, random in R^d ---
    if orthogonal_seed and M <= d:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        dirs = Q[:, :M].T                                 # near-orthogonal candidates
        if coherence > 0.0:
            # Tilt every candidate equally toward their own mean direction, giving a
            # symmetric portfolio with pairwise overlap ghat_m . ghat_n = coherence
            # for all m != n.  This is the transposability knob: cross-axis seeding
            # needs overlap (orthogonal contexts couple only competitively).
            s = np.sqrt(M)
            A = coherence / (1.0 - coherence)
            t = -1.0 / s + np.sqrt(1.0 / s**2 + A)
            v = dirs.sum(axis=0) / s                      # unit mean direction
            dirs = dirs + t * v[None, :]
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    else:
        dirs = rng.standard_normal((M, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    u_seed = u0 * np.exp(seed_spread * rng.standard_normal(M))   # lognormal seed spread
    G = dirs * np.sqrt(u_seed)[:, None]                   # g_m = sqrt(u_m) ghat_m

    # --- agents, resources ---
    w = 0.01 * rng.standard_normal((N, d))
    r = np.ones(N)
    f_ctx = np.full(M, 1.0 / M)                           # uniform play frequencies

    # alpha set so that a schema at the subcritical noise floor sits near the
    # invasion margin: alpha * v ~ kappa, with v = sdyn^2 / (2 gamma).
    if alpha is None:
        alpha = 2.5 * kappa / (sigma_dyn**2 / (2 * gamma))

    beta0 = beta
    idx = np.arange(N)
    rec = {k: [] for k in ["t", "u", "lam", "Neff", "Sbw", "K", "Cspec", "beta", "Nact"]}

    for step in range(T):
        if beta_end is not None:                 # linear ramp of the loop gain, in beta
            beta = beta0 + (beta_end - beta0) * step / T
        # ---- encounters: one random perfect matching (N even) ----
        perm = rng.permutation(N)
        partner = np.empty(N, dtype=int)
        partner[perm[0::2]] = perm[1::2]
        partner[perm[1::2]] = perm[0::2]

        gaps = w - w[partner]                              # (N,d)
        proj = gaps @ G.T                                  # (N,M) true gaps per context
        s = proj + sigma_obs * rng.standard_normal((N, M)) # noisy observation
        a = np.sign(s)                                     # hard decision, +-1

        # ---- agents ----
        fb = a @ G                                         # sum_m a_im g_m
        drift = -gamma * w * (1.0 + np.sum(w**2, axis=1, keepdims=True)) + beta * fb
        w += dt * drift + sigma_dyn * np.sqrt(dt) * rng.standard_normal((N, d))

        # ---- resources: realized coordination payoff per encounter ----
        # Pe from the true gap and observation noise; W = w0 (1-2Pe)^2
        # 1-2Pe = erf(|Delta| / (sigma_obs sqrt 2))  for the threshold agent
        W = w0 * erf(np.abs(proj) / (sigma_obs * np.sqrt(2.0))) ** 2   # (N,M)
        # play frequencies weight the income as they weight the gain matrix; under
        # the uniform f_m used throughout this is an overall constant, which C_r
        # divides out by normalizing, so it does not change any result here.
        r += dt * (W @ f_ctx - c * r)
        np.maximum(r, 1e-12, out=r)

        # ---- schemas (slowest) ----
        rw = r / r.sum()
        Cr = (w * rw[:, None]).T @ w                       # resource-weighted covariance
        u = np.sum(G**2, axis=1)
        growth = Cr @ G.T                                  # (d,M): C_r g_m
        if hard_ceiling:
            gate = np.ones(M)
        else:
            gate = np.clip(1.0 - u / u_sat, 0.0, 1.0)      # smooth saturation
        if radial:
            # radial-only: keep the strength replicator, drop the rotation.
            # gdot_m = g_m (-kappa + alpha lambda_m),  lambda_m = ghat_m^T C_r ghat_m.
            # Directions are set by the game contexts (environment) and never move;
            # the loop selects only how strongly each schema is institutionalized.
            lam_m = np.einsum('md,md->m', G, growth.T) / np.maximum(u, 1e-300)
            G += dt * G * (-kappa + alpha * lam_m * gate)[:, None]
        else:
            G += dt * (-kappa * G + alpha * growth.T * gate[:, None])
        if hard_ceiling:
            un = np.sum(G**2, axis=1)
            over = un > u_sat
            if over.any():
                G[over] *= np.sqrt(u_sat / un[over])[:, None]
        # budget: the replicator does not conserve sum_m u_m, so if a finite social
        # bandwidth is imposed we project onto it (Theta = M u_sat / S_bw is then a control)
        if S_bw is not None:
            tot = np.sum(G**2)
            if tot > S_bw:                 # budget is a bound, not a normalization
                G *= np.sqrt(S_bw / tot)
        # floor: dormant schemas are held at the seed scale, not driven to zero
        un = np.sum(G**2, axis=1)
        low = un < u0
        if low.any():
            G[low] *= np.sqrt(u0 / np.maximum(un[low], 1e-300))[:, None]

        if step % record_every == 0:
            u = np.sum(G**2, axis=1)
            C = (w.T @ w) / N
            Neff = r.sum() ** 2 / np.sum(r**2)
            rec["t"].append(step * dt)
            rec["u"].append(u.copy())
            rec["lam"].append(np.linalg.eigvalsh(Cr)[::-1].copy())
            rec["Cspec"].append(np.linalg.eigvalsh(C)[::-1].copy())
            rec["Neff"].append(Neff)
            rec["Sbw"].append(u.sum())
            rec["K"].append(int((u > 3 * u0).sum()))
            rec["beta"].append(beta)
            floor = sigma_dyn**2 / (2 * gamma)
            rec["Nact"].append(int((np.linalg.eigvalsh(C) > 1.6 * floor).sum()))

    for k in rec:
        rec[k] = np.array(rec[k])
    rec["G_final"] = G.copy()          # to check directions (radial: must equal dirs0)
    rec["dirs0"] = dirs.copy()
    rec["params"] = dict(S_bw=S_bw, Theta=(M*u_sat/S_bw if S_bw else None),
                         N=N, d=d, M=M, dt=dt, gamma=gamma, c=c, kappa=kappa,
                         alpha=alpha, beta=beta, sigma_obs=sigma_obs,
                         sigma_dyn=sigma_dyn, u_sat=u_sat, u0=u0,
                         hard=hard_ceiling, seed=seed, piprime=piprime,
                         radial=radial, coherence=coherence)
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=40000)
    p.add_argument("--beta", type=float, default=2.2)
    p.add_argument("--usat", type=float, default=0.55)
    p.add_argument("--c", type=float, default=0.10)
    p.add_argument("--hard", action="store_true")
    p.add_argument("--radial", action="store_true")
    p.add_argument("--coherence", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="m6.npz")
    a = p.parse_args()
    R = run(T=a.T, beta=a.beta, u_sat=a.usat, c=a.c, hard_ceiling=a.hard, seed=a.seed,
            radial=a.radial, coherence=a.coherence)
    np.savez_compressed(a.out, **{k: v for k, v in R.items() if k != "params"},
                        params=np.array([str(R["params"])], dtype=object))
    u = R["u"][-1]
    print(f"done. K={int((u>3*R['params']['u0']).sum())}  "
          f"S_bw={u.sum():.3f}  N_eff={R['Neff'][-1]:.0f}  "
          f"top u: {np.round(np.sort(u)[::-1][:6],4)}")
