# Plan: learning the role-following policy (instead of fixing it)

Drop this file at the root of a local clone of `mptouzel/role_diff`, start Claude Code there, and say:
"Read PLAN_learned_policy.md and execute it phase by phase. Stop after each phase and show me the check results before continuing."

---

## 0. Question and hypotheses

The paper fixes the policy `a = sign(s)` (Bayes-optimal role following). Here the policy is parametrized and learned from payoff experience, so the read gain becomes a dynamical variable.

Policy (per context m, single sample, T = 1):

    P(Go | s) = ½ (1 + tanh(k (s − θ)))

- `k = 0`: ignore the signal, 50/50 play = the game's mixed Nash equilibrium (payoff 2.5).
- `k → ∞, θ = 0`: the paper's sign rule.
- `k < 0`: anti-follow.

Effective read gain (replaces `g = 2π̄'(0)` in Λ):

    g_eff(k) = k · E_ξ[ sech²(k σ_read ξ) ],   g_eff(∞) = sqrt(2/π)/σ_read

so the per-mode loop gain is `Λ_m(k) = β (1 − ρ_pair) g_eff(k) u_m / γ`, and mode m ignites when learning carries k past `k_c^(m)` defined by `g_eff(k_c^(m)) = γ / (β (1−ρ_pair) u_m)`.

Hypotheses (H-numbers are referenced below):

- **H1 (mixed Nash is an unstable stationary point).** With all agents at k = 0 the expected policy gradient is zero. For small k > 0 under random pairing, the predicted mean-field growth is
  `dk/dt ≈ ε · 3 · u_m · v_m · k` (game payoffs (G,G)=−1, (G,D)=6, (D,G)=2, (D,D)=3; v_m the covariance eigenvalue along context m, at the noise floor σ_ω²/2γ before ignition). This is a back-of-envelope derivation; Phase 2 must re-derive and check it, including the factor 3.
- **H2 (learning is the sweep).** At fixed β where the sign rule is supercritical, learning drives k upward through the cascade thresholds `k_c^(m)`; modes ignite in strength order, and k accelerates after ignition because the gradient scales with v.
- **H3 (cognitive floor).** At β where even the sign rule has Λ_1 < 1, k still grows (the gradient is positive whenever v > 0) but no mode ignites; payoff stays near Nash.
- **H4 (sign selection needs long horizons or population selection).** Myopic per-encounter policy gradient is symmetric under k → −k: started at k0 < 0, it goes to anti-follow and stays there, with Λ < 0, v pushed below the floor, lower payoff. Population-level selection on long-run average payoff (evolution strategies across replica populations) does see the write-back and flips k to positive.
  *Note:* this corrects the earlier conversational claim that slow shared learning alone would see the write-back. A per-step score-function gradient does not, however slow; it requires returns over horizons ≫ 1/γ or between-population comparison.
- **H5 (pooling vs independent learners).** Per-agent (k_i, θ_i) learned from own experience: extreme-status agents rarely see one sign of s, so their θ_i is poorly constrained. Shared (k, θ) learned from pooled experience does not have this problem. Measure whether it matters for payoff.
- **Stretch H6 (status climbing).** With per-agent long-horizon returns, low-status agents may learn θ_i < 0 (Go-biased) to write themselves up. Only after H1–H5.

---

## 1. Ground rules for Claude Code

- Do not modify `main.tex`, `supplement.tex`, existing scripts in `code/`, `data/fig1_data.npz`, or anything in `figures/`. New work goes in:
  - `code/learn/` (new package), `data/learn/`, `figures/learn/`, `notes/learn_results.md`.
- Environment as in the README: build `.venv` with an interpreter that has `ensurepip`, install `requirements.txt`, always call `.venv/bin/python`. If `.venv` exists and is broken, delete it before rebuilding.
- Every script: fixed seed, saves an `.npz` with all parameters alongside results, prints a one-line summary. Plots read from `.npz` only (same split as `sim_fig1.py` / `make_fig1.py`).
- Keep runs short by default (minutes on a laptop); expose step counts as CLI args.
- Add a `learn` target to the `Makefile` only in a separate block; do not touch the `figures` target.

---

## Phase 0 — Orient (read, don't write)

1. Read `README.md`, `code/sim_fig1.py`, `Makefile`, and grep `code/` and `code/scratch/` for any resource / dynamic-schema simulation ("Model 6", `resource`, `wealth`, `u_m`). Report what exists.
2. Read the correlated-equilibrium section of `supplement.tex` (S1, S1 C) and the Λ factorization (S2). Report the paper's exact payoff convention and how `W = w0 (1 − 2 Pe)²` is defined.
3. Confirm the variable mapping in `sim_fig1.py`:
   `sigma_obs = σ_read = 0.6`, `sigma_dyn = σ_ω = 0.25`, `gamma = 1`, `dt = 0.02`, `N = 2500`, `d = M = 3`,
   strengths `|g_m| = [0.533, 0.434, 0.302]`, all M contexts played every step against one random partner, feedback `β Σ_m a_m g_m` (no 1/M), `mu = 2 π̄'(0) |g_m|²`, sign-rule thresholds `β_c ≈ [2.65, 4.0, 8.24]`.

**Check:** short written report; no code changes.

---

## Phase 1 — Core simulator with a learnable policy

Create `code/learn/sim.py`, refactoring the step of `sim_fig1.py` (copy, don't import-modify) into:

```python
def step(omega, G, policy, beta, rng, rho_pair=0.0) -> (omega_new, info)
# info: per-agent, per-context s, a, partner index, realized reward, Go prob
```

- Policy object with `prob_go(s, agent_idx, m)` and a `score(s, a, ...)` returning ∂ log π / ∂(k, θ).
  For the tanh policy, with t = tanh(k(s−θ)): `∂_k log π = (s−θ)(a − t)`, `∂_θ log π = −k (a − t)` (a ∈ {−1, +1}). Unit-test these against finite differences.
- Policy modes: `"sign"` (paper baseline), `"shared"` (one k_m, θ_m per context), `"independent"` (k_{i,m}, θ_{i,m}).
- Actions are **sampled** (stochastic), rewards realized from the payoff matrix per encounter.
- Clip |k| at a large `k_max` (e.g. 50) for numerical safety; record if hit.

`code/learn/theory.py`:
- `g_eff(k, sigma_read)` via Gauss–Hermite quadrature.
- `Lambda_m(k, beta, u, gamma, rho_pair)`, `k_c(m, ...)`, subcritical `v_m = σ_ω² / (2γ(1 − Λ_m))`.
- Expected payoff vs Pe and the check `W = 1.5 (1 − 2 Pe)²` above Nash.

**Checks (write `code/learn/test_core.py`, run with pytest):**
1. `"sign"` mode reproduces `sim_fig1.py`'s covariance eigenvalues at a few β (e.g. 2, 6, 12) within sampling error, using the same seed protocol.
2. Large fixed k in `"shared"` mode matches `"sign"`.
3. Fixed k sweep (no learning): measured onsets of each eigenvalue (3× floor criterion as in `sim_fig1.py`) agree with `k_c^(m)` from `theory.py` at β = 6 and β = 12. Plot `figures/learn/fixed_k_sweep.pdf`: eigenvalues vs k with linear-theory overlay. This validates g_eff before any learning.
4. At k = 0 the average realized payoff ≈ 2.5 (mixed Nash).

---

## Phase 2 — Learning rules

`code/learn/learners.py`, all updating once per `step` with batch = all agents × contexts:

1. **Myopic policy gradient (PG):** `k += ε dt · mean[(r − b) · ∂_k log π]`, same for θ. Baseline b = batch mean reward per context (note the O(1/N) bias). Options: `shared` (average over agents) or `independent` (each agent uses only its own encounter, with a per-agent running baseline).
2. **Analytic mean-field gradient:** given current ω and partners, compute the expected-payoff gradient exactly (Gaussian expectations over read noise by quadrature). Noise-free reference for PG.
3. **Evolution strategies across replicas (ES):** R replica populations (e.g. R = 8, smaller N allowed, e.g. 500) at shared k ± δ; each runs for a window ≫ 1/γ (e.g. 20/γ after a burn-in); update `k += ε_ES (W̄₊ − W̄₋)/(2δ)`. This is the between-population selection for H4.

**Check:** derive the small-k gradient analytically in `notes/learn_results.md` (H1 formula), and compare it with the analytic-gradient learner and with PG averaged over seeds at fixed small k (say k = 0.05, 0.1, 0.2). Report the measured coefficient against 3·u_m·v_m.

---

## Phase 3 — Experiments

Common settings: Phase 1 parameters, β ∈ {2 (subcritical even for the sign rule), 6 (two modes supercritical at k_max), 12 (all three)}, random pairing, ≥ 5 seeds each. Learning rates chosen so the learning timescale spans faster and slower than 1/γ; report measured `(dk/dt)/k` relative to γ for each run rather than ε alone.

| Exp | Tests | Setup | Primary outputs |
|-----|-------|-------|-----------------|
| E1 | H1 | shared PG and analytic, k0 ∈ {0, ±0.02, ±0.05}, β = 6 | k(t) on log axis; fitted early growth rate vs 3 ε u_1 v_1; k = 0 exactly stays at 0 (analytic) |
| E2 | H2 | shared PG, k0 = 0.05, β = 6 and 12, ε grid | eigenvalues of C vs k(t), overlaid with fixed-k curves from Phase 1 check 3 and `k_c^(m)` lines; k(t) showing acceleration at ignition; payoff vs time; adiabaticity: measured v(t) vs v(k(t)) from the fixed-k sweep |
| E3 | H3 | as E2 at β = 2 | k grows, eigenvalues stay at floor, payoff ≈ Nash + small |
| E4 | H4 | k0 = −0.5 and +0.5; PG (shared) vs ES; β = 6 | k(t) for both learners from both starts; long-run W at k = ±|k| (expect W(−k) < W(k)); eigenvalues showing anti-follow suppresses v below floor |
| E5 | H5 | independent vs shared PG, k0 = 0.05, β = 6 | after convergence: θ_i and k_i vs agent status rank along g_1; payoff per agent vs rank; aggregate payoff, ignition time |
| E6 (stretch) | H6 | independent, per-agent discounted returns over horizon H ∈ {1/γ, 5/γ, 20/γ} (REINFORCE over trajectory segments) | θ_i vs status rank; collision rate (G,G); status mobility |

Scripts: `code/learn/exp_E1.py` … `exp_E6.py` writing `data/learn/E*.npz`; `code/learn/make_learn_figs.py` writing `figures/learn/E*.pdf`.

---

## Phase 4 — Checks and write-up

1. Re-run each experiment's headline number from a fresh seed and confirm it holds.
2. Recompute key theory numbers (`k_c^(m)`, g_eff(∞), β_c, H1 coefficient) in a standalone cell and compare to what is reported.
3. Open every figure and check axes, labels, overlays.
4. `notes/learn_results.md`: one section per hypothesis — prediction, result, figure, verdict (supported / refuted / inconclusive), and caveats. Include anything that contradicted the plan, especially H4.

---

## Known pitfalls to watch for

- Near k = 0 the PG signal is tiny (∝ v at the noise floor ≈ 0.03); with sampled actions it can be swamped by noise. Use the analytic learner to separate signal from estimator noise before concluding anything about E1.
- `sim_fig1.py` plays all contexts every step with the same partner and has no 1/M; keep that convention or document any change.
- Fast learning makes the k sweep non-adiabatic (the paper's rate-induced regime, ratio τ_k · k̇); compare against fixed-k curves rather than assuming quasi-static tracking.
- Schemas are fixed here (Model 4, no resources). Coupling learning to dynamic schemas is a follow-up; only attempt it if Phase 0 finds an existing resource/schema simulator to build on.
