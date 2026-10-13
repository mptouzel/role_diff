# Learned role-following policy — results of a reduced run

Run on a fresh clone of `mptouzel/role_diff` (commit at main, 2026-10-13). No existing file was modified.
New files: `code/learn/{core,exp_fixed_k,exp_learn,make_learn_figs}.py`, `data/learn/*.npz|log`,
`figures/learn/learn_summary.{pdf,png}`.

Scope: Phase 1 checks, H1–H4 of `PLAN_learned_policy.md`. H5 (independent learners) and H6 (status
climbing) were not run. Policy learns k_m and θ_m per context; θ stayed at 0 within ±0.05 in every run.

Setting: identical to `sim_fig1.py` (N=2500, d=M=3, γ=1, σ_read=0.6, σ_ω=0.25, dt=0.02, fixed
hierarchical schemas, random pairing). Policy P(Go|s)=½(1+tanh(k(s−θ))). Myopic learner = per-agent
REINFORCE on the encounter payoff with a batch-mean baseline, shared parameters, update every step.

## Core checks — pass

| check | result |
|---|---|
| score function vs finite differences | agree to 1e-10 |
| k=0 payoff | 2.5004 ± 0.0009 (mixed Nash 2.5) |
| sign mode vs committed `fig1_data.npz`, β=2/6/12 | eigenvalues within 2% (e.g. β=6: 0.571/0.158/0.057 vs 0.582/0.162/0.058) |
| large-k tanh vs sign | k=20 payoff 2.732 vs sign 2.734 (β=6) |
| g_eff(∞) | 1.3298 = √(2/π)/σ_read |

## H1 — mixed Nash is an unstable stationary point: **supported, formula exact**

Exact (quadrature) mean gradient at fixed small k divided by k matches 3·u_m·v_m to within 1–2% for
|k| ≤ 0.12 at all β and all contexts (fig. b), and equals 1.5·E[Δ²] as derived. It falls below the
prediction as k grows (0.104 vs 0.129 at k=0.37, β=6) because tanh saturates. In the learning runs the
fitted early log-growth rate of k_1 at ε=1 was 0.089 vs predicted 0.084 (β=6), 0.228 vs 0.207 (β=12).

## Side finding — the simulated noise floor is mostly encounter shot noise

At k=0 the status variance is 0.037 / 0.099 / 0.243 along context 1 at β = 2 / 6 / 12, versus the
flat floor σ_ω²/2γ = 0.031. The step kicks every agent by β·a·g·dt with a random sign each step, i.e.
encounter rate ν=1/dt=50, so the shot-noise intensity β²u_m/ν exceeds σ_ω² for β ≳ 2. Appendix A says
the flat floor requires β²/ν ≪ σ_ω², i.e. β ≪ 1.8 here, so the Fig. 1 sweep (β up to 21) runs outside
that regime. Consequences seen here: flat-floor linear theory under-predicts subcritical v by ~3× at
β=6 (fig. a, dotted), and the onset near k_c is smoothed rather than divergent. This touches the
paper's flat-floor and subcritical-inversion arguments (Secs. II A, V), not the thresholds themselves.
Worth checking directly in `sim_fig1.py` with smaller dt.

## H2 — learning is the sweep: **supported, except the "acceleration" part**

From k₀=0.05 the learned k_m rise through k_c in strength order (β=6: k_c=0.67, 1.25 crossed at
t≈27 and 73; context 3 unreachable at β=6 still grows to ≈1.4; β=12: all three crossed at t≈8, 19, 192).
Learning is slow relative to γ and the population tracks the fixed-k curve: median |v − v_fixed(k)|/v
≈ 2% for contexts 1–2 (fig. d). Final payoffs approach the sign rule (β=6: 2.705 vs 2.734).
Not supported: k does not accelerate after ignition. d ln k/dt is flat (~0.1 at β=6) up to k_c and
then decays; the absolute dk/dt peaks just after k_c (0.07) and falls, because saturation of tanh
outweighs the growth of v.

## H3 — cognitive floor: **supported**

β=2 (sign-rule Λ_1=0.76): k grows to ≈3 (ε=1) without any mode igniting; v_1 rises from 0.037 to 0.073;
payoff 2.538 vs Nash 2.5 (ceiling at k=∞: 2.559). A sub-threshold signal still pays a little.

## H4 — sign selection: **myopic part supported; ES part refuted**

- Myopic PG from k₀=0: the sign of each context's k is a coin flip — 13/30 context-runs went to
  follow in the main run, 16/36 in a fresh-seed rerun. From k₀=−0.5 all 3 seeds stay anti-follow.
- Anti-follow is a real, stable learned outcome: |k|→3, status variance pushed *below* the k=0 level
  (0.041 vs 0.099 at β=6), payoff 2.53 vs 2.73 for follow. Follow's gain over Nash is ~6× larger.
- ES across replica populations (local finite differences of long-run payoff, N=500, 30/γ windows)
  started at k=−0.5 also moves *away* from zero (−1.27 after 20 iterations); from +0.5 it goes to +2.9.
  Reason (fig. f): long-run payoff W(k) increases with |k| on both sides; follow's advantage is
  global, not local. The plan's claim that population-level selection would flip k was wrong for any
  local population-selection rule.
- Implication: in this model nothing that follows a local payoff gradient — individual or
  population — prefers follow over anti-follow. Follow is selected by the initial bias, or would need
  a non-local comparison (whole societies competing, large mutations across k=0). Untested here.

## Caveats

- Single repertoire geometry, random pairing, T=1, fixed schemas (no resources), 2–12 seeds per cell.
- The "analytic" gradient uses Gauss–Hermite quadrature and was only used for |k| ≤ 0.5.
- ES used a scalar k shared across contexts, δ=0.2, learning rate 2; other ES settings not explored.
