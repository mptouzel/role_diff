# Plan: integrate the learned-policy results into the paper

This file is committed at the root of the `learned-policy` branch, alongside all the code, data and
figures it refers to. In your local `role_diff` clone, check out that branch, start Claude Code, and say:
"Read PLAN_paper_integration.md and carry it out phase by phase. Stop at each STOP and show me the diff."

Design rule: **streamlined**. The new material lives in one new Supplement subsection (S1 D) with one
figure. The main text gets about 150 words in three places. Existing claims that the results contradict
get corrected. No new main-text figure, no new top-level SM section (that would renumber S2–S10, which the
main text cites as plain text through `\smref{S…}`).

---

## What the paper gains (the content to integrate)

Results are on β = 6 unless stated. Source: `notes/learn_results.md` in the bundle.

1. **The sign rule can be learned, not assumed.** Policy family π_ℓ(s) = ½[1 + tanh(ℓ s)]: ℓ = 0 is the
   mixed Nash equilibrium (2.5), ℓ → ∞ is the paper's sign rule. The slope sets the read gain,
   g(ℓ) = ℓ E_ξ[sech²(ℓ σ_read ξ)], with g(∞) = √(2/π)/σ_read (T = 1), so Λ_k ∝ g(ℓ).
2. **Mixed Nash is an unstable stationary point of payoff learning.** The myopic payoff gradient vanishes
   at ℓ = 0, and for small ℓ, dℓ/dt ∝ (3/2) E[Δ²] ℓ = 3 u_m v_m ℓ under random pairing. Measured: within
   1–2% for |ℓ| ≤ 0.12, all β and contexts.
3. **Learning carries the population through the cascade.** Learned ℓ_m crosses the thresholds ℓ_c^(m)
   (where Λ_m(ℓ) = 1) in strength order. The population tracks the quasi-static fixed-ℓ curve
   (median deviation about 2%) and ends near sign-rule payoff (2.705 vs 2.734).
4. **The orientation of the convention is not selected.** The game is symmetric under ℓ → −ℓ.
   - Anti-follow (defer to lower status) is the anti-Bourgeois convention. It is locally stable, and the
     write-back suppresses status variance below the role-free level: payoff 2.53 vs 2.73.
   - From ℓ₀ = 0 the sign is a coin flip, per context: 13/30, then 16/36 on fresh seeds.
   - Local selection between populations (evolution strategies on long-run payoff) also fails to flip an
     anti-follow start. Payoff increases with |ℓ| on both branches.
   - This is Selten's result for role-asymmetric games, and the Bourgeois / anti-Bourgeois pair of
     Maynard Smith's Hawk–Dove analysis with an uncorrelated asymmetry.
5. **A payoff-relevant (correlated) asymmetry selects it.** Collision payoff −1 + c·tanh(Δ/D₀), D₀ = 0.5.
   This is antisymmetric within a pair, so welfare is exactly unchanged.
   - Gradient at ℓ = 0 = (c/4) E[Δ tanh(Δ/D₀)] > 0. Match is exact to 4 digits.
   - Follow fraction from ℓ₀ = 0: 29/30 at c = 0.1, 30/30 at every c ≥ 0.25.
   - Cascade in strength order: 10/10 runs at every c.
6. **An entrenched wrong convention persists.** Anti-follow keeps a basin ℓ < ℓ*(c), with
   |ℓ*| ≈ c/(6D₀). Measured 0.071, 0.145, 0.301, 0.705 against predicted 0.083, 0.167, 0.333, 0.667, for
   c = 0.25, 0.5, 1, 2. From ℓ₀ = −0.5 learning escapes only at c = 2; from ℓ₀ = −3 it never escapes.

**Headline for the paper.** Roles can emerge from scratch, from the mixed equilibrium, under
reward-maximizing learning, provided status matters even slightly to who prevails in a collision. Local
learning does not reform an entrenched inverted convention.

---

## Phase 0 — set up (no paper edits)

1. Confirm you are on branch `learned-policy` and that it sits on top of `main` with three commits that
   add only new paths: `code/learn/`, `data/learn/`, `figures/learn/`, `notes/`,
   `PLAN_learned_policy.md`, and this file. Check with `git diff --stat main...learned-policy`: no
   existing file may be modified. If `main` has moved on since, rebase the branch onto it first.
   Read `notes/learn_results.md`; it is the record of every number quoted below.
2. Build `.venv` per the README (pinned matplotlib 3.8.4) and run
   `.venv/bin/python code/learn/make_learn_figs.py` and `make_tilt_fig.py`. Both must run from the committed
   `.npz` files without re-simulating.
3. Read `supplement.tex` S1 (`sm:ladder`, `sm:payoff`, `sm:ce`), `main.tex` "The game", "Agents, game
   contexts…", Appendix A (`app:micro`), the Discussion, and `sm:simdetails`. Note: the SM cross-references
   main-text sections by plain-text number, and the main text references SM sections and equations by
   plain-text number (`\smref{S2}`, `\smeq{S25}`).
4. **Pick symbols.** `k` is the mode index and `\kappa` is taken. Proposed, verify they are free in both
   files: policy slope `\ell`, collision tilt `c_{\mathrm{col}}`, tilt scale `D_0`. Add macros to both
   preambles only if used more than twice.

**STOP:** report symbol choices and any surprises.

---

## Phase 1 — code and figure into repo conventions

1. Move the paper-relevant scripts into the repo's pattern of a sim script that writes `.npz` and a make
   script that reads it:
   - Keep `code/learn/core.py`, `exp_fixed_k.py`, `exp_learn.py`, `exp_tilt.py` (simulation).
   - Add `code/make_fig_learning.py` (plotting only), modeled on the existing `make_*` scripts, including
     their matplotlib-version warning.
   - Move `notes/learn_results.md`, `PLAN_learned_policy.md`, `make_learn_figs.py` and `make_tilt_fig.py`
     into `code/scratch/learn/` (the README says scratch is outside the submission).
2. Keep in `data/learn/` only what the figure and text use: `fixed_k.npz`, `learn_sweep.npz`,
   `learn_sign.npz`, `learn_es.npz`, `tilt_basin.npz`, `tilt_learn_seed300.npz`. Delete `*.log`.
3. **Fig. S-learn**: four panels, one column width ×2, style matched to the existing figures.
   - (a) Small-ℓ growth rate: exact gradient/ℓ vs 3 u_m v_m, identity line (from `fixed_k.npz`).
   - (b) Learned ℓ_m(t) at β = 6 and 12 with ℓ_c^(m) lines; inset or twin panel of v_m vs ℓ, learned dots
     over the fixed-ℓ curve (from `learn_sweep.npz`, `fixed_k.npz`).
   - (c) Orientation: ℓ₁(t) from ℓ₀ = 0, 10 seeds at c = 0 (splits) and at c = 0.1 (all follow) (from
     `learn_sign.npz`, `tilt_learn_seed300.npz`).
   - (d) Basin edge |ℓ*| vs c with the line c/(6D₀); markers per context (from `tilt_basin.npz`). Plot only
     |ℓ| ≤ 4: beyond that the 40-node quadrature is unreliable.
   Write `figures/fig_s_learning.pdf`.
4. `Makefile`: add `make_fig_learning.py` to the `figures` target (reads committed data). Add a separate
   `learn-data` target that reruns the four sim scripts (~1.5 h). Do not put the sims in `figures`.
5. README "Figure sources" table: add the row for the new SM figure.

**STOP:** show the figure.

---

## Phase 2 — Supplement

### 2a. New subsection S1 D, right after `sm:ce`

`\subsection{Learning the policy and selecting the convention}\label{sm:learn}`, about 700–900 words, the
figure, no more than 4 numbered equations. Content in this order (numbers above):

1. Policy family π_ℓ and g(ℓ); Λ_k(ℓ) = Λ_k|_{sign} · g(ℓ)/g(∞); thresholds ℓ_c^(k). Learning the slope is a
   sweep along the same axis as β.
2. Myopic payoff gradient. One equation: dE[r]/dℓ = E[∂_ℓ π̄ (3 − 6q)], giving the small-ℓ result
   (3/2)E[Δ²]ℓ = 3u_m v_m ℓ. Statement: mixed Nash is a stationary but unstable point; the population leaves
   it on its own because v_m > 0 at the noise floor. Fig. panel (a).
3. Learning through the cascade: strength-ordered crossings, quasi-static tracking, final payoff. Panel (b).
4. Orientation symmetry. ℓ → −ℓ; anti-follow is a self-enforcing correlated equilibrium as well;
   write-back makes it self-suppressing but still locally stable; coin-flip statistics; local selection
   between populations fails (one sentence on the ES test). Connect to Selten (1980) and to Maynard Smith's
   Bourgeois / anti-Bourgeois with an uncorrelated asymmetry. Panel (c), c = 0 traces.
5. Correlated asymmetry: define the tilt, its welfare neutrality, the gradient (c/4)E[Δ tanh(Δ/D₀)], the
   basin edge |ℓ*| ≈ c/(6D₀) (one line of derivation: tilt balanced against 1.5E[Δ²]ℓ), and the selection
   numbers. Panels (c) and (d).
6. Two-sentence conclusion: emergence from the mixed equilibrium requires only that status matter weakly to
   who prevails in a collision; an entrenched inverted convention is not undone by local learning unless
   |ℓ| < c/(6D₀).
7. Scope line: fixed schemas (Model 4 level of the ladder), random pairing, T = 1, β = 6 for the convention
   runs, shared policy learners. Independent learners and status climbing are not treated.

### 2b. Correct S1 C (`sm:ce`)

These sentences are now wrong or incomplete; replace them, and keep the rest of the paragraph:

- "That adoption need not be assumed: replicator dynamics on strategies … and the same selection would make
  following an outcome here." **Replace with:** imitation selects among strategies when the correlating
  device is fixed. Here the device is co-produced by the actions, and both orientations of the convention
  are self-enforcing. Payoff learning reaches the role-following policy from the mixed equilibrium, and
  which orientation it reaches is fixed by any payoff-relevant asymmetry (S1 D).
- "Against a partner who also follows, the policy used here is therefore the one any algorithm that
  maximizes reward would find." **Keep, and add:** "…and such an algorithm reaches it from the mixed
  equilibrium (S1 D)."
- Last paragraph (role asymmetry 6 vs 2): add one clause noting that the incentive of the lower-paying role
  to climb by over-playing Go is not tested by the shared-policy learners of S1 D.

### 2c. Bibliography (verify each entry against the source before adding)

- R. Selten, "A note on evolutionarily stable strategies in asymmetric animal conflicts,"
  J. Theor. Biol. 84, 93 (1980).
- J. Maynard Smith and G. A. Parker, "The logic of asymmetric contests," Anim. Behav. 24, 159 (1976).
- J. Maynard Smith, *Evolution and the Theory of Games* (Cambridge University Press, 1982).

**STOP:** compile `supplement.tex`, show the new subsection, the S1 C diff, and the rendered figure.

---

## Phase 3 — main text (about 150 words total, no new figure)

1. **"The game"**, after "(\smref{S1\,C})": one sentence. Payoff-maximizing learning reaches role-following
   from the mixed equilibrium, and a weak dependence of collision cost on status selects its orientation
   (\smref{S1\,D}).
2. **"Agents, game contexts…"**, after the sentence defining the read gain g = √(2T/π)/σ_read: one sentence.
   The sign rule is the limit of a learnable family whose slope sets g, so learning the policy moves the
   population along the same axis as β (\smref{S1\,D}).
3. **Discussion.** Add one short paragraph, *Where the convention comes from* (≤ 90 words), after
   "Monitoring in practice" or folded into "Other Extensions":
   - learning supplies the gain but not the orientation;
   - a payoff-relevant status asymmetry supplies the orientation;
   - inverted conventions, once established, persist under local learning;
   - in the distributional-AGI reading, reward-trained agents can lock into either convention, so the
     platform's collision costs, not agent capability, decide which.
4. Re-check every plain-text SM cross-reference in `main.tex`. Adding S1 D renumbers nothing, but confirm
   that the new SM figure does not shift any SM figure number cited as plain text.

**STOP:** compile `main.tex`, report the word-count change, show the diffs.

---

## Phase 4 — a separate issue found along the way (needs your decision; do not fix silently)

`sm:simdetails` says Fig. 1 integrates "the dense-encounter limit of Eq. (1)". `code/sim_fig1.py` does not.
It samples one hard ±1 action per context per step and kicks ω by β·a·g·dt. That is a finite-ν encounter
process with ν = 1/dt = 50 and shot-noise intensity of order β²u_m/ν.

Appendix A states the flat floor needs β²/ν ≪ σ_ω², here β ≪ 1.8. Fig. 1 sweeps β to 21. Measured at
ℓ = 0 (no signal following), the status variance along the strongest axis is 0.037 / 0.099 / 0.243 at
β = 2 / 6 / 12, against the flat floor 0.031.

Appendix A also argues the thresholds are unaffected, which matches the reproduced onsets. Two options:

- **(A) Describe it.** Correct the `sm:simdetails` wording to say what is simulated, and add one sentence
  that Fig. 1 runs in the shot-noise regime of Appendix A, where onsets are unchanged and pre-onset
  amplitudes exceed the flat floor.
- **(B) Make it match.** Rerun `sim_fig1.py` in the dense limit: replace the sampled action by its noise
  average erf(Δ/√2σ_read), or shrink dt so that β²dt ≪ σ_ω². Regenerate Fig. 1 and confirm the onsets.

Option (B) changes a committed figure. It is your call.

**STOP:** ask which option before touching Fig. 1, `sim_fig1.py`, or `sm:simdetails`.

---

## Phase 5 — final checks

1. `make figures PY=.venv/bin/python` and `make pdf` run clean. No undefined references or citations.
2. Every number in S1 D matches `code/scratch/learn/learn_results.md` or is recomputed from `data/learn/`;
   list any you could not trace.
3. Re-derive two results independently in a scratch cell: the small-ℓ coefficient 3u_m v_m, and
   |ℓ*| ≈ c/(6D₀).
4. Commit on `learned-policy` in logical commits (code+figure, SM, main text, Phase 4 if done).
   Do not push or merge without asking.
