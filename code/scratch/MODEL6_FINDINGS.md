# Model 6 simulation: findings

Code: `sim_model6.py` (closed loop), `diagnose_portfolio_geometry.py` (fixed-schema
cascade depth vs portfolio geometry).

Conventions used, matching the paper: one random perfect matching per step
(dense-encounter limit, nu=1); all M contexts played per encounter at f_m=1 with
the uniform constant absorbed into beta; hard-sign decisions on noisy pairwise
gaps; f_m not renormalized as schemas go extinct.

## 1. The budget must be imposed, and as an inequality

The replicator does not conserve sum_m u_m. Left unconstrained every schema
saturates and Theta pins at 1, so the other regimes are unreachable. Imposing it
as an equality (proportional projection onto S_bw) is also wrong: it pushes
schemas above u_sat and manufactures exact degeneracy. Correct: project only when
sum_m u_m > S_bw.

## 2. The ceiling implementation changes the result qualitatively

Soft gate (1 - u_m/u_sat) on the growth term  -> spectrum collapses to exact
equality. Hard clip at u_sat (what the paper specifies) -> graded hierarchy,
76x spread, same parameters. The soft form adds a term -lambda_m/u_sat that
exceeds the destabilizing d lambda_m/d u_m by ~7x, so it stabilizes the pinned
state. The regime map is NOT robust to this choice; only the hard cap reproduces
the theory.

## 3. Escape from the pinned state is slow

Condensation rate at the symmetric point is 2 alpha v (M-1)(1-1/d)/M^2, smaller
than the growth rate by roughly 1/M (measured ratio 0.04-0.19). The system fills
the budget, sits in the degenerate state, and escapes only much later. The claim
that the pinned fixed point "is a repeller, so it is never occupied" overstates
this: it is a long-lived transient.

## 4. Cascade depth follows the gain spectrum, not the strengths (theory confirmed)

Fixed-schema sweeps at beta=3, geometric strengths u = 2.6 exp(-k/2.2):

| portfolio                  | mu_k (top 4)              | predicted depth | observed |
|----------------------------|---------------------------|-----------------|----------|
| near-orthogonal M=d=6      | 3.46 2.19 1.39 0.88       | 4               | 4        |
| random M=d=6               | 4.36 3.10 0.97 0.37       | 3               | 2        |
| random overcomplete M=8 d=6| 4.37 3.13 1.09 0.40       | 3               | 2        |
| coherent chi=0.7, M=d=6    | 6.78 1.09 0.52 0.38       | 2               | 1        |

Depth tracks mu_k = eig(2 pi'(0) sum_m u_m ghat_m ghat_m^T), not {u_m}.
Orthogonality is what makes the two coincide; non-orthogonal directions turn a
geometric strength hierarchy into a cliffed or spiked gain spectrum. Observed
depth runs one mode short of the linear prediction because the quartic
confinement raises later thresholds (differentiation along the leading axis adds
gamma|w|^2 to the effective decay of every other axis).

## 5. Degenerate spectra do give a simultaneous avalanche (class (i) confirmed)

Exactly degenerate, orthogonal portfolios: all eigenvalues rise together
(d=3, u=1: 2.63/2.58/2.46 at beta=0.8 -> 11.95/11.46/11.10 at beta=2.5).

## 6. Niche partitioning holds subcritically and is destroyed at ignition

Seeded with M=6 orthogonal candidate directions in d=6, the directions stay
distinct (max |overlap| 0.02-0.10) through three decades of strength growth,
then collapse to |overlap| = 1.000 within ~8 time units of ignition.

Ordering is robust to the initial strength asymmetry -- ignition always comes
first, collapse follows:

| seed spread | t(collapse) | t(ignition) | order          |
|-------------|-------------|-------------|----------------|
| 0.0         | 132         | 124         | ignition first |
| 0.2         | 125         | 117         | ignition first |
| 0.5         | 113         | 103         | ignition first |
| 0.8         | 102         |  93         | ignition first |

So partitioning is real and robust where the paper derives it (the sub-critical
Gaussian regime, where C is near-isotropic and C_r inherits anisotropy from the
portfolio). It fails once the leading axis differentiates: C is then dominated
by that axis, C_r inherits the dominance, and the power iteration alpha C_r g_m
pulls every schema onto it.

The failure is self-accelerating, which is why runs nominally below threshold
also ignite. Collapse concentrates the whole budget on one direction, so mu_1
jumps from 2 pi'(0) u_max to 2 pi'(0) sum_m u_m (observed 0.57 -> 9.97, a factor
17), pushing the system further past threshold. Collapse and ignition are one
event.

Consequence: the thick phase, meaning several simultaneously active role axes,
was not reached from an orthogonal start in any run. Sustained multiplicity
appears to require something outside the closed loop, which is where the paper's
Outlook already places it -- but this makes it a forced conclusion rather than a
scope preference.

## 7. Pinning, condensation and capacity are confirmed

These are separate from (6): they concern the strength dynamics, not the
directions, and they behave as the paper describes.

Ceiling non-binding (u_sat = 50, S_bw = 7.5): the system condenses rather than
equalizing -- final u = [4.97, 2.40, 0.06, 0.06, 0.02, 0.00]. The equal-strength
interior fixed point is therefore not an attractor, exactly as the pinning
argument predicts (it is a saddle; condensation is what is observed).

Ceiling binding: condensation is arrested into a cluster of saturated schemas,
and the count matches M_cap = S_bw/u_sat quantitatively.

| u_sat | S_bw | M_cap predicted | # at ceiling | final u (sorted)                       |
|-------|------|-----------------|--------------|----------------------------------------|
| 2.5   |  5.0 | 2               | 2            | 2.43 2.43 0.06 0.06 0.02 0.00          |
| 2.5   | 12.5 | 5               | 5            | 2.35 2.35 2.35 2.35 2.35 0.75          |
| 2.5   |  7.5 | 3               | 2 + graded   | 2.39 2.39 1.19 1.13 0.35 0.06          |

The intermediate case sits between a saturated cluster and a graded tail, which
is the E1/E3 boundary region. The budget is saturated in every case (sum u_m =
S_bw to three figures).

So the strength sector of the closed loop -- pinning as an unstable fixed point,
condensation as the realized dynamics, and the ceiling converting runaway into a
cluster of M_cap schemas -- is validated. It is the direction sector that fails
past ignition.
