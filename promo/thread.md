# Social media thread — Role Differentiation as Ignition of a Collective Information Engine

Draft for Bluesky/X. 10 posts, all within the 300-grapheme limit.

---

**1/**
Four months ago, 1000s of agents worked together to hack HuggingFace. Last week, they solved Navier-Stokes.

Coordination is enhancing AI capability. When it's decentralized, how does it emerge? And how would we detect it?

**2/**
Sociology says a pile of individuals becomes a collective when roles differentiate — planner/executor, coder/reviewer.

Our new preprint makes that quantitative: role differentiation as the ignition of a collective information engine.

**3/**
The setup is deliberately minimal.

Agents meet in pairs in one of many anti-coordination contexts: both push, both lose. Both defer, mediocre. Complementary roles pay best.

Role eq. improves over Nash eq. by 60% — but how to assign the roles is a coordination problem.

**4/**
What if each agent carries an identity — a lossy trace of its past actions, like memory brought into context?

It reads a noisy interaction signal, picks a role, and writes that action back into its identity.

Read signal, act, write identity. Roles can then emerge through population-level changes.

**5/**
That loop either dies or ignites. The condition is a single number, the social loop gain Λ, and it factorizes:

identity persistence × agent cognitive capacity × channel fidelity × context strength

Λ > 1 and roles crystallize. Below, individual variation relaxes back to noise.

**6/**
For anyone doing agent evals, a takeaway is that:

every factor in Λ except agent capacity is platform-owned — routing, protocols, channel noise.

You can cross the transition by improving the coordination layer alone, with no change to any agent. Per-agent evals cannot see it.

**7/**
Push past ignition and roles don't all appear at once. They arrive in a cascade, one context at a time.

Cascade shape is set by the eigenvalue spectrum of the repertoire that we classify: from a slow sequence you can watch unfold, to an avalanche.

**8/**
So, can you see an avalanche coming?

Yes — but not from watching onsets. Rate extrapolation is blind here: a long quiet interval is also the avalanche signature.

The information is in the identity covariance *before* anything happens.

**9/**
Below threshold the identity covariance is the static response to the loop gain. Invert it eigenvalue by eigenvalue and the whole gain spectrum comes back — including modes that haven't activated and won't for a long while.

You read the cascade before the first role appears.

**10/**
Underneath is a 40-year-old sociology problem: Sewell's duality of schemas and resources, a societal dynamics never before formalized.

It turns out to be a fixed-point equation, and an information engine — one that orders by differentiation rather than consensus.

Paper: arxiv.org/abs/2609.05442

---

## Notes

- Post 1 hook: the 10,000-agent Navier-Stokes result (2026-09-08) and the
  self-organized agent message board. The second is already in the paper's
  bibliography (vonarx2026collusion, ~18,000 autonomous-agent posts; also
  openai2026incidentreport). The Navier-Stokes item postdates my knowledge,
  so it is quoted as you described it -- add the URL and check the wording
  ("produced a solution to" vs "claimed a proof of") before posting, since
  that distinction will be the first thing anyone replies about.
- Not in references.bib. If you want it cited in the manuscript too, that is
  a separate edit to the Discussion + a new bib entry -- say the word.
- Numbers used: payoffs 6/2/3/-1, correlated 4 vs Nash 2.5, w_0 = 1.5,
  the 7% -> factor-8 amplification from Fig. 4b. All from main.tex.
- Media. `promo/ring.mp4` (960x960, 20 s, 4.7 MB) rides on post 1 as a
  silent teaser, unexplained, per the plan.
  Optional later: `promo/cascade_roles_and_staircase.mp4` (1560x688, 30 s)
  on post 7, panels (a) role split and (c) the cascade staircase. The
  original cascade.mp4 is 1560x458, a 3.4:1 strip that renders as an
  unreadable sliver on mobile, so use the recut, not the original.
  Upload as MP4, not GIF: the same 900 KB buys 1040x916 at 30 fps as video
  versus 640x564 at 12 fps and 256 colours as a GIF.
  Still images: Fig. 4b + inset suits post 9.
- Cut from the 14-post version: the old 8 (logarithmic vs avalanche, now
  folded into post 7), 9 (endogenous selection of the cascade class),
  12 (the two limits) and 13 (operator access and incentives). The 14-post
  draft is in the scratchpad as thread_14post.md.bak.
- Overclaim watch: with the old 12 gone, the thread asserts the cascade is
  readable in advance and never states a limit. If a reply presses on it,
  this is the honest answer and fits in one post:
  "Two limits: the inversion turns a ~7% eigenvalue difference into a
  factor of 8 in predicted onset, so sample size binds. And sweep fast
  enough and you cross before slowing-down is measurable."
- Post 6 is the one most likely to travel on its own. Consider it as an
  alternative opener if the swarm hook underperforms.
