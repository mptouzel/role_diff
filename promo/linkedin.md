# LinkedIn post — Role Differentiation as Ignition of a Collective Information Engine

Single post. Link goes in the FIRST COMMENT, not the body (see notes).
The first ~210 characters are what shows before "…see more" — the fold is
marked below. Everything above it has to earn the click.

---

Four months ago, 1000s of agents worked together to hack HuggingFace. Last week, they solved Navier-Stokes. What does this coordination enable next? When decentralized, how do we detect it?

<<< FOLD — roughly what a desktop reader sees before "…see more" >>>

Capability increasingly comes from coordination, not in any single agent.

That raises a question our evaluation practice cannot currently answer: when does a population of agents stop being a pile of individuals and start behaving like an organization, with stable complementary roles?

Our new pre-print puts that question on a phase diagram.

The model is deliberately toyish. Agents meet in pairs and play an anti-coordination game, where complementary role-play pays more than acting alone. Nobody assigns the roles. Each agent carries a persistent identity that summarizes its own past actions; in an encounter it reads a noisy social signal relative to its partner, picks a role, and writes that choice back into its identity. Read, act, write.

That loop either dies out or ignites, and the condition is a single number that factorizes into four terms: identity persistence, agent cognitive capacity, communication channel fidelity, and the strength of the task repertoire. Above 1, roles crystallize on their own.

Some points worth mentioning for AI safety practice:

1. Every factor except agent capacity is owned by the platform, not the agents. You can cross the transition by improving routing, protocols, or channel noise alone, with no change to any model. Per-agent capability evaluations will not see it coming.

2. Once ignited, roles do not arrive all at once. They come in a cascade whose shape is set by the spectrum of the task repertoire, ranging from a slow readable sequence to a single avalanche with no precursor in the onset times. The system dynamics selects which type.

3. An avalanche is still foreseeable, but not by watching onsets. The covariance of agent identities, measured before anything happens, carries the whole spectrum. Invert it and you can read the cascade in advance.

The inversion is numerically sensitive, so the number of agents you can sample is the binding constraint. And if the system is pushed across the threshold quickly, you cross before the warning signal is measurable at all. Fast deployment destroys early warning.

Underneath all of this is a 40-year-old problem in sociology, Sewell's duality of schemas and resources, for which our paper gives the first-ever model. It turns out to be a fixed-point equation, and an information engine, one that produces differentiation rather than consensus.

Lots more in linked paper.

---

## First comment

Paper: https://arxiv.org/abs/2609.05442
Blog:
Grateful for conversations with John Bechhoefer and Miloš Bročić, and for
support from the Future of Life Institute and IVADO.

---

## Notes

- Link in first comment: LinkedIn's feed suppresses posts carrying external
  links in the body. Put the arXiv abs page (not the PDF) in the comment and
  keep "Link in the comments" as the closing line. If you would rather have
  it inline, delete that line and accept some reach cost -- both are common.
- Do NOT link this post from Bluesky/X, or those from here. Each platform
  points at the arXiv page directly. Cross-linking socials costs a click and
  reads as engagement farming.
- The fold marker is not part of the post -- delete it before pasting. Only
  the first ~210 chars (desktop) / ~140 (mobile) show before "…see more",
  so the Navier-Stokes line has to carry the click on its own.
- Timing: post both on the same day, LinkedIn on a weekday morning in your
  audience's timezone. Reply to comments in the first hour or two.
- No hashtags in the body. If you want them, 3 at most at the very end;
  #AISafety #MultiAgentSystems #StatisticalPhysics would be the set.
- Register is deliberately less technical than the thread: no Λ, no
  eigenvalues, no ω-identity. LinkedIn's audience overlaps less with the
  arXiv one and skews toward policy/industry.
- Em-dashes: I have used them sparingly here on purpose. Heavy em-dash use
  now reads to many people as LLM-generated text, which is a bad look on a
  post announcing your own work. Worth a scan before posting.
