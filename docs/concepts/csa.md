# CSA

Conformational Space Annealing (CSA) is `variopt`'s most opinionated built-in
population optimizer. It keeps a diverse elite archive — the *bank* — and
schedules perturbations against that archive while a distance cutoff anneals
from exploration toward exploitation.

The public entry point is:

```python
from variopt.algorithms.population import CSAOptimizer
```

Use `CSAOptimizer.from_space_defaults(...)` when the space semantics can
derive the sampler, diversity metric, and perturbation schedule.

## What CSA Maintains

CSA operates on three state objects:

- **The bank.** A bounded set of diverse elite candidates. Bank size is
  controlled by `bank_capacity`; a staged growth ceiling is supported through
  the update policy.
- **A perturbation schedule.** An explicit menu of variation operators split
  into *regular*, *initial*, and *mutation* families. Each family has a
  declared count of proposals produced per generation.
- **A distance-cutoff schedule.** A decaying threshold that defines when two
  candidates are "close" in the space's diversity metric. Reducing the cutoff
  shrinks the neighbourhood that counts as "already covered" by an existing
  bank entry.

The bank is not a replaceable generation. It is an elite archive that
accumulates diversity across the run; cutoff annealing controls how tightly
the archive packs.

## One Generation Cycle

A single CSA generation proceeds roughly as:

1. **Seed selection.** Pick bank entries to act as perturbation seeds.
2. **Perturbation.** Apply the scheduled operator counts to the seeds,
   drawing from the *regular* family in steady state and the *initial* family
   while the bank is still being seeded.
3. **Evaluation.** The generated proposals are handed off to the evaluator.
4. **Update.** Each new candidate is compared to the bank under the current
   cutoff:
    - **Local update** — if the candidate is close to an existing bank entry,
      it replaces that entry only if it is better.
    - **Far update** — if the candidate is far from every entry, it is added
      (or displaces the weakest entry if the bank is full), possibly modulated
      by clustering to avoid over-concentration.
5. **Annealing.** The distance cutoff contracts, tightening the "close"
   neighbourhood for the next generation.

The *mutation* family sits alongside the regular schedule and is used to
inject occasional diversity, particularly late in the run when the cutoff has
contracted.

## Local Refinement Feedback

When a kernel or evaluator refines a CSA proposal before evaluation, the
resulting `CandidateRefinement` can include the structured leaf paths that
changed. CSA proposal adaptation treats those explicit paths as authoritative
local-displacement feedback. It falls back to comparing the proposed and
evaluated candidates only when no refinement metadata is present.

Explicit empty path metadata means "no local displacement paths were reported",
not "infer them later". CSA therefore records no local displacement for that
outcome and avoids candidate comparison in that path. This feedback affects only
proposal adaptation; bank admission, scoring, evaluation accounting, and
checkpoint state remain governed by the evaluated records.

When proposal adaptation is enabled, CSA updates it only after a complete
generation has reached conclusive bank transitions. A proposal has positive
survival efficiency only if it remains in the final post-generation bank. The
signal is divided by `max(1, evaluation_count)` so additional logical evaluations
reduce the measured efficiency without introducing wall-clock nondeterminism.
Mutation and local-displacement leaf associations share one outcome's signal
rather than each receiving a copy of the full result. Numeric covariance treats
a successful displacement vector as one sample weighted by the same survival
efficiency.

Adaptive family sampling starts only after every configured mutation family has
at least one conclusive outcome. Before then, CSA emits the declared per-family
counts exactly and does not consume family-selection RNG. Proposal adaptation
remains disabled by default; fixed scheduling is therefore both the default and
the explicit baseline for comparisons.

That cold-start guarantee is limited to mutation-family counts and
family-selection RNG. Enabling adaptation routes structured mutations through
the explicit leaf planner even before leaf evidence exists. Its initial weights
are uniform, but this path is not candidate- or RNG-identical to the disabled
operator-native path.

Treat proposal adaptation as experimental rather than as a generally stronger
schedule. A preregistered equal-budget development panel did not justify a
default or named-preset promotion. Adaptation preserved final-bank diversity in
more pairs, while fixed scheduling produced better best-observed objective values
in more pairs. Validate the tradeoff on the target problem before enabling it.

For the broader execution boundary, see
[Candidate Refinement](candidate-refinement.md).

## How CSA Differs From GA And DE

CSA shares the ask/tell contract with the other population optimizers in
`variopt`, but its state object is qualitatively different:

- A genetic algorithm replaces a whole population each generation. CSA
  incrementally updates an elite archive.
- Differential evolution uses one mutation recipe (`mutation_range`,
  `recombination_probability`) applied uniformly. CSA uses an explicit
  perturbation menu with per-family counts.
- Both GA and DE treat diversity implicitly (niching or fitness pressure).
  CSA treats diversity as the primary state: the bank encodes it directly and
  the cutoff schedule decides how aggressively to preserve it.

The payoff is that CSA tends to keep spread across multimodal or
topologically structured landscapes longer than GA or DE would at comparable
evaluation budgets. The cost is more configuration surface.

## Current Presets

- `variopt` — current house defaults. Crowd-aware far-update, a staged bank
  capacity ceiling, and a modest seed count. Adaptive bank growth and
  cluster-aware admission remain available policies, but they are disabled by
  default.
- `joung_2018` — literature-aligned baseline matching the generic CSA paper,
  modulo the deltas listed below.

Both presets fill the same profile slot set from the same
`CSAProfileDefaults`. They do *not* pin the perturbation schedule — that
comes from `derive_csa_defaults(...)` or an explicit override.

## Going Deeper

- [Customize an Optimizer Profile](../guides/customize-optimizer-profile.md)
  — the task-oriented guide for overriding presets and profile slots.
- [Presets and Contracts](../reference/presets-and-contracts.md) — the
  supported preset surface and slot catalog.
