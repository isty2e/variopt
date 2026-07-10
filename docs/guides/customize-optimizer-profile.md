# Customize an Optimizer Profile

This guide is for users who have already run `CSAOptimizer.from_space_defaults`
and need to change one or more components behind it.

The scope is `CSAProfile`. The same shape applies to `DEProfile` and `GAProfile`
on a smaller scale (one section at the end covers the differences).

## When To Customize

- **Bare `from_space_defaults(...)`** — stay here when the preset knobs are
  acceptable. Override nothing.
- **One-axis override** — when you need one or two non-default components but
  want everything else to stay ergonomic and space-derived. Pass a partial
  `CSAProfile` into `from_space_defaults`.
- **Full-custom profile** — when you want to control the perturbation schedule
  or the full policy set explicitly, or when you want to skip space-derived
  defaults entirely (for example, a non-structured space or a strict
  reproducibility target). Build `CSAOptimizer(...)` directly.

## Three Axes, Briefly

CSA configuration has three axes. The stable public summary lives in
[Presets and Contracts][presets-and-contracts]. For the purposes of this guide:

1. **`CSAProfile(preset=...)`** — boundary-level policy defaults (clustering,
   growth, cutoff schedule, acceptance, update, etc.). The preset does *not*
   define a perturbation schedule.
2. **`derive_csa_defaults(..., style=...)`** — space-derived execution defaults
   (sampler, diversity metric, perturbation schedule).
3. **Explicit keyword overrides** passed to `CSAOptimizer.from_space_defaults`
   or directly to `CSAOptimizer(...)`.

Override precedence is:

1. explicit keyword overrides
2. fields already present on the `CSAProfile`
3. space-derived defaults

## Pattern 1: Narrow Override Through `from_space_defaults`

Turn on a single policy without losing space-derived perturbation schedule,
sampler, or diversity metric.

```python
from variopt import RealSpace, RecordSpace
from variopt.algorithms.population.csa import (
    CSAClusteringPolicy,
    CSAOptimizer,
    CSAProfile,
)


space = RecordSpace(
    x=RealSpace(low=-5.0, high=5.0),
    y=RealSpace(low=-5.0, high=5.0),
)

optimizer = CSAOptimizer.from_space_defaults(
    space=space,
    bank_capacity=16,
    profile=CSAProfile(
        clustering_policy=CSAClusteringPolicy(enabled=True),
    ),
)
```

What actually happens:

- `CSAProfile(clustering_policy=...)` leaves every other slot as `None`, so
  preset defaults fill them in.
- `from_space_defaults` sees `profile.perturbation_schedule is None` and
  fills it from `derive_csa_defaults(space, style="variopt")`.
- The sampler and diversity metric also come from the space.

This is the pattern to reach for first. It keeps the house defaults except for
the one axis you meant to change.

## Pattern 2: Full Custom Perturbation Schedule

When the space-derived perturbation schedule is not what you want — different
operator counts, different family composition, or a domain-specific operator
— build the schedule yourself and pass it in.

```python
from variopt import IntegerSpace
from variopt.algorithms.population.csa import (
    BoundedMutation,
    CSAOptimizer,
    CSAPerturbationSchedule,
    CSAPerturbationSpec,
    CSAProfile,
    RandomResetMutation,
    UniformCrossover,
)


space = IntegerSpace(low=0, high=31)

schedule = CSAPerturbationSchedule(
    regular_family=(
        CSAPerturbationSpec(UniformCrossover(space=space), count=4),
    ),
    initial_family=(
        CSAPerturbationSpec(UniformCrossover(space=space), count=2),
    ),
    mutation_family=(
        CSAPerturbationSpec(BoundedMutation(space=space), count=3),
        CSAPerturbationSpec(RandomResetMutation(space=space), count=1),
    ),
)

optimizer = CSAOptimizer.from_space_defaults(
    space=space,
    bank_capacity=16,
    profile=CSAProfile(perturbation_schedule=schedule),
)
```

The `from_space_defaults` entry point still fills the sampler and diversity
metric from space-derived defaults. Only the perturbation schedule is
replaced.

The factory helpers `CSAProfile.variopt(...)` and `CSAProfile.joung_2018(...)`
are equivalent to passing `preset=` and an explicit `perturbation_schedule`
through the normal constructor — they mainly exist to force the schedule to
be declared up front for reproducibility.

## Space-Derived Default Overrides

`CSAOptimizer.from_space_defaults(...)` derives three components from the
space: `sampler`, `diversity_metric`, and `perturbation_schedule`. Any of
these can be overridden as a keyword argument without building a full custom
profile.

```python
from variopt import IntegerSpace
from variopt.algorithms.population.csa import CSAOptimizer
from variopt.diversity import StructuredSpaceDiversityMetric
from variopt.sampling import SearchSpaceSampler


space = IntegerSpace(low=0, high=63)

optimizer = CSAOptimizer.from_space_defaults(
    space=space,
    bank_capacity=12,
    sampler=SearchSpaceSampler(space=space),
    diversity_metric=StructuredSpaceDiversityMetric(space=space),
)
```

Use this layer when the space-derived defaults are almost right but the
sampler or the diversity metric needs replacing — for example, a biased
sampler that concentrates initial draws in a known promising region, or a
diversity metric with non-default leaf weights.

## Pattern 3: Build `CSAOptimizer` Directly

Use the direct constructor when `from_space_defaults` does not apply:

- the space is not a `StructuredSearchSpace` (custom `SearchSpace`)
- you want a specific diversity metric, not the space-derived one
- you want to pin every component for strict determinism

```python
from variopt import IntegerSpace
from variopt.algorithms.population.csa import (
    BoundedMutation,
    CSAAcceptancePolicy,
    CSABankUpdatePolicy,
    CSACutoffSchedule,
    CSAOptimizer,
    CSAPerturbationSchedule,
    CSAPerturbationSpec,
    CSAProfile,
)
from variopt.diversity import StructuredSpaceDiversityMetric


space = IntegerSpace(low=0, high=99)

schedule = CSAPerturbationSchedule(
    mutation_family=(
        CSAPerturbationSpec(BoundedMutation(space=space), count=3),
    ),
)

profile: CSAProfile[int] = CSAProfile(
    preset="variopt",
    perturbation_schedule=schedule,
    cutoff_schedule=CSACutoffSchedule(
        initial_distance_cutoff=4.0,
        reduction_factor=0.97,
    ),
    acceptance_policy=CSAAcceptancePolicy(initial_temperature=1.0),
    update_policy=CSABankUpdatePolicy(
        minimum_significant_score_gap_ratio=0.01,
        local_update_mode="normal",
        far_update_mode="crowding_aware",
    ),
)

optimizer = CSAOptimizer(
    space=space,
    diversity_metric=StructuredSpaceDiversityMetric(space=space),
    bank_capacity=8,
    profile=profile,
    random_state=0,
)
```

Compared to `from_space_defaults`, direct construction:

- requires an explicit `diversity_metric`
- accepts any `SearchSpace`, not only a `StructuredSearchSpace`
- never fills a missing `perturbation_schedule` — the profile must carry one
  or `CSAProfile.resolve()` raises `ValueError`

## The `CSAProfile` Slot Catalog

For a one-line description of every `CSAProfile` slot together with the
relevant API page, see
[Presets and Contracts][presets-and-contracts].

## Picking A Preset As A Starting Point

- `preset="variopt"` — current house defaults. Adaptive clustering-aware
  far-update, staged bank capacity ceiling of 24, modest seed count. Start
  here unless you need literature alignment.
- `preset="joung_2018"` — literature-aligned baseline. Matches what the
  generic CSA paper reports.

Both presets fill the same slot set from the same `CSAProfileDefaults`, so
the override mechanics in this guide are identical between them.

## A Note On DE And GA

`DEProfile` and `GAProfile` follow the same *Profile / *ResolvedProfile
split, but they are far simpler than `CSAProfile`:

- neither carries a `preset=` axis
- neither has a `from_space_defaults` entry point (the optimizers take the
  profile as a plain constructor argument)
- both resolve to flat boundary-level dataclasses

So for DE and GA, customization means building the profile object directly.

```python
from variopt.algorithms.population import (
    DEProfile,
    DifferentialEvolutionOptimizer,
    GAProfile,
    GeneticAlgorithmOptimizer,
)

de_profile = DEProfile(
    mutation_range=(0.4, 0.9),
    recombination_probability=0.8,
    n_cross=2,
)

ga_profile = GAProfile(
    tournament_size=3,
    crossover_probability=0.85,
    mutation_probability=0.15,
    elite_count=2,
)
```

## Related Reading

- [Concepts / CSA](../concepts/csa.md)
- [Presets and Contracts][presets-and-contracts]

[presets-and-contracts]: ../reference/presets-and-contracts.md
