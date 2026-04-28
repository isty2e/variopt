# Presets and Contracts

This page is the stable public summary of optimizer preset contracts.

The practical rule is:

- use `CSAProfile(preset="variopt")` for the current house preset
- use `CSAProfile(preset="joung_2018")` for the current literature-aligned path
- prefer `CSAOptimizer.from_space_defaults(...)` when you want the ergonomic
  entry point

See the task-oriented overrides guide in
[Customize an Optimizer Profile](../guides/customize-optimizer-profile.md).

## CSA Preset Axes

CSA construction has three independent axes:

1. `CSAProfile(preset=...)`
2. `derive_csa_defaults(..., style=...)`
3. explicit keyword overrides passed to `CSAOptimizer.from_space_defaults(...)`

`CSAProfile(preset=...)` fills profile-level policy defaults such as seed
count, cutoff schedule, acceptance, update, clustering, growth, restart, and
cycle defaults. It does not define a perturbation schedule by itself.

`derive_csa_defaults(..., style=...)` derives execution-facing defaults from a
structured search space: sampler, diversity metric, and perturbation schedule.

`CSAOptimizer.from_space_defaults(...)` wires those axes together. Explicit
keyword overrides win over fields already present on the profile, and profile
fields win over space-derived defaults.

## CSA Preset Meanings

### `variopt`

Use this when you want the ergonomic built-in CSA path and are not aiming for
literature alignment.

Current meaning:

- `CSAProfile(preset="variopt")`
- `derive_csa_defaults(..., style="variopt")`
- regular family: `UniformCrossover x2`
- initial family: `UniformCrossover x2`
- mutation family: `BoundedMutation x2 + RandomResetMutation x1`
- profile-level far-update default: `far_update_mode="crowding_aware"`
- profile-level staged bank-growth default: `max_bank_capacity=24`

This is the default used by `CSAOptimizer.from_space_defaults(...)` when no
special literature-aligned profile is supplied, and it is also the default
`CSAProfile` preset when no preset is named explicitly.

### `joung_2018`

Use this when you want a literature-aligned generic CSA configuration rather
than the variopt house configuration.

Current meaning:

- profile defaults aligned to representable Joung-2018 generic CSA settings
- when used through `from_space_defaults(...)`, a literature-aligned
  `10/10/10` perturbation schedule

This is a literature-aligned preset, not a parity preset. In particular, the
current engine does not reproduce the original first-bank and new-entry
crossover control flow exactly. It approximates that part of the paper through
the existing `initial_new_bank_cut` boundary and current generation semantics.

## `CSAProfile` Slot Catalog

Each row lists one `CSAProfile` slot, the type it accepts, and a one-line
semantic. Slots left at `None` on the `CSAProfile` inherit from the selected
preset via `profile_defaults_for_preset(preset)`. `perturbation_schedule` is
the only slot that is never filled from the preset — it must come from the
caller, from `derive_csa_defaults(...)`, or from
`CSAOptimizer.from_space_defaults(...)`.

### Generation slots

| Slot | Type | What it controls |
| --- | --- | --- |
| `perturbation_schedule` | [`CSAPerturbationSchedule`][variopt.algorithms.population.csa.CSAPerturbationSchedule] | Regular, initial, and mutation family operators used to generate children. |
| `proposal_policy` | [`CSAProposalPolicy`][variopt.algorithms.population.csa.CSAProposalPolicy] | History-aware adaptive proposal weighting (family, leaf, covariance). Disabled by default. |

### Banking slots

| Slot | Type | What it controls |
| --- | --- | --- |
| `clustering_policy` | [`CSAClusteringPolicy`][variopt.algorithms.population.csa.CSAClusteringPolicy] | Cluster-aware bank admission (disabled by default). |
| `growth_policy` | [`CSABankGrowthPolicy`][variopt.algorithms.population.csa.CSABankGrowthPolicy] | Adaptive bank growth based on energy gap. |
| `update_policy` | [`CSABankUpdatePolicy`][variopt.algorithms.population.csa.CSABankUpdatePolicy] | Near/far admission logic and the significant-score-gap threshold. |
| `max_bank_capacity` | `int \| None` | Staged bank ceiling used for step-wise growth. `None` keeps the bank fixed. |

### Progression slots

| Slot | Type | What it controls |
| --- | --- | --- |
| `cutoff_schedule` | [`CSACutoffSchedule`][variopt.algorithms.population.csa.CSACutoffSchedule] | Distance-cutoff initialization, decay, and recovery. |
| `refresh_policy` | [`CSARefreshPolicy`][variopt.algorithms.population.csa.CSARefreshPolicy] | Refresh and restart behavior at run boundaries. |
| `restart_lite` | `bool \| None` | Whether lightweight restarts are enabled after convergence events. |
| `cycle_limit` | `int \| None` | Maximum number of CSA cycles before staged lifecycle actions fire. |
| `initial_new_bank_cut` | `int \| None` | Initial cutoff applied before adaptive cutoff updates begin. |

### Scoring slots

| Slot | Type | What it controls |
| --- | --- | --- |
| `acceptance_policy` | [`CSAAcceptancePolicy`][variopt.algorithms.population.csa.CSAAcceptancePolicy] | Temperature schedule for probabilistic score acceptance. |
| `score_model` | [`CSAScoreModel`][variopt.algorithms.population.csa.CSAScoreModel] | Objective-to-CSA-score mapping, including biased and adaptive potentials. |

### Selection slots

| Slot | Type | What it controls |
| --- | --- | --- |
| `seed_count` | `int \| None` | Number of seeds tracked in each CSA generation. |
| `random_seed_mode` | `int \| None` | Legacy-compatible seed-selection mode identifier. |
| `weighted_partner_selection` | `bool \| None` | Whether partner sampling is weighted by CSA scores. |

## `DEProfile` And `GAProfile`

`DEProfile` and `GAProfile` are flat boundary dataclasses. They have no preset
axis, no space-derived defaults path, and resolve to their own
`*ResolvedProfile` counterparts with the same fields.

See
[`DEProfile`][variopt.algorithms.population.DEProfile] and
[`GAProfile`][variopt.algorithms.population.GAProfile] for the full field list.
