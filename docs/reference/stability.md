# Stability Policy

This page defines what `variopt` considers "supported" public surface, what
counts as a breaking change, and how deprecations are handled.

## Supported Public Surface

The supported public surface is intentionally narrower than the set of
importable modules. It consists of:

### Facade modules

- `variopt`
- `variopt.spaces`
- `variopt.sampling`
- `variopt.diversity`
- `variopt.evaluators`
- `variopt.study`
- `variopt.artifacts`
- `variopt.algorithms.population`
- `variopt.algorithms.local_search`

Every name exported through the `__all__` of one of these modules is part of
the supported surface. See [API Surface](api.md) for the generated reference
pages.

### Advanced CSA policy types

`variopt.algorithms.population.csa` additionally exposes the concrete CSA
profile, schedule, and policy dataclasses used when overriding individual
`CSAProfile` slots. These types are public and supported, but they form a
narrower advanced contract aimed at users who already work against
`CSAProfile`. See [Customize an Optimizer
Profile](../guides/customize-optimizer-profile.md).

### What is *not* supported

- Deep submodule paths not listed above, even when importable.
- Names prefixed with a single underscore.
- Internal modules re-exported only for implementation convenience.
- Internal test helpers under `tests/`.

## What Counts As A Breaking Change

The following changes to the supported surface are breaking:

- Removing or renaming any public name.
- Removing or renaming a public callable's parameter, or changing the order of
  positional-only parameters.
- Narrowing a parameter's accepted type, or widening a return type in a way
  that breaks existing type-checked callers.
- Removing a preset (`variopt`, `joung_2018`) or changing its documented
  invariants in a way that alters optimization outcomes beyond numerical
  noise. Additive preset knobs that preserve documented behaviour are *not*
  breaking.
- Changing the contract of `Study.optimize(...)` / `Study.run(...)` return
  types, including the shape of `RunResult` and `RunReport`.
- Removing or renaming a public protocol (`EvaluationProtocol`, `Kernel`,
  `Evaluator`, `RunMethod`, `Objective`) or changing its required methods.

The following are *not* treated as breaking:

- Adding a new public name.
- Adding a new optional keyword argument with a documented default that
  preserves prior behaviour.
- Internal refactors that do not change any documented input or output.
- Changes to docstrings, examples, or documentation pages.

## Deprecation Cadence

- Deprecated names raise `DeprecationWarning` for at least one minor release
  before removal.
- The deprecation message should name the replacement or link to migration
  guidance.
- Deprecations are recorded in [CHANGELOG.md](../changelog.md) under
  `### Deprecated` and again under `### Removed` when the removal lands.

## Versioning

The project uses semantic-version-shaped identifiers but applies them in two
phases:

### Pre-1.0 (`0.x`)

- Breaking changes are allowed between minor releases (`0.1 → 0.2`) when
  justified by design correction, ontology tightening, or safety.
- Every breaking change is documented in the changelog, ideally with a
  migration note.
- Deprecation cadence still applies when feasible; when it is not (for
  example, a protocol whose accidental shape was actively misleading), the
  changelog must explain why a direct break was chosen.

### Post-1.0

- Breaking changes require a major version bump.
- Deprecation cadence is mandatory for renames and removals of anything in
  the supported surface.

A `1.0` release will only be cut once the facade surface, preset semantics,
and evaluation-protocol contract have stabilised across at least one external
usage cycle.

## Tracking Changes

- [CHANGELOG.md](../changelog.md) is the authoritative list of user-visible
  changes.
- Breaking changes are called out in a `### Breaking` subsection of the
  relevant release.
- The docs build is gated by `mkdocs build --strict` in CI so that renamed or
  removed public names surface as autoref failures before release.
