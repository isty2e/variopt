# API Surface

The supported public facade modules are:

- `variopt`
- `variopt.spaces`
- `variopt.sampling`
- `variopt.diversity`
- `variopt.evaluators`
- `variopt.study`
- `variopt.artifacts`
- `variopt.algorithms.population`
- `variopt.algorithms.local_search`

Deeper submodules may be importable, but they are not automatically stable
public contract.

API reference generation is tracked separately so this page stays aligned with
the supported facade surface instead of exposing every internal package by
accident.

## Generated Facade Reference

The generated API pages are limited to the supported facade modules:

- [variopt](api/variopt.md)
- [variopt.spaces](api/spaces.md)
- [variopt.sampling](api/sampling.md)
- [variopt.diversity](api/diversity.md)
- [variopt.evaluators](api/evaluators.md)
- [variopt.study](api/study.md)
- [variopt.artifacts](api/artifacts.md)
- [variopt.algorithms.population](api/population.md)
- [variopt.algorithms.population.csa](api/csa.md)
- [variopt.algorithms.local_search](api/local-search.md)

Those pages intentionally stop at the facade boundary. Importable deep
submodules are not automatically part of the stable public contract.
