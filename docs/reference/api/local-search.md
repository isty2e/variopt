# `variopt.algorithms.local_search`

These public kernels define bounded local-search semantics. For method choice,
see [Choose a Local Optimization Method](../../guides/local-optimization-methods.md).
For evaluator-owned episode placement, see
[Run Local Search in Evaluator Workers](../../guides/request-local-episodes.md).
Episode-capability methods on these kernels are called by `Study`; configure
them through the public kernel constructors rather than constructing internal
episode objects.

::: variopt.algorithms.local_search
