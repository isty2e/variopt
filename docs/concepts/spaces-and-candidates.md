# Spaces and Candidates

`variopt` treats the candidate as a canonical runtime value inside exactly one
[`SearchSpace`][variopt.SearchSpace]. For the one-line definition of every
core term mentioned below, see the [Glossary](../reference/glossary.md).

The space is not just metadata — it actively participates at every stage of the
optimization pipeline.

## What The Space Does

| Stage | How the space participates |
| --- | --- |
| **Ingress** | Converts raw inputs into canonical candidate values |
| **Validation** | Rejects out-of-bounds or wrong-type candidates with clear errors |
| **Sampling** | Knows how to draw uniform or scale-aware random candidates |
| **Diversity** | Exposes topology and scale metadata that diversity metrics can consume |
| **Local search** | Exposes leaf paths and leaf spaces that structured kernels can use |
| **Result** | Candidates come back in the same structured form they went in |

This means you do not need to write your own coordinate transforms, distance
functions, or decode logic when using a structured space. The optimizer,
evaluator, and kernel all see the same typed candidate throughout.

## Built-In Families

- **scalar:** [`RealSpace`][variopt.RealSpace],
  [`IntegerSpace`][variopt.IntegerSpace],
  [`CategoricalSpace`][variopt.CategoricalSpace]
- **composites:** [`TupleSpace`][variopt.TupleSpace],
  [`RecordSpace`][variopt.RecordSpace],
  [`ArraySpace`][variopt.ArraySpace]
- **permutation:** [`PermutationSpace`][variopt.PermutationSpace]

Scalar spaces have optional `scale` parameters (`"log"` for `RealSpace`)
that affect sampling and normalization transparently.

Composite spaces compose leaf spaces into richer structures. A `RecordSpace`
produces `RecordCandidate` mapping values with named fields; a `TupleSpace`
produces `tuple` candidates; an `ArraySpace` produces fixed-length
homogeneous sequences.

## Custom Spaces

Any class that implements the [`SearchSpace`][variopt.SearchSpace] protocol
works with `Problem`, `Study`, and all evaluators. For structured spaces
that want geometry-aware fast paths, implement the
[`CompiledStructuredGeometryProvider`][variopt.spaces.CompiledStructuredGeometryProvider]
sidecar protocol — see the [API reference](../reference/api/spaces.md).
