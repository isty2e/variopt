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
homogeneous sequences with a positive declared length. Declaration metadata is
canonical: for example, `RealSpace` bounds are floats, while integer bounds
belong to `IntegerSpace`.

Built-in structured spaces also expose validated-candidate leaf traversal hooks.
These are operation-local fast paths for optimizers and kernels that already
validated the whole candidate before scanning or replacing many leaves. Public
leaf methods remain the safe boundary for ordinary caller code, and replacement
values still flow through their owning leaf-space validation rules.

For built-in composite spaces, CSA can batch distance calculations without
changing leaf geometry or the order in which distances are accumulated.
Small batches and admitted integer intervals whose differences exceed `int64`
keep the scalar calculation path. This does not
change candidate or checkpoint formats, and custom spaces and metric overrides
keep their own distance implementations.

## Numeric Limits

Numeric space constructors reject bounds that their sampling or distance
arithmetic cannot support. A nonconstant `RealSpace` needs a finite, positive
coordinate span: `high - low` on the linear scale, or `log(high) - log(low)`
on the log scale. Finite endpoints alone are not enough. For example,
`RealSpace(-1e308, 1e308)` overflows on subtraction and raises `ValueError`.

`IntegerSpace` bounds must fit NumPy `RandomState`'s default C-long sampling
dtype, given by `numpy.iinfo("l")`. This platform-dependent limit applies to
linear, log, and constant intervals. Distinct log bounds must also remain
distinct after float conversion and taking logarithms. Linear integer distances
subtract integers before converting to float, so values above `2**53` do not
lose their integer differences. Float coordinate projections can still round
large integers; they are not exact encodings of every candidate.

Arbitrary-precision bounds are outside the built-in numeric contract. Rescale
or reparameterize an unsupported interval, or provide a custom space with its
own sampling and distance arithmetic.

## Custom Spaces

Any class that implements the [`SearchSpace`][variopt.SearchSpace] protocol
works with `Problem`, `Study`, and all evaluators. For structured spaces
that want geometry-aware fast paths, implement the
[`CompiledStructuredGeometryProvider`][variopt.spaces.CompiledStructuredGeometryProvider]
sidecar protocol — see the [API reference](../reference/api/spaces.md).
Custom structured spaces that do not override validated-candidate leaf hooks
keep the conservative public traversal behavior, so their own validation and
candidate-conditioned topology contracts remain authoritative.
