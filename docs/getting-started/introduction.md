# Introduction

`variopt` is a typed optimization library for structured search spaces.

Most optimization libraries expect you to flatten your problem into a numeric
vector and decode it yourself. `variopt` keeps the structure you declare —
named fields, integer ranges, categorical choices, permutations — all the way
from sampling through to the result.

The library splits the optimization pipeline into a few explicit roles:

- [`SearchSpace`][variopt.SearchSpace] — what a valid candidate looks like
- [`Problem`][variopt.Problem] — how to evaluate a candidate
- [`RunMethod`][variopt.RunMethod] — the optimizer (CSA, DE, GA, etc.)
- [`Evaluator`][variopt.Evaluator] — how evaluation runs (serial, parallel, async)
- [`Study`][variopt.Study] — wires everything together into one `optimize()` call

You can swap any one of these without touching the others. For example,
switching from sequential to parallel evaluation means replacing only the
`Evaluator` — the optimizer, problem, and study call stay the same.

## What To Read Next

- install and extras:
  [Installation](installation.md)
- smallest runnable example:
  [Quickstart](quickstart.md)
- conceptual model:
  [Optimization Model](../concepts/optimization-model.md)
