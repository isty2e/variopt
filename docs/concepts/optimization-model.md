# Optimization Model

The semantic root is:

- one [`SearchSpace`][variopt.SearchSpace]
- one [`Problem`][variopt.Problem]
- one [`RunMethod`][variopt.RunMethod]
- one [`Evaluator`][variopt.Evaluator]
- optional [`Kernel`][variopt.Kernel]
- one [`Study`][variopt.Study] orchestrating the run

For one-line definitions of each term, see the
[Glossary](../reference/glossary.md). This page focuses on *why* the model
splits these roles.

## Why The Model Is Split

Each role owns one thing only:

| Role | Owns | Does not own |
| --- | --- | --- |
| `SearchSpace` | what a valid candidate looks like | how to evaluate it |
| `Problem` | evaluation meaning (objective or protocol) | how to execute it |
| `RunMethod` | persistent search memory (ask/tell) | execution mechanics |
| `Kernel` | bounded local-search episodes | cross-run search state |
| `Evaluator` | execution mechanics (serial, parallel, async) | search strategy |
| `Study` | orchestration wiring | any of the above |

## What This Buys You

When roles are collapsed (as in many optimization libraries that bundle
"optimizer + evaluator + space" into one object), changing one concern forces
changes in others:

- **Switching from serial to parallel evaluation** in a collapsed design
  often means a different optimizer class or a different constructor
  signature. In `variopt`, you replace only the `Evaluator` — the
  `RunMethod`, `Problem`, and `Study.optimize(...)` call stay identical.

- **Adding local search** in a collapsed design typically means a new
  optimizer variant. In `variopt`, you pass a `Kernel` to the same `Study`
  and the same `RunMethod` — the kernel runs bounded refinement episodes
  without touching the global search strategy.

- **Structured spaces** in a collapsed design often require the user to
  flatten candidates into a numeric vector and decode them manually. In
  `variopt`, the `SearchSpace` preserves structure all the way through
  sampling, diversity, and result extraction. The optimizer never sees a
  flat vector unless the space is one.

Candidate refinement is another boundary case. It is execution-side provenance
between proposal and evaluation record, carried by successful attempt metadata
and terminal reports rather than by the evaluation protocol itself. See
[Candidate Refinement](candidate-refinement.md) for the proposed, refined,
evaluated, and accepted candidate vocabulary.

The split matters most when a problem combines several of these concerns at
once: a structured space, a parallel evaluator, and a local-search kernel.
Each component stays in its lane, and the `Study` wires them together.
