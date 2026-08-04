# Evaluator-Owned Local Search

Evaluator-owned local search changes where a bounded episode executes, not
which optimization algorithm the episode implements.

## Four independent boundaries

| Concept | Question it answers |
| --- | --- |
| Objective evaluation | What value or payload does one candidate produce? |
| Request-local kernel episode | Which candidates should one proposal-local search evaluate before returning one top-level attempt? |
| Evaluator backend | Where does evaluation work execute? |
| Execution model | When and in what order does the run method assimilate completed work? |

A request-local episode may make several serial objective calls, but it returns
exactly one top-level `EvaluationSuccess` or `EvaluationFailure` slot for its
source proposal. Its actual objective-call cost is reported through
`evaluation_count`.

```text
Coordinator                  Evaluator workers                 Coordinator

RunMethod.ask()
      |
      v
validate requests
reserve hard budget --------> bounded serial episode
                              (one per proposal)
                                      |
                                      v
                              restore request order ----------> validate attempts
                                                                settle actual cost
                                                                refund unused capacity
                                                                       |
                                                                       v
                                                              RunMethod.tell_attempts()
```

The coordinator continues to own optimizer state, hard-budget accounting, and
assimilation order. Workers own only the bounded episode they receive. Inner
objective calls remain serial so the evaluator remains the outer parallelism
owner and nested worker pools are avoided.

## Placement and semantics

Eligible synchronous execution can move an episode from the coordinator into a
`SequentialEvaluator` or `JoblibEvaluator`. If an eligibility condition is not
met, execution falls back to the coordinator path. The placement changes while
kernel behavior remains the same.

This does not make `sequential` and `sync_batch` equivalent. A run method that
commits a whole batch after evaluation can follow a different trajectory from a
method that updates after each proposal, even when the evaluator restores
request order.

## When Joblib helps

Joblib helps when a batch contains multiple eligible episodes and each episode
has enough objective work to amortize scheduling and transport. It usually
hurts when:

- `batch_size=1`
- objective and local-search bookkeeping are very cheap
- process serialization dominates the episode
- Python-level GIL-bound work uses the threading backend
- a short run repeatedly pays worker or worker-session setup

One development measurement on Apple arm64 with Python 3.13 and Joblib 1.5.3
used four workers and five measured repeats per cell. For GIL-releasing
synthetic episode work, both threading and loky were slower with no objective
delay, reached roughly 2.5x at 2 ms per objective call, and roughly 2.7x to 3.2x
at 10-50 ms. At a fixed 5 ms objective delay, one-proposal batches were slower,
while four- and eight-proposal batches were roughly 1.5x to 2.6x faster depending
on backend and worker count.

For Python-level GIL-bound work, threading stayed near break-even even around
10 ms, while loky reached roughly 2.7x in that synthetic case. A failure-heavy
workload and a stochastic workload preserved their semantic digests while
running roughly 2.1x to 2.7x faster. These are workload-specific observations,
not portable crossover thresholds.

Across the 28 measured comparison cells, timing-independent result, failure,
ordering, refinement, trace, final-state, and evaluation-accounting digests
matched the coordinator baseline.

Measure the real objective, candidate serialization cost, episode-length
distribution, batch size, backend, and worker count. A faster run with different
assimilation semantics is not an equivalent benchmark.

Use [Run Local Search in Evaluator Workers](../guides/request-local-episodes.md)
for configuration and [Evaluator Contracts](../reference/evaluator-contracts.md)
for the exact dispatch and accounting rules.
