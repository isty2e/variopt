# Reuse a Problem in Joblib Workers

Use Joblib worker-session transport when a synchronous run repeatedly sends the
same large, immutable `Problem` to a worker pool. Keep the default `per_request`
transport for small contexts, short runs, or one-request calls.

## Enable worker-session transport

```python
from variopt.evaluators import JoblibEvaluator

evaluator = JoblibEvaluator(
    n_jobs=4,
    backend="loky",
    problem_transport="worker_session",
)
```

The session binds the exact `Problem` instance for one `Study.run(...)`,
`Study.optimize(...)`, or `Study.step(...)` scope. A custom kernel cannot
substitute another problem inside that scope.

## Keep evaluation behavior request-derived

Each loky process decodes an independent problem instance and retains at most
one current generation. A worker may replace that generation when concurrent
runs share the process pool, or reconstruct it after worker replacement.

Observable evaluation behavior must therefore follow from the request and the
immutable snapshot. Process-local memoization is suitable. Mutable random
streams, counters, wall-clock-dependent state, and cross-worker coordination
are not restart-equivalent unless their observable results are request-derived.

Threading backends and effective single-worker execution use direct shared
references and do not create snapshot transport. Checkpoints never contain the
worker session; resuming a run creates a new generation from the
coordinator-owned problem. Concurrent sessions remain isolated, but frequent
generation switching can reduce cache reuse.

## Treat the snapshot as runtime transport

The problem snapshot is serialized with `cloudpickle` into an owner-only
temporary directory and exposed to workers as a read-only NumPy memory map. It
is trusted runtime data for related processes, not a portable or long-term file
format.

Normal and exceptional scope closure removes the transport. Abrupt coordinator
termination can leave an artifact in the operating system temporary directory
for ordinary temporary-file cleanup to reclaim.

## Measure whether reuse pays off

Do not select `worker_session` from problem size alone. The number of evaluator
calls, requests per call, worker count, objective cost, and Joblib's own batching
all affect whether the one-time snapshot setup is recovered.

For one synthetic reference measurement on Apple arm64 with Python 3.13,
Joblib 1.5.3, two loky workers, and seven repetitions, a cheap objective
carrying an 8 MiB immutable context produced these median wall times for 64
evaluations:

| Requests per evaluator call | Evaluator calls | `per_request` | `worker_session` |
| ---: | ---: | ---: | ---: |
| 1 | 64 | 0.940 s | 1.018 s |
| 4 | 16 | 0.414 s | 0.289 s |
| 16 | 4 | 0.291 s | 0.065 s |

The same 32-evaluation, four-request-per-call workload took 0.115 s versus
0.123 s with a 1 KiB context, 0.120 s versus 0.126 s with a 1 MiB context, and
0.226 s versus 0.129 s with an 8 MiB context. Setup overhead made session reuse
slower for small contexts or single-request calls, while a large repeated
context made it substantially faster.

The transport oracle for 64 evaluations in 16 calls reduced coordinator
serialization of the 8 MiB context from 64 serializations, approximately
512 MiB in aggregate, to one 8 MiB snapshot. A representative process-memory
profile also reduced the observed worker-process peak from about 451 MiB to
170 MiB. These values are workload-specific evidence, not portable capacity
guarantees.

Keep the session open across the full synchronous run when possible. Repeatedly
opening one-call sessions pays setup repeatedly and preserves less of the
benefit. There is intentionally no automatic size threshold because the
measured crossover changes with evaluator-call shape.

See [Choose an Evaluator](choose-an-evaluator.md) for the broader backend choice
and [Evaluator Contracts](../reference/evaluator-contracts.md) for lifecycle and
failure guarantees.
