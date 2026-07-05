# `variopt.study`

The study facade exposes the orchestration objects users may import directly:
`Study`, `StudyExactAsyncStepSession`, `StudyExactAsyncStepResumeHandle`, and
`RunExecutionFailed`.

`Study` carries separate generic axes for evaluator/kernel payload attempts and
run-method feedback records. Exact-async step sessions and resume handles keep
the same split: suspended sessions store payload attempts, and `finish()`
materializes successful payloads into request-aligned records immediately before
calling `RunMethod.tell_attempts(...)`.

Use the root facade or `variopt.artifacts` for attempt and diagnostics artifacts
such as `EvaluationAttemptBatch`, `EvaluationSuccess`, `EvaluationFailure`,
`KernelDiagnostics`, and `KernelStatus`. Deep modules such as `variopt.kernel`
are importable implementation modules, not supported artifact facades.

::: variopt.study
