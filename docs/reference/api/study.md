# `variopt.study`

The study facade exposes the orchestration objects users may import directly:
`Study`, `StudyExactAsyncStepSession`, `StudyExactAsyncStepResumeHandle`, and
`RunExecutionFailed`.

`Study` carries separate generic axes for evaluator/kernel payload attempts and
run-method feedback records. Exact-async step sessions and resume handles keep
the same split: suspended sessions store payload attempts, and `finish()`
materializes successful payloads into request-aligned records immediately before
calling `RunMethod.tell_attempts(...)`.

Exact-async resume handles are live evaluator/session handles, not durable
checkpoint artifacts. The built-in joblib async evaluator keeps suspended work
inside the same evaluator instance's process-local runtime state, so those
handles are not a crash-recovery or process-restart persistence format.

Use the root facade or `variopt.artifacts` for `EvaluationAttemptBatch`,
`EvaluationFailure`, `KernelDiagnostics`, and `KernelStatus`. Use
`variopt.artifacts` for artifact-construction names that are intentionally not
exported from the trimmed root facade, including `EvaluationSuccess`.
Deep modules such as `variopt.kernel` are importable implementation modules,
not supported artifact facades.

::: variopt.study
