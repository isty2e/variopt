# Checkpointing

`variopt` exposes explicit CSA state checkpointing through
[`CSAOptimizer.state_to_dict()`][variopt.algorithms.population.CSAOptimizer.state_to_dict]
and
[`CSAOptimizer.state_from_dict()`][variopt.algorithms.population.CSAOptimizer.state_from_dict].
See [Checkpoint and Resume a CSA Run](../guides/checkpoint-and-resume-csa.md)
for the save and restore procedure.

## Scope

The current contract covers CSA engine state only. It is an exact safe-boundary
checkpoint: resuming with the same optimizer configuration, seed, objective,
and execution model continues exactly from the saved boundary.

The payload is JSON-safe and intended for JSON or another structured
serialization format. The supported durable persistence surface is the explicit
`to_dict()` / `from_dict()` contract. Python pickle round trips are runtime
compatibility conveniences only and are not a cross-version or crash-recovery
checkpoint format.

A checkpoint does not contain the optimizer configuration that gives its state
meaning. Persist a
[`CSAConfigurationManifest`][variopt.algorithms.population.CSAConfigurationManifest]
beside the checkpoint and compare it before restoration. The manifest does not
cover caller-owned problem, objective, data, evaluator, environment, or
dependency identity.

See [Record CSA Configuration Provenance](../guides/csa-configuration-provenance.md)
for manifest creation and restore guards.

## Safe boundary

!!! warning "Safe boundary only"

    `state_to_dict()` only accepts states between CSA generation batches. A
    checkpoint state has:

    - no pending proposals
    - no active generation queue
    - no buffered generation observations waiting to commit
    - no reference-refresh pool in progress
    - no pending proposal attributions

    If one of these runtime domains is active, checkpointing raises `ValueError`
    instead of serializing a partial state.

## Persisted state

The checkpoint captures the authoritative optimizer memory needed for exact
continuation:

- RNG state
- bank and reference-bank contents
- growth and clustering state
- cutoff and stage progression state
- seed-selection state
- proposal adaptation statistics
- scoring state
- monotone proposal-id counter

## Excluded state

The checkpoint does not capture:

- live evaluator or worker state
- exact-async suspended sessions
- exact-async resume handles
- in-flight proposal batches
- `Study.run(...)` reports or `Study.optimize(...)` results
- trace or telemetry reducer state
- derived caches that can be recomputed from authoritative state

Evaluator-owned request-local local-search episodes fall under the live-worker
and in-flight-batch exclusions. Resuming creates fresh evaluator runtime state
and derives later proposal-local random streams from restored optimizer state.

Restored CSA state has no accumulated CSA trace reducer state. This does not
affect exact optimization continuation because tracing is diagnostic and the
checkpoint stores the authoritative state that determines future proposals and
acceptance decisions. Payloads that include CSA `trace_state` are rejected
rather than silently dropping the data.

## Candidate encoding

For CSA optimizers over `StructuredSearchSpace`, the optimizer supplies a
built-in recursive candidate codec. Other spaces require explicit candidate
serialization callbacks.

The structured codec is JSON-safe and bounded. Candidate payloads must be
acyclic and must not exceed the codec nesting-depth limit. Malformed in-memory
payloads are rejected before interpreter recursion failures occur.

## Terminal results

`RunReport`, `RunResult`, and `NondominatedRunSurface` are terminal result
objects, not optimizer checkpoints. They may carry candidate-refinement
provenance, but `variopt` does not define `to_dict()` / `from_dict()`
serialization for those terminal surfaces. Persistence of reports, traces, and
result summaries is caller-owned.

## Unsupported checkpoint modes

The current contract does not support:

- mid-step checkpoint and resume
- exact-async suspended-session checkpointing
- exact-async resume-handle crash recovery
- terminal report or result serialization
- generic `Study`-level persistence across arbitrary run methods

These modes require restoring evaluator-owned lifecycle state in addition to
optimizer memory.
