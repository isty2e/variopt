---
id: pn-crz0
status: closed
deps: [pn-9euu, pn-h0tc]
links: []
created: 2026-06-30T22:47:53Z
type: task
priority: 1
assignee: isty2e
parent: pn-qiwp
tags: [candidate-refinement, var-ln22]
---
# Edge-hunt refinement accounting and no-op overhead

Logical ID: var-ln22. Run an adversarial campaign over refinement accounting, changed-path metadata, and no-op overhead after implementation tickets land.

## Acceptance Criteria

Campaign covers at least three consecutive all-green rounds after any fix; edge tests include no-op, duplicate/invalid paths, batch reordering, async resume, checkpoint/serialization, and CSA adaptation fallback; overhead-sensitive no-op paths avoid needless metadata construction.


## Notes

**2026-07-01T00:35:09Z**

Edge-hunt campaign completed with adversarial coverage and one runtime fix.

Round 1 candidates: no-op refinement allocation, duplicate/invalid changed paths, explicit-empty CSA paths, checkpoint serialization, exact-async resume ordering, public docs/import drift. Result: all-green after adding regression coverage for bool path rejection, explicit-empty no-fallback, disabled/empty path non-materialization, checkpoint adaptation-vs-provenance boundary, and exact-async resume outcome-feedback ordering.

Round 2 candidates: candidate equality that raises during comparison, candidate equality that returns truthy non-scalar values, terminal-surface normalization parity, outcome validation parity, CSA fallback with changed candidate plus explicit empty paths, path invalidity under type pressure. Result: defect found. Existing validation converted equality results through bool(...) and accepted truthy non-scalar results such as [True]. Fixed by moving scalar candidate equality validation into require_scalar_candidate_equality and using it from EvaluationOutcome and terminal surfaces.

Round 3 candidates: disabled policy no-op overhead, empty observation batch no-op overhead, nondominated direct construction validation, type-checkable invalid path inputs, checkpoint payload boundary, exact-async resumed metadata order. Result: all-green after adding tests; no runtime changes required.

Round 4 candidates: full-suite regression, lint, type check, strict docs build, whitespace check, local artifact cleanup. Result: all-green. Stop condition met: after the Round 2 fix, three consecutive green rounds completed.

Verification: uv run pytest -> 623 passed, 16 skipped, 3 known joblib warnings; uv run pytest tests/core/test_runtime_artifacts.py tests/csa/test_csa_proposal.py tests/csa/test_csa_checkpoint.py tests/study/test_study_exact_async.py -> 103 passed; uv run ruff check src tests -> clean; uv run basedpyright -> 0 errors, 0 warnings; uv run mkdocs build --strict -> built successfully with upstream Material warning; git diff --check -> clean.
