---
id: pn-qiwp
status: closed
deps: []
links: []
created: 2026-06-30T22:47:53Z
type: epic
priority: 1
assignee: isty2e
tags: [candidate-refinement, var-sbl2]
---
# First-class candidate refinement surface

Logical ID: var-sbl2. Establish candidate refinement as a first-class execution artifact without collapsing it into evaluation semantics. The final surface must preserve exact accounting, explicit changed-path metadata, CSA adaptation input clarity, and no-op hot-path efficiency.

## Acceptance Criteria

All child tickets are closed; refinement metadata is modeled, emitted, threaded through study assimilation, exposed at terminal surfaces, consumed by CSA adaptation, documented, and edge-hunted without avoidable no-op overhead.


## Notes

**2026-07-01T00:36:25Z**

Closed after all child tickets landed.

Final surface: CandidateRefinement is the first-class execution artifact; local-search kernels and evaluator paths can attach it through EvaluationOutcome; Study sync, exact-async, stale-async, run, and optimize surfaces preserve aligned refinement metadata; terminal surfaces expose record/observation-aligned refinements with compact no-metadata sentinels; CSA adaptation consumes explicit changed_leaf_paths when available and falls back only when refinement metadata is absent; docs now define proposed/source/refined/evaluated/accepted boundaries and limitations.

Hardening: edge campaign added coverage for no-op allocation, invalid/duplicate paths, explicit-empty no-fallback behavior, disabled/empty-batch non-materialization, batch reordering, exact-async resume, checkpoint serialization boundaries, terminal alignment, and non-scalar candidate equality. Runtime validation now rejects truthy non-scalar equality results instead of accepting bool(result).

Verification before close: latest full gate was uv run pytest -> 623 passed, 16 skipped, 3 known joblib warnings; uv run ruff check src tests -> clean; uv run basedpyright -> 0 errors, 0 warnings; uv run mkdocs build --strict -> built successfully with upstream Material warning; git diff --check -> clean.
