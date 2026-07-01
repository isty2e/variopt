---
id: pn-h0tc
status: closed
deps: [pn-fo46, pn-0n2b]
links: []
created: 2026-06-30T22:47:53Z
type: task
priority: 1
assignee: isty2e
parent: pn-qiwp
tags: [candidate-refinement, var-g60y]
---
# Feed explicit refinement paths into CSA adaptation

Logical ID: var-g60y. CSA adaptation should use explicit refinement changed paths instead of inferring path changes from source/refined candidate comparison whenever metadata is available.

## Acceptance Criteria

CSA adaptation consumes explicit changed paths when present; fallback inference remains correct only where metadata is absent; no eager path inference happens for no-op refinement; tests cover categorical/numeric/mixed structured paths and disabled adaptation paths.


## Notes

**2026-07-01T00:07:19Z**

Implemented explicit refinement-path feedback for CSA adaptation.

Ontology split: EvaluationOutcome owns execution-side refinement metadata; RunMethod.tell remains record-only; RunMethod.tell_outcomes is the narrow optional assimilation hook for run methods that need execution metadata; CSA consumes only CandidateRefinement.changed_leaf_paths for proposal adaptation and leaves semantic records, banking, scoring, trace, checkpoint, and numeric covariance attribution unchanged.

Lean decisions: no new public DTO, no broad candidate-type expansion of RunMethod.tell, no trace payload expansion, no bank/checkpoint schema change, and no eager path inference when policy is disabled or explicit refinement paths are present.

Edge hunt findings: Study previously discarded EvaluationOutcome before RunMethod assimilation, so CSA could not observe refinement metadata. Fixed Study sync, exact async, and stale async paths to use tell_outcomes. Added adversarial coverage for explicit metadata precedence, fallback only when metadata is absent, empty explicit paths as authoritative no-op, disabled adaptation avoiding inference and metadata validation, categorical/numeric/mixed structured paths, misaligned explicit-path rejection, and Study-level outcome-aware assimilation.

Verification: uv run pytest -> 614 passed, 16 skipped, 3 known joblib warnings; uv run ruff check src tests -> clean; uv run basedpyright -> 0 errors, 0 warnings; uv run mkdocs build --strict -> built successfully with upstream Material warning; git diff --check -> clean.
